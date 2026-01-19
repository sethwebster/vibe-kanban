pub mod client;
pub mod jsonrpc;
pub mod normalize_logs;
pub mod review;
pub mod session;
use std::{
    collections::HashMap,
    env,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
};

/// Returns the Codex home directory.
///
/// Checks the `CODEX_HOME` environment variable first, then falls back to `~/.codex`.
/// This allows users to configure a custom location for Codex configuration and state.
pub fn codex_home() -> Option<PathBuf> {
    if let Ok(codex_home) = env::var("CODEX_HOME")
        && !codex_home.trim().is_empty()
    {
        return Some(PathBuf::from(codex_home));
    }
    dirs::home_dir().map(|home| home.join(".codex"))
}

use async_trait::async_trait;
use codex_app_server_protocol::{JSONRPCNotification, NewConversationParams, ReviewTarget};
use codex_core::{
    AuthManager, ThreadManager,
    config::{Config, ConfigOverrides},
    protocol::{
        AgentMessageEvent, ErrorEvent, Event, EventMsg, Op as CoreOp, RolloutItem, SessionSource,
        TokenUsageInfo, TurnContextItem,
    },
    RolloutRecorder,
};
use codex_protocol::{
    config_types::SandboxMode as CodexSandboxMode, protocol::AskForApproval as CodexAskForApproval,
};
use command_group::AsyncCommandGroup;
use derivative::Derivative;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use strum_macros::AsRefStr;
use tokio::process::Command;
use ts_rs::TS;
use workspace_utils::msg_store::MsgStore;

use self::{
    client::{AppServerClient, LogWriter},
    jsonrpc::JsonRpcPeer,
    normalize_logs::normalize_logs,
    session::SessionHandler,
};
use crate::{
    approvals::ExecutorApprovalService,
    command::{CmdOverrides, CommandBuildError, CommandBuilder, CommandParts, apply_overrides},
    env::ExecutionEnv,
    executors::{
        AppendPrompt, AvailabilityInfo, ExecutorError, ExecutorExitResult, SlashCommand,
        SpawnedChild, StandardCodingAgentExecutor,
        codex::{jsonrpc::ExitSignalSender, normalize_logs::Error},
    },
    stdout_dup::create_stdout_pipe_writer,
};

/// Sandbox policy modes for Codex
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TS, JsonSchema, AsRefStr)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum SandboxMode {
    Auto,
    ReadOnly,
    WorkspaceWrite,
    DangerFullAccess,
}

/// Determines when the user is consulted to approve Codex actions.
///
/// - `UnlessTrusted`: Read-only commands are auto-approved. Everything else will
///   ask the user to approve.
/// - `OnFailure`: All commands run in a restricted sandbox initially. If a
///   command fails, the user is asked to approve execution without the sandbox.
/// - `OnRequest`: The model decides when to ask the user for approval.
/// - `Never`: Commands never ask for approval. Commands that fail in the
///   restricted sandbox are not retried.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TS, JsonSchema, AsRefStr)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum AskForApproval {
    UnlessTrusted,
    OnFailure,
    OnRequest,
    Never,
}

/// Reasoning effort for the underlying model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TS, JsonSchema, AsRefStr)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
}

/// Model reasoning summary style
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TS, JsonSchema, AsRefStr)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
    None,
}

/// Format for model reasoning summaries
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, TS, JsonSchema, AsRefStr)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum ReasoningSummaryFormat {
    None,
    Experimental,
}

enum CodexSessionAction {
    Chat { prompt: String },
    Review { target: ReviewTarget },
}

const CODEX_INIT_PROMPT: &str = include_str!("codex/init_prompt.md");
const DEFAULT_PROJECT_DOC_FILENAME: &str = "AGENTS.md";

#[derive(Debug, Clone)]
enum CodexSlashCommand {
    Init,
    Compact { instructions: Option<String> },
    Status,
    Mcp,
    Logout,
}

impl CodexSlashCommand {
    fn parse(prompt: &str) -> Option<Self> {
        let trimmed = prompt.trim_start();
        let command_line = trimmed.split('\n').next().unwrap_or(trimmed).trim();
        let without_slash = command_line.strip_prefix('/')?;
        let mut parts = without_slash.splitn(2, |ch: char| ch.is_whitespace());
        let name = parts.next()?.trim();
        if name.is_empty() {
            return None;
        }
        let rest = parts.next().map(str::trim).filter(|s| !s.is_empty());
        match name {
            "init" => Some(Self::Init),
            "compact" => Some(Self::Compact {
                instructions: rest.map(|s| s.to_string()),
            }),
            "status" => Some(Self::Status),
            "mcp" => Some(Self::Mcp),
            "logout" => Some(Self::Logout),
            _ => None,
        }
    }
}

#[derive(Derivative, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[derivative(Debug, PartialEq)]
pub struct Codex {
    #[serde(default)]
    pub append_prompt: AppendPrompt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SandboxMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask_for_approval: Option<AskForApproval>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oss: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reasoning_effort: Option<ReasoningEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reasoning_summary: Option<ReasoningSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_reasoning_summary_format: Option<ReasoningSummaryFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_apply_patch_tool: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compact_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub developer_instructions: Option<String>,
    #[serde(flatten)]
    pub cmd: CmdOverrides,

    #[serde(skip)]
    #[ts(skip)]
    #[derivative(Debug = "ignore", PartialEq = "ignore")]
    approvals: Option<Arc<dyn ExecutorApprovalService>>,
}

#[async_trait]
impl StandardCodingAgentExecutor for Codex {
    fn use_approvals(&mut self, approvals: Arc<dyn ExecutorApprovalService>) {
        self.approvals = Some(approvals);
    }

    fn slash_commands(&self) -> Vec<SlashCommand> {
        Self::hardcoded_slash_commands()
    }

    async fn spawn(
        &self,
        current_dir: &Path,
        prompt: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        self.spawn_with_slash_handling(current_dir, prompt, None, env)
            .await
    }

    async fn spawn_follow_up(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        self.spawn_with_slash_handling(current_dir, prompt, Some(session_id), env)
            .await
    }

    fn normalize_logs(&self, msg_store: Arc<MsgStore>, worktree_path: &Path) {
        normalize_logs(msg_store, worktree_path);
    }

    fn default_mcp_config_path(&self) -> Option<PathBuf> {
        codex_home().map(|home| home.join("config.toml"))
    }

    fn get_availability_info(&self) -> AvailabilityInfo {
        if let Some(timestamp) = codex_home()
            .and_then(|home| std::fs::metadata(home.join("auth.json")).ok())
            .and_then(|m| m.modified().ok())
            .and_then(|modified| modified.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
        {
            return AvailabilityInfo::LoginDetected {
                last_auth_timestamp: timestamp,
            };
        }

        let mcp_config_found = self
            .default_mcp_config_path()
            .map(|p| p.exists())
            .unwrap_or(false);

        let installation_indicator_found = codex_home()
            .map(|home| home.join("version.json").exists())
            .unwrap_or(false);

        if mcp_config_found || installation_indicator_found {
            AvailabilityInfo::InstallationFound
        } else {
            AvailabilityInfo::NotFound
        }
    }

    async fn spawn_review(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let command_parts = self.build_command_builder()?.build_initial()?;
        let review_target = ReviewTarget::Custom {
            instructions: prompt.to_string(),
        };
        let action = CodexSessionAction::Review {
            target: review_target,
        };
        self.spawn_inner(current_dir, command_parts, action, session_id, env)
            .await
    }
}

impl Codex {
    pub fn base_command() -> &'static str {
        "npx -y @openai/codex@0.86.0"
    }

    fn build_command_builder(&self) -> Result<CommandBuilder, CommandBuildError> {
        let mut builder = CommandBuilder::new(Self::base_command());
        builder = builder.extend_params(["app-server"]);
        if self.oss.unwrap_or(false) {
            builder = builder.extend_params(["--oss"]);
        }

        apply_overrides(builder, &self.cmd)
    }

    fn build_new_conversation_params(&self, cwd: &Path) -> NewConversationParams {
        let sandbox = match self.sandbox.as_ref() {
            None | Some(SandboxMode::Auto) => Some(CodexSandboxMode::WorkspaceWrite), // match the Auto preset in codex
            Some(SandboxMode::ReadOnly) => Some(CodexSandboxMode::ReadOnly),
            Some(SandboxMode::WorkspaceWrite) => Some(CodexSandboxMode::WorkspaceWrite),
            Some(SandboxMode::DangerFullAccess) => Some(CodexSandboxMode::DangerFullAccess),
        };

        let approval_policy = match self.ask_for_approval.as_ref() {
            None if matches!(self.sandbox.as_ref(), None | Some(SandboxMode::Auto)) => {
                // match the Auto preset in codex
                Some(CodexAskForApproval::OnRequest)
            }
            None => None,
            Some(AskForApproval::UnlessTrusted) => Some(CodexAskForApproval::UnlessTrusted),
            Some(AskForApproval::OnFailure) => Some(CodexAskForApproval::OnFailure),
            Some(AskForApproval::OnRequest) => Some(CodexAskForApproval::OnRequest),
            Some(AskForApproval::Never) => Some(CodexAskForApproval::Never),
        };

        NewConversationParams {
            model: self.model.clone(),
            profile: self.profile.clone(),
            cwd: Some(cwd.to_string_lossy().to_string()),
            approval_policy,
            sandbox,
            config: self.build_config_overrides(),
            base_instructions: self.base_instructions.clone(),
            include_apply_patch_tool: self.include_apply_patch_tool,
            model_provider: self.model_provider.clone(),
            compact_prompt: self.compact_prompt.clone(),
            developer_instructions: self.developer_instructions.clone(),
        }
    }

    fn build_config_overrides(&self) -> Option<HashMap<String, Value>> {
        let mut overrides = HashMap::new();

        if let Some(effort) = &self.model_reasoning_effort {
            overrides.insert(
                "model_reasoning_effort".to_string(),
                Value::String(effort.as_ref().to_string()),
            );
        }

        if let Some(summary) = &self.model_reasoning_summary {
            overrides.insert(
                "model_reasoning_summary".to_string(),
                Value::String(summary.as_ref().to_string()),
            );
        }

        if let Some(format) = &self.model_reasoning_summary_format
            && format != &ReasoningSummaryFormat::None
        {
            overrides.insert(
                "model_reasoning_summary_format".to_string(),
                Value::String(format.as_ref().to_string()),
            );
        }

        if overrides.is_empty() {
            None
        } else {
            Some(overrides)
        }
    }

    fn slash_command_description(name: &str) -> Option<&'static str> {
        match name {
            "init" => Some("create an AGENTS.md file with instructions for Codex"),
            "compact" => Some("summarize conversation to prevent hitting the context limit"),
            "status" => Some("show current session configuration and token usage"),
            "mcp" => Some("list configured MCP tools"),
            "logout" => Some("log out of Codex"),
            _ => None,
        }
    }

    fn hardcoded_slash_commands() -> Vec<SlashCommand> {
        const NAMES: [&str; 5] = ["init", "compact", "status", "mcp", "logout"];
        NAMES
            .into_iter()
            .map(|name| SlashCommand {
                name: name.to_string(),
                description: Self::slash_command_description(name).map(|d| d.to_string()),
            })
            .collect()
    }

    async fn spawn_with_slash_handling(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        if let Some(command) = CodexSlashCommand::parse(prompt) {
            return match command {
                CodexSlashCommand::Init => {
                    let init_target = current_dir.join(DEFAULT_PROJECT_DOC_FILENAME);
                    if init_target.exists() {
                        let message = format!(
                            "{DEFAULT_PROJECT_DOC_FILENAME} already exists here. Skipping /init to avoid overwriting it."
                        );
                        self.spawn_local_message(current_dir, message).await
                    } else {
                        self.spawn_chat(current_dir, CODEX_INIT_PROMPT, session_id, env)
                            .await
                    }
                }
                CodexSlashCommand::Compact { instructions } => match session_id {
                    Some(session_id) => {
                        self.spawn_compact(current_dir, session_id, instructions, env)
                            .await
                    }
                    None => {
                        self.spawn_local_message(
                            current_dir,
                            "No session available to compact yet.".to_string(),
                        )
                        .await
                    }
                },
                CodexSlashCommand::Status => self.spawn_status(current_dir, session_id).await,
                CodexSlashCommand::Mcp => {
                    self.spawn_app_server_command(current_dir, CodexSlashCommand::Mcp, env)
                        .await
                }
                CodexSlashCommand::Logout => {
                    self.spawn_app_server_command(current_dir, CodexSlashCommand::Logout, env)
                        .await
                }
            };
        }

        self.spawn_chat(current_dir, prompt, session_id, env).await
    }

    async fn spawn_chat(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let command_parts = match session_id {
            Some(_) => self.build_command_builder()?.build_follow_up(&[])?,
            None => self.build_command_builder()?.build_initial()?,
        };
        let combined_prompt = self.append_prompt.combine_prompt(prompt);
        let action = CodexSessionAction::Chat {
            prompt: combined_prompt,
        };
        self.spawn_inner(current_dir, command_parts, action, session_id, env)
            .await
    }

    async fn spawn_status(
        &self,
        current_dir: &Path,
        session_id: Option<&str>,
    ) -> Result<SpawnedChild, ExecutorError> {
        match self.build_status_message(session_id).await {
            Ok(message) => self.spawn_local_message(current_dir, message).await,
            Err(err) => {
                let message = format!("Status unavailable: {err}");
                self.spawn_local_error(current_dir, message).await
            }
        }
    }

    async fn spawn_compact(
        &self,
        current_dir: &Path,
        session_id: &str,
        instructions: Option<String>,
        _env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let mut process = Command::new("sh");
        process
            .kill_on_drop(true)
            .arg("-c")
            .arg("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(current_dir);
        let mut child = process.group_spawn()?;
        let new_stdout = create_stdout_pipe_writer(&mut child)?;
        let log_writer = LogWriter::new(new_stdout);
        let (exit_signal_tx, exit_signal_rx) = tokio::sync::oneshot::channel();

        let codex = self.clone();
        let session_id = session_id.to_string();
        let current_dir = current_dir.to_path_buf();
        tokio::spawn(async move {
            let result = codex
                .run_compact(&current_dir, &session_id, instructions, &log_writer)
                .await;
            let exit_result = match result {
                Ok(()) => ExecutorExitResult::Success,
                Err(err) => {
                    let message = format!("Compact failed: {err}");
                    let _ = codex
                        .log_event(
                            &log_writer,
                            EventMsg::Error(ErrorEvent {
                                message,
                                codex_error_info: None,
                            }),
                        )
                        .await;
                    ExecutorExitResult::Failure
                }
            };
            let _ = exit_signal_tx.send(exit_result);
        });

        Ok(SpawnedChild {
            child,
            exit_signal: Some(exit_signal_rx),
            interrupt_sender: None,
        })
    }

    async fn run_compact(
        &self,
        current_dir: &Path,
        session_id: &str,
        instructions: Option<String>,
        log_writer: &LogWriter,
    ) -> Result<(), ExecutorError> {
        let rollout_path = SessionHandler::find_rollout_file_path(session_id)
            .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))?;
        let config = self.build_core_config(current_dir, instructions).await?;
        let auth_manager = AuthManager::shared(
            config.codex_home.clone(),
            true,
            config.cli_auth_credentials_store_mode,
        );
        let thread_manager = ThreadManager::new(
            config.codex_home.clone(),
            auth_manager.clone(),
            SessionSource::Exec,
        );
        let new_thread = thread_manager
            .resume_thread_from_rollout(config, rollout_path, auth_manager)
            .await
            .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))?;
        let thread = new_thread.thread;
        thread
            .submit(CoreOp::Compact)
            .await
            .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))?;

        loop {
            let event: Event = thread
                .next_event()
                .await
                .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))?;
            self.log_event(log_writer, event.msg.clone()).await?;
            if matches!(event.msg, EventMsg::TurnComplete(_)) {
                break;
            }
        }

        let _ = thread.submit(CoreOp::Shutdown).await;
        loop {
            match thread.next_event().await {
                Ok(event) => {
                    if matches!(event.msg, EventMsg::ShutdownComplete) {
                        break;
                    }
                }
                Err(_) => break,
            }
        }

        Ok(())
    }

    async fn spawn_app_server_command(
        &self,
        current_dir: &Path,
        command: CodexSlashCommand,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let command_parts = self.build_command_builder()?.build_initial()?;
        let (program_path, args) = command_parts.into_resolved().await?;

        let mut process = Command::new(program_path);
        process
            .kill_on_drop(true)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .current_dir(current_dir)
            .args(&args)
            .env("NODE_NO_WARNINGS", "1")
            .env("NO_COLOR", "1")
            .env("RUST_LOG", "error");

        env.clone()
            .with_profile(&self.cmd)
            .apply_to_command(&mut process);

        let mut child = process.group_spawn()?;

        let child_stdout = child.inner().stdout.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other("Codex app server missing stdout"))
        })?;
        let child_stdin = child.inner().stdin.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other("Codex app server missing stdin"))
        })?;

        let new_stdout = create_stdout_pipe_writer(&mut child)?;
        let (exit_signal_tx, exit_signal_rx) = tokio::sync::oneshot::channel();

        let auto_approve = matches!(
            (&self.sandbox, &self.ask_for_approval),
            (Some(SandboxMode::DangerFullAccess), None)
        );
        let approvals = self.approvals.clone();
        tokio::spawn(async move {
            let exit_signal_tx = ExitSignalSender::new(exit_signal_tx);
            let log_writer = LogWriter::new(new_stdout);
            let client = AppServerClient::new(log_writer.clone(), approvals, auto_approve);
            let rpc_peer = JsonRpcPeer::spawn(
                child_stdin,
                child_stdout,
                client.clone(),
                exit_signal_tx.clone(),
            );
            client.connect(rpc_peer);

            let result = async {
                client.initialize().await?;
                match command {
                    CodexSlashCommand::Mcp => {
                        let message = fetch_mcp_status_message(&client).await?;
                        log_event_raw(&log_writer, message).await?;
                    }
                    CodexSlashCommand::Logout => {
                        client.logout_account().await?;
                        log_event_raw(&log_writer, "Logged out of Codex.".to_string()).await?;
                    }
                    _ => {
                        return Err(ExecutorError::Io(std::io::Error::other(
                            "Unsupported Codex slash command",
                        )));
                    }
                }
                Ok::<(), ExecutorError>(())
            }
            .await;

            if let Err(err) = result {
                let message = format!("Slash command failed: {err}");
                let _ = log_event_notification(
                    &log_writer,
                    EventMsg::Error(ErrorEvent {
                        message,
                        codex_error_info: None,
                    }),
                )
                .await;
                exit_signal_tx
                    .send_exit_signal(ExecutorExitResult::Failure)
                    .await;
                return;
            }

            exit_signal_tx
                .send_exit_signal(ExecutorExitResult::Success)
                .await;
        });

        Ok(SpawnedChild {
            child,
            exit_signal: Some(exit_signal_rx),
            interrupt_sender: None,
        })
    }

    async fn spawn_local_message(
        &self,
        current_dir: &Path,
        message: String,
    ) -> Result<SpawnedChild, ExecutorError> {
        self.spawn_local_events(
            current_dir,
            vec![EventMsg::AgentMessage(AgentMessageEvent { message })],
        )
        .await
    }

    async fn spawn_local_error(
        &self,
        current_dir: &Path,
        message: String,
    ) -> Result<SpawnedChild, ExecutorError> {
        self.spawn_local_events(
            current_dir,
            vec![EventMsg::Error(ErrorEvent {
                message,
                codex_error_info: None,
            })],
        )
        .await
    }

    async fn spawn_local_events(
        &self,
        current_dir: &Path,
        events: Vec<EventMsg>,
    ) -> Result<SpawnedChild, ExecutorError> {
        let mut process = Command::new("sh");
        process
            .kill_on_drop(true)
            .arg("-c")
            .arg("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(current_dir);
        let mut child = process.group_spawn()?;
        let new_stdout = create_stdout_pipe_writer(&mut child)?;
        let log_writer = LogWriter::new(new_stdout);
        let (exit_signal_tx, exit_signal_rx) = tokio::sync::oneshot::channel();

        tokio::spawn(async move {
            let mut exit_result = ExecutorExitResult::Success;
            for event in events {
                if let Err(err) = log_event_notification(&log_writer, event).await {
                    tracing::error!("Failed to emit slash command output: {err}");
                    exit_result = ExecutorExitResult::Failure;
                    break;
                }
            }
            let _ = exit_signal_tx.send(exit_result);
        });

        Ok(SpawnedChild {
            child,
            exit_signal: Some(exit_signal_rx),
            interrupt_sender: None,
        })
    }

    async fn build_core_config(
        &self,
        current_dir: &Path,
        compact_prompt_override: Option<String>,
    ) -> Result<Config, ExecutorError> {
        let approval_policy = match self.ask_for_approval.as_ref() {
            Some(policy) => Some(Self::map_ask_for_approval(policy)),
            None if matches!(self.sandbox.as_ref(), None | Some(SandboxMode::Auto)) => {
                Some(CodexAskForApproval::OnRequest)
            }
            None => None,
        };
        let sandbox_mode = match self.sandbox.as_ref() {
            None | Some(SandboxMode::Auto) => Some(CodexSandboxMode::WorkspaceWrite),
            Some(SandboxMode::ReadOnly) => Some(CodexSandboxMode::ReadOnly),
            Some(SandboxMode::WorkspaceWrite) => Some(CodexSandboxMode::WorkspaceWrite),
            Some(SandboxMode::DangerFullAccess) => Some(CodexSandboxMode::DangerFullAccess),
        };

        let mut overrides = ConfigOverrides::default();
        overrides.cwd = Some(current_dir.to_path_buf());
        overrides.model = self.model.clone();
        overrides.model_provider = self.model_provider.clone();
        overrides.config_profile = self.profile.clone();
        overrides.approval_policy = approval_policy;
        overrides.sandbox_mode = sandbox_mode;
        overrides.base_instructions = self.base_instructions.clone();
        overrides.developer_instructions = self.developer_instructions.clone();
        overrides.compact_prompt = compact_prompt_override.or_else(|| self.compact_prompt.clone());
        overrides.include_apply_patch_tool = self.include_apply_patch_tool;

        Config::load_with_cli_overrides_and_harness_overrides(Vec::new(), overrides)
            .await
            .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))
    }

    async fn build_status_message(
        &self,
        session_id: Option<&str>,
    ) -> Result<String, ExecutorError> {
        let mut model = self.model.clone();
        let mut approval_policy = self
            .ask_for_approval
            .as_ref()
            .map(|policy| policy.as_ref().to_string());
        let mut sandbox = self.sandbox.as_ref().map(|mode| mode.as_ref().to_string());
        let mut reasoning = None;
        let mut token_usage = None;

        if let Some(session_id) = session_id {
            let items = Self::load_rollout_items(session_id).await?;
            if let Some(context) = Self::latest_turn_context(&items) {
                model = Some(context.model);
                approval_policy = Some(context.approval_policy.to_string());
                sandbox = Some(context.sandbox_policy.to_string());
                reasoning = Some(format!(
                    "effort: {} summary: {}",
                    context
                        .effort
                        .map(|effort| effort.to_string())
                        .unwrap_or_else(|| "default".to_string()),
                    context.summary
                ));
            }
            token_usage = Self::latest_token_usage(&items);
        }

        let mut lines = Vec::new();
        lines.push("Status".to_string());
        lines.push(format!(
            "Model: {}",
            model.unwrap_or_else(|| "unknown".to_string())
        ));
        if let Some(approval_policy) = approval_policy {
            lines.push(format!("Approvals: {approval_policy}"));
        }
        if let Some(sandbox) = sandbox {
            lines.push(format!("Sandbox: {sandbox}"));
        }
        if let Some(reasoning) = reasoning {
            lines.push(format!("Reasoning: {reasoning}"));
        }
        if let Some(token_usage) = token_usage {
            lines.extend(Self::format_token_usage(&token_usage));
        } else {
            lines.push("Tokens: unavailable".to_string());
        }

        Ok(lines.join("\n"))
    }

    async fn load_rollout_items(session_id: &str) -> Result<Vec<RolloutItem>, ExecutorError> {
        let rollout_path = SessionHandler::find_rollout_file_path(session_id)
            .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))?;
        let history = RolloutRecorder::get_rollout_history(&rollout_path)
            .await
            .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))?;
        Ok(history.get_rollout_items())
    }

    fn latest_turn_context(items: &[RolloutItem]) -> Option<TurnContextItem> {
        items.iter().rev().find_map(|item| match item {
            RolloutItem::TurnContext(context) => Some(context.clone()),
            _ => None,
        })
    }

    fn latest_token_usage(items: &[RolloutItem]) -> Option<TokenUsageInfo> {
        items.iter().rev().find_map(|item| match item {
            RolloutItem::EventMsg(EventMsg::TokenCount(payload)) => payload.info.clone(),
            _ => None,
        })
    }

    fn format_token_usage(info: &TokenUsageInfo) -> Vec<String> {
        let total = &info.total_token_usage;
        let last = &info.last_token_usage;
        let mut lines = Vec::new();
        lines.push(format!(
            "Tokens total: {} (input {}, output {}, reasoning {}, cached {})",
            total.total_tokens,
            total.input_tokens,
            total.output_tokens,
            total.reasoning_output_tokens,
            total.cached_input_tokens,
        ));
        lines.push(format!(
            "Last turn: {} (input {}, output {}, reasoning {}, cached {})",
            last.total_tokens,
            last.input_tokens,
            last.output_tokens,
            last.reasoning_output_tokens,
            last.cached_input_tokens,
        ));
        if let Some(window) = info.model_context_window {
            lines.push(format!("Context window: {window}"));
        }
        lines
    }

    fn map_ask_for_approval(value: &AskForApproval) -> CodexAskForApproval {
        match value {
            AskForApproval::UnlessTrusted => CodexAskForApproval::UnlessTrusted,
            AskForApproval::OnFailure => CodexAskForApproval::OnFailure,
            AskForApproval::OnRequest => CodexAskForApproval::OnRequest,
            AskForApproval::Never => CodexAskForApproval::Never,
        }
    }

    async fn log_event(
        &self,
        log_writer: &LogWriter,
        event: EventMsg,
    ) -> Result<(), ExecutorError> {
        log_event_notification(log_writer, event).await
    }

    async fn spawn_inner(
        &self,
        current_dir: &Path,
        command_parts: CommandParts,
        action: CodexSessionAction,
        resume_session: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let (program_path, args) = command_parts.into_resolved().await?;

        let mut process = Command::new(program_path);
        process
            .kill_on_drop(true)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .current_dir(current_dir)
            .args(&args)
            .env("NODE_NO_WARNINGS", "1")
            .env("NO_COLOR", "1")
            .env("RUST_LOG", "error");

        env.clone()
            .with_profile(&self.cmd)
            .apply_to_command(&mut process);

        let mut child = process.group_spawn()?;

        let child_stdout = child.inner().stdout.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other("Codex app server missing stdout"))
        })?;
        let child_stdin = child.inner().stdin.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other("Codex app server missing stdin"))
        })?;

        let new_stdout = create_stdout_pipe_writer(&mut child)?;
        let (exit_signal_tx, exit_signal_rx) = tokio::sync::oneshot::channel();

        let params = self.build_new_conversation_params(current_dir);
        let resume_session = resume_session.map(|s| s.to_string());
        let auto_approve = matches!(
            (&self.sandbox, &self.ask_for_approval),
            (Some(SandboxMode::DangerFullAccess), None)
        );
        let approvals = self.approvals.clone();
        tokio::spawn(async move {
            let exit_signal_tx = ExitSignalSender::new(exit_signal_tx);
            let log_writer = LogWriter::new(new_stdout);
            let launch_result = match action {
                CodexSessionAction::Chat { prompt } => {
                    Self::launch_codex_app_server(
                        params,
                        resume_session,
                        prompt,
                        child_stdout,
                        child_stdin,
                        log_writer.clone(),
                        exit_signal_tx.clone(),
                        approvals,
                        auto_approve,
                    )
                    .await
                }
                CodexSessionAction::Review { target } => {
                    review::launch_codex_review(
                        params,
                        resume_session,
                        target,
                        child_stdout,
                        child_stdin,
                        log_writer.clone(),
                        exit_signal_tx.clone(),
                        approvals,
                        auto_approve,
                    )
                    .await
                }
            };
            if let Err(err) = launch_result {
                match &err {
                    ExecutorError::Io(io_err)
                        if io_err.kind() == std::io::ErrorKind::BrokenPipe =>
                    {
                        // Broken pipe likely means the parent process exited, so we can ignore it
                        return;
                    }
                    ExecutorError::AuthRequired(message) => {
                        log_writer
                            .log_raw(&Error::auth_required(message.clone()).raw())
                            .await
                            .ok();
                        // Send failure signal so the process is marked as failed
                        exit_signal_tx
                            .send_exit_signal(ExecutorExitResult::Failure)
                            .await;
                        return;
                    }
                    _ => {
                        tracing::error!("Codex spawn error: {}", err);
                        log_writer
                            .log_raw(&Error::launch_error(err.to_string()).raw())
                            .await
                            .ok();
                    }
                }
                // For other errors, also send failure signal
                exit_signal_tx
                    .send_exit_signal(ExecutorExitResult::Failure)
                    .await;
            }
        });

        Ok(SpawnedChild {
            child,
            exit_signal: Some(exit_signal_rx),
            interrupt_sender: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn launch_codex_app_server(
        conversation_params: NewConversationParams,
        resume_session: Option<String>,
        combined_prompt: String,
        child_stdout: tokio::process::ChildStdout,
        child_stdin: tokio::process::ChildStdin,
        log_writer: LogWriter,
        exit_signal_tx: ExitSignalSender,
        approvals: Option<Arc<dyn ExecutorApprovalService>>,
        auto_approve: bool,
    ) -> Result<(), ExecutorError> {
        let client = AppServerClient::new(log_writer, approvals, auto_approve);
        let rpc_peer =
            JsonRpcPeer::spawn(child_stdin, child_stdout, client.clone(), exit_signal_tx);
        client.connect(rpc_peer);
        client.initialize().await?;
        let auth_status = client.get_auth_status().await?;
        if auth_status.requires_openai_auth.unwrap_or(true) && auth_status.auth_method.is_none() {
            return Err(ExecutorError::AuthRequired(
                "Codex authentication required".to_string(),
            ));
        }
        match resume_session {
            None => {
                let params = conversation_params;
                let response = client.new_conversation(params).await?;
                let conversation_id = response.conversation_id;
                client.register_session(&conversation_id).await?;
                client.add_conversation_listener(conversation_id).await?;
                client
                    .send_user_message(conversation_id, combined_prompt)
                    .await?;
            }
            Some(session_id) => {
                let (rollout_path, _forked_session_id) =
                    SessionHandler::fork_rollout_file(&session_id)
                        .map_err(|e| ExecutorError::FollowUpNotSupported(e.to_string()))?;
                let overrides = conversation_params;
                let response = client
                    .resume_conversation(rollout_path.clone(), overrides)
                    .await?;
                tracing::debug!(
                    "resuming session using rollout file {}, response {:?}",
                    rollout_path.display(),
                    response
                );
                let conversation_id = response.conversation_id;
                client.register_session(&conversation_id).await?;
                client.add_conversation_listener(conversation_id).await?;
                client
                    .send_user_message(conversation_id, combined_prompt)
                    .await?;
            }
        }
        Ok(())
    }
}

async fn log_event_notification(
    log_writer: &LogWriter,
    event: EventMsg,
) -> Result<(), ExecutorError> {
    let event = match event {
        EventMsg::SessionConfigured(mut configured) => {
            configured.initial_messages = None;
            EventMsg::SessionConfigured(configured)
        }
        other => other,
    };
    let notification = JSONRPCNotification {
        method: "codex/event".to_string(),
        params: Some(json!({ "msg": event })),
    };
    let raw = serde_json::to_string(&notification)
        .map_err(|err| ExecutorError::Io(std::io::Error::other(err.to_string())))?;
    log_writer.log_raw(&raw).await
}

async fn log_event_raw(log_writer: &LogWriter, message: String) -> Result<(), ExecutorError> {
    log_event_notification(
        log_writer,
        EventMsg::AgentMessage(AgentMessageEvent { message }),
    )
    .await
}

async fn fetch_mcp_status_message(client: &AppServerClient) -> Result<String, ExecutorError> {
    let mut cursor = None;
    let mut servers = Vec::new();
    loop {
        let response = client.list_mcp_server_status(cursor).await?;
        servers.extend(response.data);
        cursor = response.next_cursor;
        if cursor.is_none() {
            break;
        }
    }
    Ok(format_mcp_status(&servers))
}

fn format_mcp_status(servers: &[codex_app_server_protocol::McpServerStatus]) -> String {
    if servers.is_empty() {
        return "No MCP servers configured.".to_string();
    }
    let mut lines = vec![format!("MCP servers ({})", servers.len())];
    for server in servers {
        let auth = format_mcp_auth_status(&server.auth_status);
        lines.push(format!("- {} (auth: {auth})", server.name));
        let mut tools: Vec<String> = server.tools.keys().cloned().collect();
        tools.sort();
        let tool_line = if tools.is_empty() {
            "  tools: none".to_string()
        } else {
            format!("  tools: {}", tools.join(", "))
        };
        lines.push(tool_line);
        if !server.resources.is_empty() {
            let mut names: Vec<String> = server
                .resources
                .iter()
                .map(|res| res.name.clone())
                .collect();
            names.sort();
            lines.push(format!("  resources: {}", names.join(", ")));
        }
        if !server.resource_templates.is_empty() {
            let mut names: Vec<String> = server
                .resource_templates
                .iter()
                .map(|template| template.name.clone())
                .collect();
            names.sort();
            lines.push(format!("  resource templates: {}", names.join(", ")));
        }
    }
    lines.join("\n")
}

fn format_mcp_auth_status(
    status: &codex_app_server_protocol::McpAuthStatus,
) -> &'static str {
    match status {
        codex_app_server_protocol::McpAuthStatus::Unsupported => "unsupported",
        codex_app_server_protocol::McpAuthStatus::NotLoggedIn => "not logged in",
        codex_app_server_protocol::McpAuthStatus::BearerToken => "bearer token",
        codex_app_server_protocol::McpAuthStatus::OAuth => "oauth",
    }
}
