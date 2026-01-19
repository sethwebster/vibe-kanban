use std::{
    collections::HashMap,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use command_group::AsyncCommandGroup;
use derivative::Derivative;
use lru::LruCache;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncBufReadExt, process::Command};
use ts_rs::TS;
use workspace_utils::msg_store::MsgStore;

use crate::{
    approvals::ExecutorApprovalService,
    command::{CmdOverrides, CommandBuildError, CommandBuilder, apply_overrides},
    env::{ExecutionEnv, RepoContext},
    executors::{
        AppendPrompt, AvailabilityInfo, ExecutorError, ExecutorExitResult, SlashCommand,
        SpawnedChild, StandardCodingAgentExecutor,
    },
    stdout_dup::create_stdout_pipe_writer,
};

mod normalize_logs;
mod sdk;
mod types;

use sdk::{LogWriter, RunConfig, run_session, run_slash_command};

const SLASH_COMMANDS_CACHE_CAPACITY: usize = 32;
const SLASH_COMMANDS_CACHE_TTL: Duration = Duration::from_secs(60 * 5);

#[derive(Clone, Debug)]
struct CachedSlashCommands {
    cached_at: Instant,
    commands: Arc<Vec<SlashCommand>>,
}

static SLASH_COMMANDS_CACHE: OnceLock<Mutex<LruCache<PathBuf, CachedSlashCommands>>> =
    OnceLock::new();

fn slash_commands_cache() -> &'static Mutex<LruCache<PathBuf, CachedSlashCommands>> {
    SLASH_COMMANDS_CACHE.get_or_init(|| {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(SLASH_COMMANDS_CACHE_CAPACITY)
                .expect("SLASH_COMMANDS_CACHE_CAPACITY must be > 0"),
        ))
    })
}

fn get_cached_slash_commands(key: &PathBuf) -> Option<Arc<Vec<SlashCommand>>> {
    let mut cache = slash_commands_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let mut should_evict = false;
    let mut commands = None;

    if let Some(entry) = cache.get_mut(key) {
        if entry.cached_at.elapsed() <= SLASH_COMMANDS_CACHE_TTL {
            commands = Some(Arc::clone(&entry.commands));
        } else {
            should_evict = true;
        }
    }

    if should_evict {
        cache.pop(key);
    }

    commands
}

#[derive(Debug, Clone)]
enum OpencodeSlashCommand {
    Compact,
    Share,
    Unshare,
    Commands,
    Models { provider: Option<String> },
    Agents,
    Status,
    Mcp,
    Dynamic { name: String, arguments: String },
}

impl OpencodeSlashCommand {
    fn parse(prompt: &str) -> Option<Self> {
        let trimmed = prompt.trim_start();
        let without_slash = trimmed.strip_prefix('/')?;
        let mut parts = without_slash.splitn(2, |ch: char| ch.is_whitespace());
        let name = parts.next()?.trim();
        if name.is_empty() {
            return None;
        }
        let arguments = parts.next().unwrap_or("").trim().to_string();
        let key = name.to_ascii_lowercase();

        let command = match key.as_str() {
            "compact" | "summarize" => OpencodeSlashCommand::Compact,
            "share" => OpencodeSlashCommand::Share,
            "unshare" => OpencodeSlashCommand::Unshare,
            "commands" => OpencodeSlashCommand::Commands,
            "models" => {
                let provider = arguments
                    .split_whitespace()
                    .next()
                    .map(|value| value.to_string());
                OpencodeSlashCommand::Models { provider }
            }
            "agents" => OpencodeSlashCommand::Agents,
            "status" => OpencodeSlashCommand::Status,
            "mcp" => OpencodeSlashCommand::Mcp,
            _ => OpencodeSlashCommand::Dynamic {
                name: name.to_string(),
                arguments,
            },
        };

        Some(command)
    }

    fn requires_existing_session(&self) -> bool {
        matches!(
            self,
            OpencodeSlashCommand::Compact
                | OpencodeSlashCommand::Share
                | OpencodeSlashCommand::Unshare
        )
    }

    fn should_fork_session(&self) -> bool {
        !matches!(
            self,
            OpencodeSlashCommand::Share | OpencodeSlashCommand::Unshare
        )
    }
}

#[derive(Derivative, Clone, Serialize, Deserialize, TS, JsonSchema)]
#[derivative(Debug, PartialEq)]
pub struct Opencode {
    #[serde(default)]
    pub append_prompt: AppendPrompt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "agent")]
    pub mode: Option<String>,
    /// Auto-approve agent actions
    #[serde(default = "default_to_true")]
    pub auto_approve: bool,
    #[serde(flatten)]
    pub cmd: CmdOverrides,
    #[serde(skip)]
    #[ts(skip)]
    #[derivative(Debug = "ignore", PartialEq = "ignore")]
    pub approvals: Option<Arc<dyn ExecutorApprovalService>>,
}

impl Opencode {
    fn build_command_builder(&self) -> Result<CommandBuilder, CommandBuildError> {
        let builder = CommandBuilder::new("npx -y opencode-ai@1.1.3")
            // Pass hostname/port as separate args so OpenCode treats them as explicitly set
            // (it checks `process.argv.includes(\"--port\")` / `\"--hostname\"`).
            .extend_params(["serve", "--hostname", "127.0.0.1", "--port", "0"]);
        apply_overrides(builder, &self.cmd)
    }

    fn slash_command_description(name: &str) -> Option<&'static str> {
        match name {
            "compact" | "summarize" => Some("compact the session"),
            "share" => Some("share a session"),
            "unshare" => Some("unshare a session"),
            "commands" => Some("show all commands"),
            "models" => Some("list models"),
            "agents" => Some("list agents"),
            "status" => Some("show status"),
            "mcp" => Some("show MCP status"),
            _ => None,
        }
    }

    fn hardcoded_slash_commands() -> Vec<SlashCommand> {
        const NAMES: [&str; 9] = [
            "compact",
            "summarize",
            "share",
            "unshare",
            "commands",
            "models",
            "agents",
            "status",
            "mcp",
        ];

        NAMES
            .into_iter()
            .map(|name| SlashCommand {
                name: name.to_string(),
                description: Self::slash_command_description(name).map(|d| d.to_string()),
            })
            .collect()
    }

    pub async fn discover_slash_commands(
        &self,
        current_dir: &Path,
    ) -> Result<Vec<SlashCommand>, ExecutorError> {
        let key = current_dir.to_path_buf();
        if let Some(cached) = get_cached_slash_commands(&key) {
            return Ok((*cached).clone());
        }

        let command_parts = self.build_command_builder()?.build_initial()?;
        let (program_path, args) = command_parts.into_resolved().await?;

        let mut command = Command::new(program_path);
        command
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .current_dir(current_dir)
            .args(&args)
            .env("NODE_NO_WARNINGS", "1")
            .env("NO_COLOR", "1");

        ExecutionEnv::new(RepoContext::default(), false)
            .with_profile(&self.cmd)
            .apply_to_command(&mut command);

        let mut child = command.group_spawn()?;
        let server_stdout = child.inner().stdout.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other(
                "OpenCode server missing stdout (needed to parse listening URL)",
            ))
        })?;

        let base_url = wait_for_server_url(server_stdout).await?;
        let commands = sdk::discover_commands(&base_url, current_dir).await?;

        let mut merged: HashMap<String, SlashCommand> = Self::hardcoded_slash_commands()
            .into_iter()
            .map(|cmd| (cmd.name.clone(), cmd))
            .collect();

        for command in commands {
            let name = command.name.trim_start_matches('/').to_string();
            let entry = merged.entry(name.clone()).or_insert(SlashCommand {
                name: name.clone(),
                description: None,
            });
            if entry.description.is_none() {
                entry.description = command.description.clone();
            }
        }

        let mut commands: Vec<SlashCommand> = merged.into_values().collect();
        commands.sort_by(|a, b| a.name.cmp(&b.name));

        let commands_arc = Arc::new(commands.clone());
        slash_commands_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .put(
                key,
                CachedSlashCommands {
                    cached_at: Instant::now(),
                    commands: Arc::clone(&commands_arc),
                },
            );

        Ok(commands)
    }

    async fn spawn_inner(
        &self,
        current_dir: &Path,
        prompt: &str,
        resume_session: Option<&str>,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let slash_command = OpencodeSlashCommand::parse(prompt);
        let combined_prompt = if slash_command.is_some() {
            prompt.to_string()
        } else {
            self.append_prompt.combine_prompt(prompt)
        };

        let command_parts = self.build_command_builder()?.build_initial()?;
        let (program_path, args) = command_parts.into_resolved().await?;

        let mut command = Command::new(program_path);
        command
            .kill_on_drop(true)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .current_dir(current_dir)
            .args(&args)
            .env("NODE_NO_WARNINGS", "1")
            .env("NO_COLOR", "1");

        env.clone()
            .with_profile(&self.cmd)
            .apply_to_command(&mut command);

        let mut child = command.group_spawn()?;
        let server_stdout = child.inner().stdout.take().ok_or_else(|| {
            ExecutorError::Io(std::io::Error::other(
                "OpenCode server missing stdout (needed to parse listening URL)",
            ))
        })?;

        let stdout = create_stdout_pipe_writer(&mut child)?;
        let log_writer = LogWriter::new(stdout);

        let (exit_signal_tx, exit_signal_rx) = tokio::sync::oneshot::channel();
        let (interrupt_tx, interrupt_rx) = tokio::sync::oneshot::channel();

        let directory = current_dir.to_string_lossy().to_string();
        let base_url = wait_for_server_url(server_stdout).await?;
        let approvals = if self.auto_approve {
            None
        } else {
            self.approvals.clone()
        };

        let config = RunConfig {
            base_url,
            directory,
            prompt: combined_prompt,
            resume_session_id: resume_session.map(|s| s.to_string()),
            model: self.model.clone(),
            agent: self.mode.clone(),
            approvals,
            auto_approve: self.auto_approve,
        };

        tokio::spawn(async move {
            let result = match slash_command {
                Some(command) => {
                    run_slash_command(config, log_writer.clone(), command, interrupt_rx).await
                }
                None => run_session(config, log_writer.clone(), interrupt_rx).await,
            };
            let exit_result = match result {
                Ok(()) => ExecutorExitResult::Success,
                Err(err) => {
                    let _ = log_writer
                        .log_error(format!("OpenCode executor error: {err}"))
                        .await;
                    ExecutorExitResult::Failure
                }
            };
            let _ = exit_signal_tx.send(exit_result);
        });

        Ok(SpawnedChild {
            child,
            exit_signal: Some(exit_signal_rx),
            interrupt_sender: Some(interrupt_tx),
        })
    }
}

fn format_tail(captured: Vec<String>) -> String {
    captured
        .into_iter()
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n")
}

async fn wait_for_server_url(stdout: tokio::process::ChildStdout) -> Result<String, ExecutorError> {
    let mut lines = tokio::io::BufReader::new(stdout).lines();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    let mut captured: Vec<String> = Vec::new();

    loop {
        if tokio::time::Instant::now() > deadline {
            return Err(ExecutorError::Io(std::io::Error::other(format!(
                "Timed out waiting for OpenCode server to print listening URL.\nServer output tail:\n{}",
                format_tail(captured)
            ))));
        }

        let line = match tokio::time::timeout_at(deadline, lines.next_line()).await {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                return Err(ExecutorError::Io(std::io::Error::other(format!(
                    "OpenCode server exited before printing listening URL.\nServer output tail:\n{}",
                    format_tail(captured)
                ))));
            }
            Ok(Err(err)) => return Err(ExecutorError::Io(err)),
            Err(_) => continue,
        };

        if captured.len() < 64 {
            captured.push(line.clone());
        }

        if let Some(url) = line.trim().strip_prefix("opencode server listening on ") {
            // Keep draining stdout to avoid backpressure on the server, but don't block startup.
            tokio::spawn(async move {
                let mut lines = tokio::io::BufReader::new(lines.into_inner()).lines();
                while let Ok(Some(_)) = lines.next_line().await {}
            });
            return Ok(url.trim().to_string());
        }
    }
}

#[async_trait]
impl StandardCodingAgentExecutor for Opencode {
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
        let env = setup_approvals_env(self.auto_approve, env);
        self.spawn_inner(current_dir, prompt, None, &env).await
    }

    async fn spawn_follow_up(
        &self,
        current_dir: &Path,
        prompt: &str,
        session_id: &str,
        env: &ExecutionEnv,
    ) -> Result<SpawnedChild, ExecutorError> {
        let env = setup_approvals_env(self.auto_approve, env);
        self.spawn_inner(current_dir, prompt, Some(session_id), &env)
            .await
    }

    fn normalize_logs(&self, msg_store: Arc<MsgStore>, worktree_path: &Path) {
        normalize_logs::normalize_logs(msg_store, worktree_path);
    }

    fn default_mcp_config_path(&self) -> Option<std::path::PathBuf> {
        #[cfg(unix)]
        {
            xdg::BaseDirectories::with_prefix("opencode").get_config_file("opencode.json")
        }
        #[cfg(not(unix))]
        {
            dirs::config_dir().map(|config| config.join("opencode").join("opencode.json"))
        }
    }

    fn get_availability_info(&self) -> AvailabilityInfo {
        let mcp_config_found = self
            .default_mcp_config_path()
            .map(|p| p.exists())
            .unwrap_or(false);

        let installation_indicator_found = dirs::config_dir()
            .map(|config| config.join("opencode").exists())
            .unwrap_or(false);

        if mcp_config_found || installation_indicator_found {
            AvailabilityInfo::InstallationFound
        } else {
            AvailabilityInfo::NotFound
        }
    }
}

fn default_to_true() -> bool {
    true
}

fn setup_approvals_env(auto_approve: bool, env: &ExecutionEnv) -> ExecutionEnv {
    let mut env = env.clone();
    if !auto_approve && !env.contains_key("OPENCODE_PERMISSION") {
        env.insert("OPENCODE_PERMISSION", r#"{"edit": "ask", "bash": "ask", "webfetch": "ask", "doom_loop": "ask", "external_directory": "ask"}"#);
    }
    env
}
