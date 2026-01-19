import { useCallback, useMemo } from 'react';
import type { BaseCodingAgent, SlashCommand } from 'shared/types';
import { useJsonPatchWsStream } from '@/hooks/useJsonPatchWsStream';

type SlashCommandsStreamState = {
  commands: SlashCommand[];
  discovering: boolean;
  error: string | null;
};

export function useAgentSlashCommandsStream(
  agent: BaseCodingAgent | null | undefined,
  opts?: { taskAttemptId?: string; repoId?: string }
) {
  const endpoint = useMemo(() => {
    if (!agent) return undefined;

    const params = new URLSearchParams();
    params.set('executor', agent);
    if (opts?.taskAttemptId) params.set('task_attempt_id', opts.taskAttemptId);
    if (opts?.repoId) params.set('repo_id', opts.repoId);

    return `/api/agents/slash-commands/ws?${params.toString()}`;
  }, [agent, opts?.repoId, opts?.taskAttemptId]);

  const initialData = useCallback(
    (): SlashCommandsStreamState => ({
      commands: [],
      discovering: false,
      error: null,
    }),
    []
  );

  const { data, error, isConnected, isInitialized } =
    useJsonPatchWsStream<SlashCommandsStreamState>(
      endpoint,
      !!agent,
      initialData
    );

  return {
    commands: data?.commands ?? [],
    discovering: data?.discovering ?? false,
    error: data?.error ?? error,
    isConnected,
    isInitialized,
  };
}

