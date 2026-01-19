import { useQuery } from '@tanstack/react-query';
import type { BaseCodingAgent, SlashCommand } from 'shared/types';
import { agentsApi } from '@/lib/api';

type Options = {
  enabled?: boolean;
  refetchInterval?: number | false;
};

const DEFAULT_REFETCH_INTERVAL_MS = 60_000;

export function useAgentSlashCommands(
  agent: BaseCodingAgent | null | undefined,
  taskAttemptId?: string,
  opts?: Options
) {
  const enabled = (opts?.enabled ?? true) && !!agent;
  const refetchInterval = opts?.refetchInterval ?? DEFAULT_REFETCH_INTERVAL_MS;

  return useQuery<SlashCommand[]>({
    queryKey: ['agentSlashCommands', agent, taskAttemptId],
    queryFn: () => agentsApi.getSlashCommands(agent!, taskAttemptId),
    enabled,
    staleTime: typeof refetchInterval === 'number' ? refetchInterval : 0,
    refetchInterval: enabled ? refetchInterval : false,
  });
}
