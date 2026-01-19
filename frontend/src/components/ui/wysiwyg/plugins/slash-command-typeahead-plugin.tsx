import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import {
  LexicalTypeaheadMenuPlugin,
  MenuOption,
} from '@lexical/react/LexicalTypeaheadMenuPlugin';
import { $createTextNode } from 'lexical';
import { Command as CommandIcon } from 'lucide-react';
import type { BaseCodingAgent, SlashCommand } from 'shared/types';
import { usePortalContainer } from '@/contexts/PortalContainerContext';
import { useAgentSlashCommandsStream } from '@/hooks/useAgentSlashCommandsStream';
import { useTaskAttemptId } from '@/components/ui/wysiwyg/context/task-attempt-context';

class SlashCommandOption extends MenuOption {
  command: SlashCommand;

  constructor(command: SlashCommand) {
    super(`slash-command-${command.name}`);
    this.command = command;
  }
}

const VIEWPORT_MARGIN = 8;
const VERTICAL_GAP = 4;
const VERTICAL_GAP_ABOVE = 24;
const MIN_WIDTH = 360;

function getMenuPosition(anchorEl: HTMLElement) {
  const rect = anchorEl.getBoundingClientRect();
  const viewportHeight = window.innerHeight;
  const viewportWidth = window.innerWidth;

  const spaceAbove = rect.top;
  const spaceBelow = viewportHeight - rect.bottom;

  const showBelow = spaceBelow >= spaceAbove;

  let top: number | undefined;
  let bottom: number | undefined;

  if (showBelow) {
    top = rect.bottom + VERTICAL_GAP;
  } else {
    bottom = viewportHeight - rect.top + VERTICAL_GAP_ABOVE;
  }

  let left = rect.left;
  const maxLeft = viewportWidth - MIN_WIDTH - VIEWPORT_MARGIN;
  if (left > maxLeft) {
    left = Math.max(VIEWPORT_MARGIN, maxLeft);
  }

  return { top, bottom, left };
}

function filterSlashCommands(all: SlashCommand[], query: string): SlashCommand[] {
  const q = query.trim().toLowerCase();
  if (!q) return all;

  const startsWith = all.filter((c) => c.name.toLowerCase().startsWith(q));
  const includes = all.filter(
    (c) => !startsWith.includes(c) && c.name.toLowerCase().includes(q)
  );
  return [...startsWith, ...includes];
}

export function SlashCommandTypeaheadPlugin({
  agent,
}: {
  agent: BaseCodingAgent | null;
}) {
  const [editor] = useLexicalComposerContext();
  const portalContainer = usePortalContainer();
  const taskAttemptId = useTaskAttemptId();
  const [options, setOptions] = useState<SlashCommandOption[]>([]);
  const [activeQuery, setActiveQuery] = useState<string | null>(null);
  const lastMousePositionRef = useRef<{ x: number; y: number } | null>(null);

  const slashCommandsQuery = useAgentSlashCommandsStream(agent, { taskAttemptId });
  const allCommands = useMemo(
    () => slashCommandsQuery.commands ?? [],
    [slashCommandsQuery.commands]
  );
  const isLoading = !slashCommandsQuery.isInitialized && !!agent;
  const isDiscovering = slashCommandsQuery.discovering;

  useEffect(() => {
    if (!slashCommandsQuery.error) return;
    console.error('Failed to fetch slash commands', slashCommandsQuery.error);
  }, [slashCommandsQuery.error]);

  const updateOptions = useCallback(
    (query: string | null) => {
      setActiveQuery(query);

      if (!agent || query === null) {
        setOptions([]);
        return;
      }

      const filtered = filterSlashCommands(allCommands, query).slice(0, 20);
      setOptions(filtered.map((c) => new SlashCommandOption(c)));
    },
    [agent, allCommands]
  );

  const hasVisibleResults = useMemo(() => {
    if (!agent || activeQuery === null) return false;
    if (isLoading || isDiscovering) return true;
    if (!activeQuery.trim()) return true;
    return options.length > 0;
  }, [agent, activeQuery, isDiscovering, isLoading, options.length]);

  // If command list loads while menu is open, refresh options.
  useEffect(() => {
    if (activeQuery === null) return;
    updateOptions(activeQuery);
  }, [activeQuery, updateOptions]);

  return (
    <LexicalTypeaheadMenuPlugin<SlashCommandOption>
      triggerFn={(text) => {
        const match = /^(\s*)\/([^\s/]*)$/.exec(text);
        if (!match) return null;

        const slashOffset = match[1].length;
        return {
          leadOffset: slashOffset,
          matchingString: match[2],
          replaceableString: match[0].slice(slashOffset),
        };
      }}
      options={options}
      onQueryChange={updateOptions}
      onSelectOption={(option, nodeToReplace, closeMenu) => {
        editor.update(() => {
          if (!nodeToReplace) return;

          const textToInsert = `/${option.command.name}`;
          const commandNode = $createTextNode(textToInsert);
          nodeToReplace.replace(commandNode);

          const spaceNode = $createTextNode(' ');
          commandNode.insertAfter(spaceNode);
          spaceNode.select(1, 1);
        });

        closeMenu();
      }}
      menuRenderFn={(
        anchorRef,
        { selectedIndex, selectOptionAndCleanUp, setHighlightedIndex }
      ) => {
        if (!anchorRef.current) return null;
        if (!agent) return null;
        if (!hasVisibleResults) return null;

        const { top, bottom, left } = getMenuPosition(anchorRef.current);
        const isEmpty = !isLoading && !isDiscovering && allCommands.length === 0;
        const showLoadingRow = isLoading || isDiscovering;
        const loadingText = isLoading ? 'Loading commands…' : 'Discovering commands…';

        return createPortal(
          <div
            className="fixed bg-background border border-border rounded-md shadow-lg overflow-hidden"
            style={{
              top,
              bottom,
              left,
              minWidth: MIN_WIDTH,
              zIndex: 10000,
            }}
          >
            <div className="px-3 py-2 border-b bg-muted/30">
              <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                <CommandIcon className="h-3.5 w-3.5" />
                Commands
              </div>
            </div>

            {isEmpty ? (
              <div className="p-3 text-sm text-muted-foreground">
                No commands available for this agent.
              </div>
            ) : options.length === 0 && !showLoadingRow ? null : (
              <div className="py-1 max-h-[40vh] overflow-auto">
                {showLoadingRow && (
                  <div className="px-3 py-2 text-sm text-muted-foreground select-none">
                    {loadingText}
                  </div>
                )}
                {options.map((option) => {
                  const index = options.indexOf(option);
                  const details =
                    option.command.description ?? option.command.usage ?? null;

                  return (
                    <div
                      key={option.key}
                      className={`px-3 py-2 cursor-pointer text-sm ${
                        index === selectedIndex
                          ? 'bg-muted text-foreground text-high'
                          : 'hover:bg-muted text-muted-foreground'
                      }`}
                      onMouseMove={(e) => {
                        const pos = { x: e.clientX, y: e.clientY };
                        const last = lastMousePositionRef.current;
                        if (!last || last.x !== pos.x || last.y !== pos.y) {
                          lastMousePositionRef.current = pos;
                          setHighlightedIndex(index);
                        }
                      }}
                      onClick={() => selectOptionAndCleanUp(option)}
                    >
                      <div className="flex items-center gap-2 font-medium">
                        <span className="font-mono">/{option.command.name}</span>
                      </div>
                      {details && (
                        <div className="text-xs mt-0.5 truncate text-muted-foreground">
                          {details}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>,
          portalContainer ?? document.body
        );
      }}
    />
  );
}
