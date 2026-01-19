export function isSlashCommandPrompt(prompt: string): boolean {
  const trimmed = prompt.trimStart();
  if (!trimmed.startsWith('/')) return false;

  const match = /^\/([^\s/]+)(?:\s|$)/.exec(trimmed);
  if (!match) return false;

  return true;
}
