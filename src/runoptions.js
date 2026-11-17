/** Prompts and token budget checked before a build or an agent run. */
export function readRunOptions(promptsText, maxTokens) {
  const prompts = String(promptsText ?? "").split("\n").map((s) => s.trim()).filter(Boolean);
  if (!prompts.length) throw new Error("Add at least one prompt.");
  const n = Math.round(Number(maxTokens));
  if (!Number.isFinite(n) || n < 8 || n > 512) throw new Error("Max tokens must be between 8 and 512.");
  return { prompts, maxTokens: n };
}
