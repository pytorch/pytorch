/**
 * Defense in depth for CI runs, loaded by .github/actions/pi-agent for every session.
 *
 * - Blocks any tool call whose name is not in PI_ALLOWED_TOOLS, even if pi or
 *   another extension activates a tool later in the session.
 * - Confines read/grep/find/ls to the working directory, so the model cannot read
 *   runner secrets such as /proc/self/environ (AWS session credentials) or
 *   ~/.aws, /tmp, and RUNNER_TEMP.
 *
 * Env:
 *   PI_ALLOWED_TOOLS  comma-separated tool names
 */

import { realpathSync } from "node:fs";
import { isAbsolute, relative, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PATH_TOOLS = new Set(["read", "grep", "find", "ls"]);

export default function (pi: ExtensionAPI) {
  const allowed = new Set(
    (process.env.PI_ALLOWED_TOOLS ?? "")
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean),
  );
  if (allowed.size === 0) {
    throw new Error("tool-guard: PI_ALLOWED_TOOLS must list the session's tools");
  }
  const root = realpathSync(process.cwd());

  function insideRoot(path: string): boolean {
    const target = resolve(root, path);
    let real: string;
    try {
      real = realpathSync(target);
    } catch {
      real = target; // nonexistent paths fail in the tool itself; still check the lexical path
    }
    const rel = relative(root, real);
    return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
  }

  pi.on("tool_call", (event) => {
    if (!allowed.has(event.toolName)) {
      return { block: true, reason: `tool ${event.toolName} is not allowed in this run` };
    }
    if (PATH_TOOLS.has(event.toolName)) {
      const input = event.input as { path?: unknown; pattern?: unknown; glob?: unknown };
      const escapes =
        (typeof input.path === "string" && (input.path.startsWith("~") || !insideRoot(input.path))) ||
        // find's pattern and grep's glob are globs; keep them relative and inside the root.
        [event.toolName === "find" ? input.pattern : undefined, input.glob].some(
          (glob) => typeof glob === "string" && /(^[/~]|(^|[/\\])\.\.([/\\]|$))/.test(glob),
        );
      if (escapes) {
        return { block: true, reason: `${event.toolName} is limited to ${root}` };
      }
    }
  });
}
