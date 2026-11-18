/**
 * Defense in depth for CI runs, loaded by .github/actions/pi-agent for every session.
 *
 * - Blocks any tool call whose name is not in PI_ALLOWED_TOOLS, even if pi or
 *   another extension activates a tool later in the session. write/edit are
 *   always blocked, even if listed; this action does not confine file mutations.
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
    // pi strips @, expands file URLs, and normalizes these spaces before opening.
    // Reject aliases rather than validating a different path from the tool.
    if (/^[@~]|^file:\/\/|[\u00a0\u2000-\u200a\u202f\u205f\u3000]/i.test(path)) return false;
    const target = resolve(root, path);
    let real: string;
    try {
      real = realpathSync(target);
    } catch {
      // read also tries alternate Unicode spellings when a path is missing.
      return false;
    }
    const rel = relative(root, real);
    return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel));
  }

  pi.on("tool_call", (event) => {
    if (!allowed.has(event.toolName) || event.toolName === "write" || event.toolName === "edit") {
      return { block: true, reason: `tool ${event.toolName} is not allowed in this run` };
    }
    if (PATH_TOOLS.has(event.toolName)) {
      const input = event.input as { path?: unknown; pattern?: unknown; glob?: unknown };
      const escapes =
        (typeof input.path === "string" && !insideRoot(input.path)) ||
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
