import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { after, test } from "node:test";
import guard from "../extensions/tool-guard.ts";

const originalCwd = process.cwd();
const temp = mkdtempSync(join(tmpdir(), "pi-tool-guard-"));
const root = join(temp, "checkout");
const outside = join(temp, "secret.txt");
mkdirSync(root);
writeFileSync(outside, "outside");
writeFileSync(join(root, "README.md"), "inside");
symlinkSync(outside, join(root, "outside-link"));
symlinkSync(temp, join(root, "outside-dir"));
symlinkSync(join(root, "README.md"), join(root, "inside-link"));
writeFileSync(join(root, "ambiguous\u00a0path"), "inside");
symlinkSync(outside, join(root, "ambiguous path"));
symlinkSync(outside, join(root, "decomposed-e\u0301"));
process.chdir(root);
process.env.PI_ALLOWED_TOOLS = " read , grep,find, ls ,write,edit,submit_result,trusted_extension";
let handler;
guard({ on(event, callback) {
  assert.equal(event, "tool_call");
  handler = callback;
} });
after(() => {
  process.chdir(originalCwd);
  rmSync(temp, { recursive: true, force: true });
});

function call(toolName, input = {}) {
  return handler({ toolName, input });
}

function blocked(toolName, input) {
  assert.equal(call(toolName, input)?.block, true, `${toolName}: ${JSON.stringify(input)}`);
}

test("allows in-root reads and explicitly trusted extension tools", () => {
  for (const path of ["README.md", join(root, "README.md"), "inside-link"]) {
    assert.equal(call("read", { path }), undefined, path);
  }
  assert.equal(call("grep", { pattern: "needle", glob: "**/*.md" }), undefined);
  assert.equal(call("find", { pattern: "**/*.md" }), undefined);
  assert.equal(call("ls"), undefined);
  assert.equal(call("submit_result", { verdict: "ready" }), undefined);
  assert.equal(call("trusted_extension"), undefined);
});

test("blocks aliases and missing paths that pi could normalize outside the root", () => {
  for (const tool of ["read", "grep", "find", "ls"]) {
    for (const path of [
      `@${outside}`, "@../secret.txt", "@~/.aws/credentials",
      pathToFileURL(outside).href, "ambiguous\u00a0path", "decomposed-\u00e9", "missing.txt",
    ]) {
      blocked(tool, { path });
    }
  }
});

test("blocks tools outside the allowlist", () => {
  blocked("bash", { command: "true" });
  blocked("unlisted_extension", {});
});

test("blocks write and edit even when allowlisted", () => {
  for (const tool of ["write", "edit"]) {
    blocked(tool, { path: "README.md", content: "changed" });
    blocked(tool, { path: outside, content: "changed" });
  }
});

test("blocks traversal, absolute escapes, home paths, and outward symlinks", () => {
  for (const tool of ["read", "grep", "find", "ls"]) {
    for (const path of ["../secret.txt", outside, "~/.aws/credentials", "outside-link", "outside-dir/secret.txt"]) {
      blocked(tool, { path });
    }
  }
});

test("blocks glob escapes but treats grep search text as text", () => {
  for (const glob of ["../*", "nested/../../*", "/tmp/*", "~/*"]) {
    blocked("find", { pattern: glob });
    blocked("grep", { pattern: "needle", glob });
  }
  assert.equal(call("grep", { pattern: "../literal" }), undefined);
});
