// Runs against the lockfile install, not the fake pi used by test_run.py.
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createFindTool, createGrepTool, createReadTool, createLsTool } from "@earendil-works/pi-coding-agent";

assert.equal(process.env.PI_OFFLINE, "1");
const root = mkdtempSync(join(process.env.RUNNER_TEMP, "pi-real-tools-"));
writeFileSync(join(root, "probe-target.txt"), "pi_runner_probe_7391\n");
try {
  for (const [tool, params, expected] of [
    [createGrepTool(root), { pattern: "pi_runner_probe_7391", path: root }, "pi_runner_probe_7391"],
    [createFindTool(root), { pattern: "probe-target.txt", path: root }, "probe-target.txt"],
    [createReadTool(root), { path: "probe-target.txt" }, "pi_runner_probe_7391"],
    [createLsTool(root), { path: root }, "probe-target.txt"],
  ]) {
    const result = await tool.execute("probe-" + tool.name, params, AbortSignal.timeout(30000));
    assert.notEqual(result.isError, true, JSON.stringify(result));
    assert.ok(JSON.stringify(result.content).includes(expected), JSON.stringify(result));
    console.log(`PASS ${tool.name}: ${JSON.stringify(result.content)}`);
  }
} finally {
  rmSync(root, { recursive: true, force: true });
}
