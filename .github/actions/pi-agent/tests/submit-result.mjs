import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import submitResult from "../extensions/submit-result.ts";

function session(t, schema = { type: "object", properties: { verdict: { type: "string" } } }) {
  const temp = mkdtempSync(join(tmpdir(), "pi-submit-result-"));
  const keys = ["PI_RESULT_SCHEMA_FILE", "PI_RESULT_FILE"];
  const previous = keys.map((key) => process.env[key]);
  t.after(() => {
    keys.forEach((key, i) => {
      if (previous[i] === undefined) delete process.env[key];
      else process.env[key] = previous[i];
    });
    rmSync(temp, { recursive: true, force: true });
  });
  process.env.PI_RESULT_SCHEMA_FILE = join(temp, "schema.json");
  process.env.PI_RESULT_FILE = join(temp, "result.json");
  writeFileSync(process.env.PI_RESULT_SCHEMA_FILE, JSON.stringify(schema));
  let tool;
  let settle;
  submitResult({
    registerTool(value) { tool = value; },
    on(event, handler) {
      assert.equal(event, "agent_before_settle");
      settle = handler;
    },
  });
  return { tool, settle, resultFile: process.env.PI_RESULT_FILE, schema };
}

test("forwards the caller schema and records a terminating result", async (t) => {
  const { tool, settle, resultFile, schema } = session(t);
  assert.equal(tool.name, "submit_result");
  // pi owns argument validation; the extension must register the caller's schema.
  assert.deepEqual(tool.parameters, schema);
  const value = { verdict: "ready" };
  const result = await tool.execute("call-id", value);
  assert.deepEqual(JSON.parse(readFileSync(resultFile, "utf8")), value);
  assert.equal(result.terminate, true);
  assert.equal(settle({ outcome: "completed" }), undefined);
});

test("reminds only completed sessions and stops after two reminders", (t) => {
  const { settle } = session(t);
  assert.equal(settle({ outcome: "error" }), undefined);
  assert.equal(settle({ outcome: "aborted" }), undefined);
  assert.equal(settle({ outcome: "completed" }).continue, true);
  assert.equal(settle({ outcome: "completed" }).continue, true);
  assert.equal(settle({ outcome: "completed" }), undefined);
});

test("rejects missing configuration and non-object root schemas", (t) => {
  const { resultFile } = session(t);
  delete process.env.PI_RESULT_FILE;
  assert.throws(() => submitResult({}), /must be set/);
  process.env.PI_RESULT_FILE = resultFile;
  writeFileSync(process.env.PI_RESULT_SCHEMA_FILE, JSON.stringify({ type: "array" }));
  assert.throws(() => submitResult({}), /schema root must be/);
});
