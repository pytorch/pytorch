/**
 * submit_result: structured final answer for CI runs (pi's equivalent of
 * claude-code-action's `--json-schema`).
 *
 * The tool's parameter schema is the caller's JSON Schema, so pi validates the
 * model's arguments before execute() runs and returns validation errors to the
 * model for a retry. The validated object is written to PI_RESULT_FILE.
 *
 * Env:
 *   PI_RESULT_SCHEMA_FILE  JSON Schema (root must be {"type": "object", ...})
 *   PI_RESULT_FILE         where the validated result is written
 */

import { readFileSync, writeFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const MAX_REMINDERS = 2;

export default function (pi: ExtensionAPI) {
  const schemaFile = process.env.PI_RESULT_SCHEMA_FILE;
  const resultFile = process.env.PI_RESULT_FILE;
  if (!schemaFile || !resultFile) {
    throw new Error("submit-result: PI_RESULT_SCHEMA_FILE and PI_RESULT_FILE must be set");
  }
  const schema = JSON.parse(readFileSync(schemaFile, "utf8"));
  if (schema?.type !== "object") {
    throw new Error('submit-result: schema root must be {"type": "object"}');
  }

  let submitted = false;
  let reminders = 0;

  pi.registerTool({
    name: "submit_result",
    label: "Submit Result",
    description:
      "Submit your final machine-readable answer. Call exactly once, as your last action, after the investigation is complete.",
    promptSnippet: "Submit the final structured answer (required to finish)",
    promptGuidelines: [
      "The run only counts if you call submit_result with arguments matching its schema.",
      "Call submit_result as your final action; do not write anything after it.",
    ],
    parameters: schema,
    async execute(_toolCallId, params) {
      writeFileSync(resultFile, JSON.stringify(params, null, 2));
      submitted = true;
      return {
        content: [{ type: "text", text: "Result recorded." }],
        details: undefined,
        terminate: true,
      };
    },
  });

  // Models sometimes end with prose instead of the tool call. Nudge them back
  // a bounded number of times rather than failing the run outright. Do not
  // gate on event.context.canContinue: it describes the context before this
  // handler's entry, which is what makes the continuation valid.
  pi.on("agent_before_settle", (event) => {
    if (submitted || reminders >= MAX_REMINDERS || event.outcome !== "completed") return;
    reminders++;
    return {
      entries: [
        {
          type: "custom_message",
          customType: "submit-result-reminder",
          content:
            "You have not called submit_result. Call it now with your final answer; it is the only output this run records.",
          display: false,
        },
      ],
      continue: true,
    };
  });
}
