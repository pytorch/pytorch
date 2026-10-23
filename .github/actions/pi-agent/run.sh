#!/usr/bin/env bash
# Runs one hermetic, non-interactive pi session against Amazon Bedrock.
# Inputs arrive as env vars from action.yml; see that file for meanings.
set -euo pipefail

out="$PI_OUTPUT_DIR"
# Start clean: a second invocation in the same job must not see the previous
# session's result.json, events, or transcript.
rm -rf "$out"
mkdir -p "$out/session"
export PATH="$RUNNER_TEMP/pi-install/node_modules/.bin:$PATH"

# Isolate pi's user configuration from the runner's home directory.
export PI_CODING_AGENT_DIR="$RUNNER_TEMP/pi-agent-config"
mkdir -p "$PI_CODING_AGENT_DIR"
# Pinned definitions for allowlisted models newer than pi's bundled catalog, so
# they run without fetching the catalog from pi.dev (blocked under lockdown).
cp "$PI_ACTION_PATH/models.json" "$PI_CODING_AGENT_DIR/models.json"
export PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0

allowlist="$PI_ACTION_PATH/allowed-models.json"
if [[ "$PI_ALLOW_UNLISTED_MODEL" != "true" ]] \
  && ! jq -e --arg m "$PI_MODEL" 'any(.models[]; .id == $m)' "$allowlist" > /dev/null; then
  echo "::error::$PI_MODEL is not in allowed-models.json. Allowed:"
  jq -r '.models[] | "  \(.id)  (\(.use))"' "$allowlist"
  exit 1
fi

# prompt and result-schema accept literal text or a path to a file.
text_or_file() { if [[ -f "$1" ]]; then cat "$1"; else printf '%s' "$1"; fi; }
text_or_file "$PI_PROMPT" > "$out/prompt.md"

# tool-guard enforces the tool allowlist per call and confines path tools to
# the working directory; submit-result adds the schema-validated result tool.
tools="$PI_TOOLS"
extensions=(-e "$PI_ACTION_PATH/extensions/tool-guard.ts")
if [[ -n "$PI_RESULT_SCHEMA" ]]; then
  text_or_file "$PI_RESULT_SCHEMA" > "$out/result-schema.json"
  jq -e '.type == "object"' "$out/result-schema.json" > /dev/null \
    || { echo "::error::result-schema must be a JSON Schema with root type object"; exit 1; }
  export PI_RESULT_SCHEMA_FILE="$out/result-schema.json" PI_RESULT_FILE="$out/result.json"
  extensions+=(-e "$PI_ACTION_PATH/extensions/submit-result.ts")
  tools="${tools:+$tools,}submit_result"
fi
while IFS= read -r ext; do
  [[ -z "$ext" ]] || extensions+=(-e "$ext")
done <<< "$PI_EXTENSIONS"
skills=()
while IFS= read -r skill; do
  [[ -z "$skill" ]] || skills+=(--skill "$skill")
done <<< "$PI_SKILLS"
export PI_ALLOWED_TOOLS="$tools"

# Ignore everything discoverable from the checkout (.pi/ config, extensions,
# skills, AGENTS.md/CLAUDE.md): only files passed with -e and --skill load.
isolation=(--no-approve --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files)

# pi only warns on an unknown --model and falls back to another model. Refresh
# the catalog from pi.dev only when the bundled one lacks the ID.
known() {
  pi "${isolation[@]}" "${extensions[@]}" --list-models "$PI_MODEL" 2> /dev/null \
    | awk -v m="$PI_MODEL" '$1 == "amazon-bedrock" && $2 == m { f = 1 } END { exit !f }'
}
if ! known; then
  echo "$PI_MODEL is not in pi's bundled catalog; refreshing from pi.dev"
  PI_OFFLINE=0 pi update --models || true
  known || { echo "::error::amazon-bedrock/$PI_MODEL is not in pi's model catalog"; exit 1; }
fi

echo "pi $(pi --version) | amazon-bedrock/$PI_MODEL | thinking=$PI_THINKING | tools=$tools"
started=$(date +%s)
set +e
pi --mode json "${isolation[@]}" "${extensions[@]}" ${skills[@]+"${skills[@]}"} \
  --session-dir "$out/session" --provider amazon-bedrock --model "$PI_MODEL" \
  --thinking "$PI_THINKING" --tools "$tools" "@$out/prompt.md" 2> "$out/stderr.log" \
  | tee "$out/events.jsonl" \
  | jq --unbuffered -r 'select(.type == "tool_execution_start") | "→ \(.toolName) \(.args | tostring | .[0:240])"'
rc=${PIPESTATUS[0]}
set -e
duration_ms=$((($(date +%s) - started) * 1000))

session_file=$(find "$out/session" -name '*.jsonl' -print -quit)
[[ -z "$session_file" ]] || pi --export "$session_file" "$out/transcript.html" > /dev/null 2>&1 || true

events="$out/events.jsonl"
jq -s -r '[.[] | select(.type == "message_end" and .message.role == "assistant")] | last
  | (.message.content // []) | map(select(.type == "text") | .text) | join("\n")' "$events" > "$out/final.txt" || true
jq -s '[.[] | select(.type == "message_end" and .message.role == "assistant") | .message] as $m
  | {turns: ($m | length),
     input: ([$m[].usage.input // 0] | add // 0),
     output: ([$m[].usage.output // 0] | add // 0),
     cache_read: ([$m[].usage.cacheRead // 0] | add // 0),
     cache_write: ([$m[].usage.cacheWrite // 0] | add // 0),
     cost_usd: ([$m[].usage.cost.total // 0] | add // 0),
     last_stop: ($m | last | .stopReason // null),
     last_error: ($m | last | .errorMessage // null)}' "$events" > "$out/usage.json" 2> /dev/null \
  || echo '{}' > "$out/usage.json"

# Claude Code execution-file shape, for the review log inspector and usage upload.
jq -s --arg version "$(pi --version)" --arg model "$PI_MODEL" --argjson duration_ms "$duration_ms" \
  -f "$PI_ACTION_PATH/claude-execution.jq" "$events" > "$out/execution.json" || rm -f "$out/execution.json"

# Every tool declared to the model or run successfully must be in the allowlist.
declared=$(jq -r 'select(.type == "message_end" and .message.role == "system") | .message.toolsAdded[]?.name' "$events" | sort -u)
executed=$(jq -r 'select(.type == "tool_execution_end" and (.isError | not)) | .toolName' "$events" | sort -u)
blocked=$(jq -r 'select(.type == "tool_execution_end" and .isError) | .result.content[0].text // ""' "$events" \
  | grep -c -E "is not allowed in this run|is limited to |^Tool .* not found" || true)
unexpected=$(comm -23 <(printf '%s\n%s\n' "$declared" "$executed" | grep -v '^$' | sort -u) \
  <(tr ',' '\n' <<< "$tools" | sort -u) | paste -sd ' ' -)

last_stop=$(jq -r '.last_stop // ""' "$out/usage.json")
error=""
if [[ $rc -ne 0 ]]; then
  error="pi exited with $rc: $(tail -n 3 "$out/stderr.log" | tr '\n' ' ')"
elif [[ "$last_stop" =~ ^(error|aborted)$ ]]; then
  # pi exits 0 when the provider call fails (auth, throttling, model access).
  error="model call failed: $(jq -r .last_error "$out/usage.json")"
elif [[ -n "$unexpected" ]]; then
  error="tools outside the allowlist were declared or used: $unexpected"
elif [[ -n "$PI_RESULT_SCHEMA" && ! -s "$out/result.json" ]]; then
  error="model finished without calling submit_result (last stop: $last_stop)"
fi

{
  if [[ -z "$error" ]]; then
    echo "### pi · \`$PI_MODEL\` · ✅ completed"
  else
    echo "### pi · \`$PI_MODEL\` · ❌ ${error:0:300}"
  fi
  echo
  echo "| Turns | Input tokens | Output tokens | Cache read | Cache write | Cost (USD) |"
  echo "|---|---|---|---|---|---|"
  jq -r '"| \(.turns // 0) | \(.input // 0) | \(.output // 0) | \(.cache_read // 0) | \(.cache_write // 0) | \((.cost_usd // 0) * 10000 | round / 10000) |"' "$out/usage.json"
  echo
  echo "**Tools** offered: $(paste -sd ' ' - <<< "$declared") · used: $(paste -sd ' ' - <<< "$executed") · blocked calls: $blocked"
  echo
  if [[ -s "$out/result.json" ]]; then
    fence='```'
    printf '<details><summary>Result</summary>\n\n%sjson\n%s\n%s\n</details>\n\n' "$fence" "$(jq . "$out/result.json")" "$fence"
  elif [[ -s "$out/final.txt" ]]; then
    printf '<details><summary>Final message</summary>\n\n%s\n</details>\n\n' "$(head -c 4000 "$out/final.txt")"
  fi
} >> "$GITHUB_STEP_SUMMARY"

# The result is model-controlled: use an unguessable heredoc delimiter.
delim="PI_EOF_$(openssl rand -hex 16)"
{
  [[ ! -s "$out/execution.json" ]] || echo "execution-file=$out/execution.json"
  if [[ -s "$out/result.json" ]]; then
    echo "structured_output<<$delim"
    jq -c . "$out/result.json"
    echo "$delim"
  fi
} >> "$GITHUB_OUTPUT"

if [[ -n "$error" ]]; then
  echo "::error::$error"
  exit 1
fi
