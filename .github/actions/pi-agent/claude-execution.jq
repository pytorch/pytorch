# Converts pi's JSON event stream (jq -s input) into the execution-file shape
# that claude-code-action writes, so existing consumers keep working unchanged:
# the review log inspector (scripts/review_log_inspector.py) and
# pytorch/test-infra's upload-claude-usage action.
#
# Args: $version (pi version), $model, $duration_ms (number).

def iso: (. / 1000 | todate);

([.[] | select(.type == "session")] | first) as $session
| [.[] | select(.type == "message_end") | .message] as $messages
| [$messages[] | select(.role == "assistant")] as $assistant
| ($assistant | last) as $last
| [
    {
      type: "system",
      subtype: "init",
      session_id: ($session.id // ""),
      model: $model,
      claude_code_version: ("pi " + $version),
      tools: [$messages[] | select(.role == "system") | .toolsAdded[]?.name]
    }
  ]
  + [
    $messages[]
    | select(.role == "assistant" or .role == "toolResult")
    | if .role == "assistant" then
        {
          type: "assistant",
          uuid: (.timestamp | tostring),
          timestamp: (.timestamp | iso),
          message: {
            content: [
              .content[]?
              | if .type == "text" then {type: "text", text}
                elif .type == "thinking" then {type: "thinking", thinking: (.thinking // "")}
                elif .type == "toolCall" then {type: "tool_use", id, name, input: (.arguments // {})}
                else empty end
            ]
          }
        }
      else
        {
          type: "user",
          timestamp: (.timestamp | iso),
          message: {
            content: [
              {
                type: "tool_result",
                tool_use_id: .toolCallId,
                is_error: (.isError // false),
                content: [.content[]? | select(.type == "text") | {type: "text", text}]
              }
            ]
          }
        }
      end
  ]
  + [
    {
      type: "result",
      subtype: (if ($last.stopReason // "") | test("^(error|aborted)$") then "error_during_execution" else "success" end),
      result: ([$last.content[]? | select(.type == "text") | .text] | join("\n")),
      duration_ms: $duration_ms,
      num_turns: ($assistant | length),
      total_cost_usd: ([$assistant[].usage.cost.total // 0] | add // 0),
      usage: {
        input_tokens: ([$assistant[].usage.input // 0] | add // 0),
        output_tokens: ([$assistant[].usage.output // 0] | add // 0),
        cache_read_input_tokens: ([$assistant[].usage.cacheRead // 0] | add // 0),
        cache_creation_input_tokens: ([$assistant[].usage.cacheWrite // 0] | add // 0)
      },
      modelUsage: {($model): {}},
      permission_denials: []
    }
  ]
