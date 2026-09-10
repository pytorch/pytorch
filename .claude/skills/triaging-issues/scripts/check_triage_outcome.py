#!/usr/bin/env python3
"""Verify that a triage run actually did something to the issue.

The agent's own success flag is not trustworthy: a run can finish green while
every tool call failed and the final message asks for an approval nobody can
give. This script classifies the issue from its live state on GitHub plus the
execution transcript, re-applies a missing bot-triaged marker, and fails the
job when no triage action happened so the workflow can retry once.

Usage: check_triage_outcome.py <owner/repo> <issue_number> <transcript.json>
"""

import json
import os
import sys

from gh_api import gh_api


MUTATION_TOOLS = {
    "mcp__github__update_issue",
    "mcp__github__add_issue_comment",
    "mcp__github__issue_write",
    "mcp__github__transfer_issue",
}
BOT_TRIAGED_LABEL = "bot-triaged"


def summarize_transcript(path: str) -> dict:
    summary = {
        "present": False,
        "mutated": False,
        "final_text": "",
        "tool_errors": [],
        "permission_denials": [],
    }
    if not os.path.exists(path):
        return summary
    with open(path) as f:
        messages = json.load(f)
    summary["present"] = True

    tool_names: dict[str, str] = {}
    for m in messages:
        if m.get("type") == "assistant":
            for block in m.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_names[block["id"]] = block["name"]
        elif m.get("type") == "user":
            content = m.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") != "tool_result":
                    continue
                name = tool_names.get(block.get("tool_use_id"), "?")
                if block.get("is_error"):
                    text = block.get("content")
                    if not isinstance(text, str):
                        text = json.dumps(text)
                    summary["tool_errors"].append(f"{name}: {text[:200]}")
                elif name in MUTATION_TOOLS:
                    summary["mutated"] = True
        elif m.get("type") == "result":
            summary["final_text"] = str(m.get("result", ""))[:1000]
            summary["permission_denials"] = [
                d.get("tool_name") for d in m.get("permission_denials", [])
            ]
    return summary


def classify(labels: set[str], state: str, mutated: bool) -> str:
    if any(label.startswith("oncall:") for label in labels):
        return "routed"
    if "triaged" in labels:
        return "triaged"
    if "triage review" in labels:
        return "review"
    if state == "closed":
        return "closed"
    if mutated:
        return "waiting_on_reporter"
    return "no_action"


def main() -> int:
    repo, issue_number, transcript_path = sys.argv[1:4]
    transcript = summarize_transcript(transcript_path)

    issue = json.loads(gh_api([f"repos/{repo}/issues/{issue_number}"], log=print))
    labels = {label["name"] for label in issue.get("labels", [])}
    outcome = classify(labels, issue.get("state", ""), transcript["mutated"])

    print(f"Transcript present: {transcript['present']}")
    print(f"Mutation succeeded: {transcript['mutated']}")
    print(f"Labels now: {sorted(labels)}")
    print(f"State: {issue.get('state')}")
    print(f"Outcome: {outcome}")

    if transcript["mutated"] and BOT_TRIAGED_LABEL not in labels:
        print(f"Re-applying missing {BOT_TRIAGED_LABEL} marker")
        endpoint = f"repos/{repo}/issues/{issue_number}/labels"
        gh_api(
            ["-X", "POST", endpoint, "-f", f"labels[]={BOT_TRIAGED_LABEL}"], log=print
        )

    if github_output := os.environ.get("GITHUB_OUTPUT"):
        with open(github_output, "a") as f:
            f.write(f"outcome={outcome}\n")

    if outcome != "no_action":
        return 0

    if not transcript["present"]:
        print("::error::No execution transcript was produced (timeout or crash?)")
    print(f"::error::Triage run took no action on #{issue_number}")
    if transcript["tool_errors"]:
        print("Tool errors:")
        for err in transcript["tool_errors"]:
            print(f"  {err}")
    if transcript["permission_denials"]:
        print(f"Permission denials: {transcript['permission_denials']}")
    if transcript["final_text"]:
        print("Agent's final message:")
        print(transcript["final_text"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
