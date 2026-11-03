/**
 * Read-only GitHub issue tools backed by `gh api` (no shell, no write calls).
 *
 * The repository is fixed by the workflow, so the model can read issues in
 * that repository only. Writes never happen here: the workflow applies the
 * model's submitted plan in a separate job that holds the write token.
 *
 * Env:
 *   GH_TOKEN                  read token for gh
 *   TRIAGE_REPO               owner/repo the tools may read
 *   TRIAGE_REPLAY_ISSUE       optional issue number to show as it was before the
 *                             triage bot acted: open, with the labels and the
 *                             human comments from before its first label event
 *                             (dry-run replays of already-triaged issues)
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const run = promisify(execFile);
const MAX_BODY = 20_000;
const MAX_COMMENT = 4_000;
const MAX_COMMENTS = 50;

const TRIAGE_BOT = "github-actions[bot]";
const repo = process.env.TRIAGE_REPO ?? "";
const replayIssue = Number(process.env.TRIAGE_REPLAY_ISSUE ?? 0);

async function ghApi(args: string[], signal?: AbortSignal): Promise<unknown> {
  const { stdout } = await run("gh", ["api", ...args], {
    maxBuffer: 32 * 1024 * 1024,
    timeout: 30_000,
    signal,
  });
  return JSON.parse(stdout);
}

function clip(text: string | null | undefined, limit: number): string {
  const value = text ?? "";
  return value.length <= limit ? value : `${value.slice(0, limit)}\n…[truncated ${value.length - limit} chars]`;
}

function text(value: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }], details: undefined };
}

interface PreTriage {
  labels: string[];
  cutoff?: string;
}

/** Labels and time of the issue just before the triage bot's first label event. */
async function preTriage(issue: number, signal?: AbortSignal): Promise<PreTriage> {
  const pages = (await ghApi(["--paginate", "--slurp", `repos/${repo}/issues/${issue}/events?per_page=100`], signal)) as any[][];
  const labels = new Set<string>();
  for (const event of pages.flat()) {
    if (event.event !== "labeled" && event.event !== "unlabeled") continue;
    if (event.actor?.login === TRIAGE_BOT) return { labels: [...labels], cutoff: event.created_at };
    if (event.event === "labeled") labels.add(event.label.name);
    else labels.delete(event.label.name);
  }
  return { labels: [...labels] };
}

export default function (pi: ExtensionAPI) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    throw new Error("github-issue-tools: TRIAGE_REPO must be owner/repo");
  }
  const issueNumber = Type.Integer({ minimum: 1, description: `Issue number in ${repo}` });

  pi.registerTool({
    name: "get_issue",
    label: "Get Issue",
    description: `Read one issue in ${repo}: title, body, labels, state, author.`,
    parameters: Type.Object({ issue_number: issueNumber }),
    async execute(_id, { issue_number }, signal) {
      const issue = (await ghApi([`repos/${repo}/issues/${issue_number}`], signal)) as any;
      const replay = issue_number === replayIssue;
      return text({
        number: issue.number,
        title: issue.title,
        state: replay ? "open" : issue.state,
        labels: replay ? (await preTriage(issue_number, signal)).labels : issue.labels.map((label: any) => label.name),
        author: issue.user?.login,
        author_association: issue.author_association,
        created_at: issue.created_at,
        is_pull_request: Boolean(issue.pull_request),
        body: clip(issue.body, MAX_BODY),
      });
    },
  });

  pi.registerTool({
    name: "get_issue_comments",
    label: "Get Issue Comments",
    description: `Read up to ${MAX_COMMENTS} comments on an issue in ${repo}.`,
    parameters: Type.Object({ issue_number: issueNumber }),
    async execute(_id, { issue_number }, signal) {
      let comments = (await ghApi([`repos/${repo}/issues/${issue_number}/comments?per_page=${MAX_COMMENTS}`], signal)) as any[];
      if (issue_number === replayIssue) {
        // The triage bot sometimes comments before its first label event.
        const { cutoff } = await preTriage(issue_number, signal);
        comments = comments.filter(
          (comment) => comment.user?.login !== TRIAGE_BOT && (!cutoff || comment.created_at < cutoff),
        );
      }
      return text(
        comments.map((comment) => ({
          author: comment.user?.login,
          author_association: comment.author_association,
          created_at: comment.created_at,
          body: clip(comment.body, MAX_COMMENT),
        })),
      );
    },
  });

  pi.registerTool({
    name: "search_issues",
    label: "Search Issues",
    description: `Search issues in ${repo} (GitHub search syntax, e.g. "flex attention backward is:open"). Returns up to 10 matches.`,
    parameters: Type.Object({ query: Type.String({ minLength: 1, maxLength: 256 }) }),
    async execute(_id, { query }, signal) {
      const result = (await ghApi(
        ["-X", "GET", "search/issues", "-f", `q=repo:${repo} is:issue ${query}`, "-f", "per_page=10"],
        signal,
      )) as any;
      return text(
        // A replayed issue must not leak its own post-triage labels.
        result.items.filter((item: any) => item.number !== replayIssue).map((item: any) => ({
          number: item.number,
          title: item.title,
          state: item.state,
          labels: item.labels.map((label: any) => label.name),
        })),
      );
    },
  });
}
