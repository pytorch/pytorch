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
 *   TRIAGE_SOURCE_ISSUE       workflow-bound source excluded from duplicate discovery
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
const MAX_DUPLICATE_BODY = 3_000;
const MAX_DUPLICATE_CALLS = 2;
const MAX_DUPLICATE_CANDIDATES = 5;
const MAX_SEARCH_RESULTS = 10;
const MAX_QUERY = 256;
const MAX_MATCH_FRAGMENTS = 3;
const MAX_MATCH_FRAGMENT = 400;

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

interface MatchingFragment {
  source: "issue" | "comment";
  field: "title" | "body";
  url: string;
  fragment: string;
  fragment_truncated: boolean;
}

interface DuplicateMatch {
  rank: number;
  title: string;
  queries: number[];
  fragments: MatchingFragment[];
  fragments_truncated: boolean;
}

function matchingFragment(metadata: any, issueNumber: number): MatchingFragment | undefined {
  const comment = metadata?.object_type === "IssueComment" && metadata.property === "body";
  if (!comment && !(metadata?.object_type === "Issue" && ["title", "body"].includes(metadata.property))) return;
  if (typeof metadata.fragment !== "string" || !metadata.fragment.trim() || typeof metadata.object_url !== "string") return;
  const identity = metadata.object_url.match(/^https:\/\/api\.github\.com\/(?:repos\/([\w.-]+\/[\w.-]+)|repositories\/([1-9]\d*))\/issues\/(comments\/)?([1-9]\d*)$/i);
  if (!identity || identity[0] !== metadata.object_url
    || (identity[1] && identity[1].toLowerCase() !== repo.toLowerCase())
    || (identity[2] && !Number.isSafeInteger(Number(identity[2])))
    || Boolean(identity[3]) !== comment || !Number.isSafeInteger(Number(identity[4]))
    || (!comment && Number(identity[4]) !== issueNumber)) return;
  // Numeric repository IDs rely on the enclosing, repository-validated search row.
  const url = `https://github.com/${repo}/issues/${issueNumber}`;
  return {
    source: comment ? "comment" : "issue", field: metadata.property,
    url: comment ? `${url}#issuecomment-${identity[4]}` : url,
    fragment: metadata.fragment.slice(0, MAX_MATCH_FRAGMENT),
    fragment_truncated: metadata.fragment.length > MAX_MATCH_FRAGMENT,
  };
}

function duplicateSearch(terms: string[], origin: "provided" | "broadened", source_group: number) {
  const literals = terms.map((term) => {
    const literal = term.trim();
    if (!/^[A-Za-z0-9_][A-Za-z0-9_ ./+-]*$/.test(literal) || /[\p{Cc}\p{Cf}]/u.test(term) || /\b(?:AND|OR|NOT)\b/.test(term)) {
      throw new Error("find_duplicate_issues: terms must be literal text without search syntax or controls");
    }
    return literal;
  });
  const query = `repo:${repo} is:issue in:title,body,comments ${literals.map((term) => `"${term}"`).join(" ")}`;
  if (query.length > MAX_QUERY) throw new Error(`find_duplicate_issues: query exceeds ${MAX_QUERY} characters`);
  return { terms: literals, query, origin, source_group };
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
  const sourceValue = process.env.TRIAGE_SOURCE_ISSUE ?? "";
  const sourceIssue = Number(sourceValue);
  let duplicateCalls = 0;

  pi.registerTool({
    name: "find_duplicate_issues",
    label: "Find Duplicate Issues",
    description: `Find possible duplicates in ${repo} using 1-2 ordered groups of 1-4 literal terms: first focused, last broader family. Put the subject anchor first and essential context last. Automatically relaxes the last group, even with nonempty results, keeping its first/last distinct terms if at least 3, or first term if 2; skips singleton or identical planned queries. At most 3 searches and 5 hydrated issues per call, including closed issues, with provided/broadened query provenance and partial-result flags; not guaranteed recall or a duplicate verdict. Limited to 2 calls per session.`,
    parameters: Type.Object({
      queries: Type.Array(Type.Object({
        terms: Type.Array(Type.String({ minLength: 1, maxLength: 80 }), { minItems: 1, maxItems: 4 }),
      }, { additionalProperties: false }), {
        minItems: 1, maxItems: 2,
        description: "Ordered literal-term groups: first focused, last broader family. Each group's first term is the subject anchor; put essential context last for automatic relaxation.",
      }),
    }, { additionalProperties: false }),
    async execute(_id, { queries }, signal) {
      if (!/^[1-9]\d*$/.test(sourceValue) || !Number.isSafeInteger(sourceIssue)) {
        throw new Error("find_duplicate_issues: TRIAGE_SOURCE_ISSUE must be a positive safe integer set by the workflow");
      }
      const searches = queries.map(({ terms }, index) => duplicateSearch(terms, "provided", index));
      const last = searches[searches.length - 1];
      const distinct = [...new Set(last.terms)];
      if (distinct.length > 1) {
        const terms = distinct.length > 2 ? [distinct[0], distinct[distinct.length - 1]] : [distinct[0]];
        const broadened = duplicateSearch(terms, "broadened", last.source_group);
        if (!searches.some((search) => search.query === broadened.query)) searches.push(broadened);
      }
      if (duplicateCalls >= MAX_DUPLICATE_CALLS) throw new Error("find_duplicate_issues: session call budget exhausted");
      // Reserve before awaiting so simultaneous calls cannot exceed the session budget.
      duplicateCalls++;
      const requestSignal = signal ? AbortSignal.any([signal, AbortSignal.timeout(90_000)]) : AbortSignal.timeout(90_000);
      const matches = new Map<number, DuplicateMatch>();
      const queryResults: Record<string, unknown>[] = [];
      const failures: Record<string, unknown>[] = [];
      const candidates: Record<string, unknown>[] = [];
      const repositoryUrl = `https://api.github.com/repos/${repo}`;
      const canonicalUrl = (number: number) => `https://github.com/${repo}/issues/${number}`;
      const isCandidate = (item: any) => {
        const number = item?.number;
        return Number.isSafeInteger(number) && number > 0 && number !== sourceIssue && number !== replayIssue
          && !item.pull_request && typeof item.url === "string"
          && item.url.toLowerCase() === `${repositoryUrl}/issues/${number}`.toLowerCase()
          && (item.repository_url === undefined || (typeof item.repository_url === "string" && item.repository_url.toLowerCase() === repositoryUrl.toLowerCase()))
          && (item.html_url === undefined || (typeof item.html_url === "string" && item.html_url.toLowerCase() === canonicalUrl(number).toLowerCase()));
      };
      for (const [index, search] of searches.entries()) {
        try {
          requestSignal.throwIfAborted();
          const result = (await ghApi(
            ["-X", "GET", "search/issues", "-H", "Accept: application/vnd.github.text-match+json", "-f", `q=${search.query}`, "-f", `per_page=${MAX_SEARCH_RESULTS}`],
            requestSignal,
          )) as any;
          if (!Array.isArray(result?.items) || !Number.isSafeInteger(result.total_count) || result.total_count < 0
            || typeof result.incomplete_results !== "boolean") {
            throw new Error("Invalid GitHub search response");
          }
          const items = result.items.slice(0, MAX_SEARCH_RESULTS);
          let excluded = 0;
          for (const [rank, item] of items.entries()) {
            if (!isCandidate(item) || typeof item.title !== "string") { excluded++; continue; }
            const match: DuplicateMatch = matches.get(item.number) ?? { rank, title: item.title, queries: [], fragments: [], fragments_truncated: false };
            match.rank = Math.min(match.rank, rank);
            if (!match.queries.includes(index)) match.queries.push(index);
            for (const metadata of Array.isArray(item.text_matches) ? item.text_matches : []) {
              const fragment = matchingFragment(metadata, item.number);
              if (!fragment) continue;
              const saved = match.fragments.find((saved) => saved.source === fragment.source && saved.field === fragment.field
                && saved.url === fragment.url && saved.fragment === fragment.fragment);
              if (saved) { saved.fragment_truncated ||= fragment.fragment_truncated; continue; }
              match.fragments.push(fragment);
              match.fragments.sort((a, b) => Number(b.source === "comment") - Number(a.source === "comment"));
              if (match.fragments.length > MAX_MATCH_FRAGMENTS) { match.fragments.pop(); match.fragments_truncated = true; }
            }
            matches.set(item.number, match);
          }
          queryResults.push({
            ...search, status: "ok", total_count: result.total_count, returned_count: items.length,
            excluded_count: excluded, incomplete_results: result.incomplete_results,
            truncated: result.total_count > items.length || result.items.length > MAX_SEARCH_RESULTS,
          });
        } catch (error) {
          failures.push({ stage: "search", query_index: index, error: String(error).slice(0, 1_000) });
          queryResults.push({ ...search, status: "error" });
        }
      }
      const ranked = [...matches.entries()].sort((a, b) => a[1].rank - b[1].rank);
      const selected = ranked.slice(0, MAX_DUPLICATE_CANDIDATES);
      for (const [number, match] of selected) {
        const matchedQueries = match.queries;
        try {
          requestSignal.throwIfAborted();
          const issue = (await ghApi([`repos/${repo}/issues/${number}`], requestSignal)) as any;
          if (!isCandidate(issue) || issue.number !== number || typeof issue.title !== "string"
            || (issue.body !== null && typeof issue.body !== "string") || !["open", "closed"].includes(issue.state)
            || (issue.state_reason !== null && (typeof issue.state_reason !== "string" || issue.state_reason.length > 100))
            || !Array.isArray(issue.labels)
            || issue.labels.some((label: any) => typeof label?.name !== "string")) {
            throw new Error("Invalid or out-of-scope GitHub issue response");
          }
          const body = issue.body ?? "";
          candidates.push({
            number, url: canonicalUrl(number), title: issue.title.slice(0, 256), title_truncated: issue.title.length > 256,
            body: body.slice(0, MAX_DUPLICATE_BODY), body_length: body.length, body_truncated: body.length > MAX_DUPLICATE_BODY,
            state: issue.state, state_reason: issue.state_reason,
            labels: issue.labels.slice(0, 20).map((label: any) => label.name.slice(0, 100)),
            labels_truncated: issue.labels.length > 20 || issue.labels.some((label: any) => label.name.length > 100),
            matched_queries: matchedQueries,
            matching_fragments: match.fragments, matching_fragments_truncated: match.fragments_truncated,
          });
        } catch (error) {
          failures.push({ stage: "hydration", issue_number: number, matched_queries: matchedQueries, error: String(error).slice(0, 1_000) });
        }
      }
      return text({
        repository: repo, source_issue: sourceIssue,
        status: failures.length || matches.size > MAX_DUPLICATE_CANDIDATES
          || queryResults.some((query) => query.incomplete_results || query.truncated) ? "partial" : "complete",
        queries: queryResults, candidates, partial_failures: failures,
        hydration_truncated: matches.size > MAX_DUPLICATE_CANDIDATES,
        unhydrated_candidates: ranked.slice(MAX_DUPLICATE_CANDIDATES).map(([number, match]) => ({
          number, url: canonicalUrl(number), title: match.title.slice(0, 256),
          title_truncated: match.title.length > 256, matched_queries: match.queries,
          matching_fragments: match.fragments.slice(0, 1),
          matching_fragments_truncated: match.fragments_truncated || match.fragments.length > 1,
        })),
        calls_remaining: MAX_DUPLICATE_CALLS - duplicateCalls,
      });
    },
  });

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
