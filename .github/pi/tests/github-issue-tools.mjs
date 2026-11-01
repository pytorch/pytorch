import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";

// Use the same host-provided TypeBox module as the action's Pi extension loader.
const hostFile = join(process.env.PI_TEST_RUNTIME, "package.json");
const host = createRequire(hostFile);
const { createJiti } = host("jiti");
const jiti = createJiti(hostFile, { alias: { typebox: host.resolve("typebox") } });
const { validateToolArguments } = await jiti.import("@earendil-works/pi-ai/utils/validation");
process.env.TRIAGE_REPO = "pytorch/pytorch";
process.env.TRIAGE_REPLAY_ISSUE = "77";
const { default: extension } = await jiti.import(new URL("../extensions/github-issue-tools.ts", import.meta.url).pathname);
const temp = mkdtempSync(join(tmpdir(), "pi-duplicate-tools-"));
const originalPath = process.env.PATH;
const fixtureFile = join(temp, "fixture.json");
const callsFile = join(temp, "calls.jsonl");
process.env.PATH = `${temp}:${originalPath}`;
process.env.GH_FIXTURE = fixtureFile;
process.env.GH_CALLS = callsFile;
writeFileSync(join(temp, "gh"), `#!${process.execPath}
const fs = require("node:fs");
const args = process.argv.slice(2);
fs.appendFileSync(process.env.GH_CALLS, JSON.stringify(args) + "\\n");
const fixture = JSON.parse(fs.readFileSync(process.env.GH_FIXTURE, "utf8"));
let route;
if (args[0] === "api" && args[1] === "-X" && args[2] === "GET" && args[3] === "search/issues") {
  route = fixture.search[args.find(arg => arg.startsWith("q=")).slice(2)];
} else if (args[0] === "api" && args.length === 2 && /^repos\\/pytorch\\/pytorch\\/issues\\/\\d+$/.test(args[1])) {
  route = fixture.issues[args[1].split("/").at(-1)];
}
if (!route) { console.error("unexpected gh request: " + JSON.stringify(args)); process.exit(1); }
if (route.error) { console.error(route.error); process.exit(1); }
const output = route.raw ?? JSON.stringify(route);
if (route.delayMs) setTimeout(() => process.stdout.write(output), route.delayMs);
else process.stdout.write(output);
`);
chmodSync(join(temp, "gh"), 0o755);
after(() => {
  process.env.PATH = originalPath;
  delete process.env.GH_FIXTURE;
  delete process.env.GH_CALLS;
  rmSync(temp, { recursive: true, force: true });
});

const query = (...terms) => `repo:pytorch/pytorch is:issue in:title,body,comments ${terms.map(term => `"${term}"`).join(" ")}`;
const issue = (number, overrides = {}) => ({
  number,
  url: `https://api.github.com/repos/pytorch/pytorch/issues/${number}`,
  html_url: `https://github.com/pytorch/pytorch/issues/${number}`,
  title: `DTensor backward failure ${number}`,
  body: "A reproducible DTensor backward failure, with its traceback.",
  state: "open",
  state_reason: null,
  labels: [{ name: "module: dtensor" }],
  ...overrides,
});
const search = (items, overrides = {}) => ({ total_count: items.length, incomplete_results: false, items, ...overrides });

function session(fixture = { search: {}, issues: {} }, source = "42") {
  writeFileSync(fixtureFile, JSON.stringify(fixture));
  writeFileSync(callsFile, "");
  if (source === undefined || source === null) delete process.env.TRIAGE_SOURCE_ISSUE;
  else process.env.TRIAGE_SOURCE_ISSUE = source;
  const tools = new Map();
  extension({ registerTool(tool) { tools.set(tool.name, tool); } });
  const tool = tools.get("find_duplicate_issues");
  assert.ok(tool, "specialized duplicate finder must be registered");
  return {
    tools,
    async call(terms = [["DTensor", "backward"]], signal = AbortSignal.timeout(10000)) {
      const args = validateToolArguments(tool, { name: tool.name, arguments: { queries: terms.map(terms => ({ terms })) } });
      const result = await tool.execute("duplicate-call", args, signal);
      return JSON.parse(result.content[0].text);
    },
  };
}

function calls() {
  return readFileSync(callsFile, "utf8").split("\n").filter(Boolean).map(line => JSON.parse(line));
}

test("automatically broadens noisy supplied queries and recovers the recorded cosine target in one call", async () => {
  const focused = ["cosine_embedding_loss", "fp16", "overflow"];
  const family = ["cosine", "loss", "overflow"];
  const noise = [issue(67463), issue(199147), issue(110761), issue(49016), issue(182279)];
  const target = issue(69512, {
    title: "Inconsistent behavior of cosine_similarity between fp16 and fp32 inputs",
    text_matches: [{ object_url: "https://api.github.com/repositories/65600975/issues/comments/988220757", object_type: "IssueComment", property: "body", fragment: "Here, w1*w2 overflows in fp16, and subsequent division results in 0." }],
  });
  const broadened = [target, issue(11202), issue(133849), issue(199490), issue(182279)];
  const { call } = session({
    search: {
      [query(...focused)]: search(noise),
      [query(...family)]: search(noise),
      [query("cosine", "overflow")]: search(broadened, { total_count: 51 }),
    },
    issues: Object.fromEntries([...noise, ...broadened].map(row => [row.number, row])),
  }, "202948");
  const result = await call([focused, family]);
  const recovered = result.candidates.find(candidate => candidate.number === 69512);
  assert.ok(recovered, "automatic broadening must hydrate the known target despite nonempty noise");
  assert.deepEqual(result.queries.map(({ terms, origin, source_group }) => ({ terms, origin, source_group })), [
    { terms: focused, origin: "provided", source_group: 0 },
    { terms: family, origin: "provided", source_group: 1 },
    { terms: ["cosine", "overflow"], origin: "broadened", source_group: 1 },
  ]);
  assert.deepEqual(recovered.matched_queries, [2]);
  assert.equal(recovered.matching_fragments[0].source, "comment");
  assert.equal(result.calls_remaining, 1, "retrieval must not depend on a second model call");
  assert.equal(result.candidates.length, 5);
  assert.equal(calls().filter(args => args.includes("search/issues")).length, 3);
  assert.equal(calls().filter(args => !args.includes("search/issues")).length, 5);
});

test("broadens only the last group using ordered distinct literal anchors", async t => {
  const cases = [
    { name: "three terms", terms: ["cholesky", "forward-mode", "autograd"], broad: ["cholesky", "autograd"] },
    { name: "four terms", terms: ["batch_norm_stats", "CUDA", "SIGFPE", "eps"], broad: ["batch_norm_stats", "eps"] },
    { name: "two terms", terms: ["cosine", "overflow"], broad: ["cosine"] },
    { name: "trimmed repeated terms", terms: [" cosine ", "loss", "cosine", "overflow"], broad: ["cosine", "overflow"] },
    { name: "two distinct terms", terms: ["cosine", "loss", "cosine"], broad: ["cosine"] },
  ];
  for (const { name, terms, broad } of cases) {
    await t.test(name, async () => {
      const literals = terms.map(term => term.trim());
      const { call } = session({
        search: { [query("focused", "context")]: search([]), [query(...literals)]: search([]), [query(...broad)]: search([]) }, issues: {},
      });
      const result = await call([["focused", "context"], terms]);
      assert.deepEqual(result.queries.map(item => item.terms), [["focused", "context"], literals, broad]);
      assert.deepEqual(result.queries.map(item => item.source_group), [0, 1, 1]);
      assert.equal(result.status, "complete");
      assert.equal(calls().length, 3);
    });
  }
  await t.test("one provided group", async () => {
    const { call } = session({ search: { [query("cosine", "overflow")]: search([]), [query("cosine")]: search([]) }, issues: {} });
    const result = await call([["cosine", "overflow"]]);
    assert.deepEqual(result.queries.map(item => [item.origin, item.source_group]), [["provided", 0], ["broadened", 0]]);
    assert.equal(calls().length, 2);
  });
});

test("deduplicates candidates across provided and broadened searches and skips identical planned queries", async t => {
  const row = issue(69512, {
    text_matches: [{ object_url: "https://api.github.com/repositories/65600975/issues/comments/988220757", object_type: "IssueComment", property: "body", fragment: "w1*w2 overflows in fp16" }],
  });
  const { call } = session({
    search: { [query("cosine", "fp16", "overflow")]: search([row]), [query("cosine", "loss", "overflow")]: search([row]), [query("cosine", "overflow")]: search([row]) },
    issues: { 69512: issue(69512) },
  });
  const result = await call([["cosine", "fp16", "overflow"], ["cosine", "loss", "overflow"]]);
  assert.deepEqual(result.candidates.map(candidate => candidate.number), [69512]);
  assert.deepEqual(result.candidates[0].matched_queries, [0, 1, 2]);
  assert.equal(result.candidates[0].matching_fragments.length, 1);
  assert.equal(calls().length, 4);
  for (const groups of [[["cosine"]], [["cosine", "cosine"]], [["cosine", "overflow"], ["cosine", "loss", "overflow"]]]) {
    await t.test(JSON.stringify(groups), async () => {
      const { call } = session({ search: Object.fromEntries(groups.map(terms => [query(...terms), search([])])), issues: {} });
      const result = await call(groups);
      assert.equal(result.queries.length, groups.length);
      assert.ok(result.queries.every(item => item.origin === "provided"));
      assert.equal(calls().length, groups.length);
    });
  }
});

test("rejects a third provided query group through the registered schema without consuming budget", async () => {
  const { call } = session({ search: { [query("a")]: search([]), [query("b")]: search([]) }, issues: {} });
  await assert.rejects(call([["a"], ["b"], ["c"]]), /Validation/);
  assert.deepEqual(calls(), []);
  const result = await call([["a"], ["b"]]);
  assert.equal(result.calls_remaining, 1);
  assert.equal(calls().length, 2);
});

test("searches title, body and comments with GitHub text-match media type", async () => {
  const fixedQuery = 'repo:pytorch/pytorch is:issue in:title,body,comments "cosine" "fp16" "overflow"';
  const { call } = session({
    search: { [fixedQuery]: search([issue(69512)]), [query("cosine", "overflow")]: search([]) },
    issues: { 69512: issue(69512) },
  });
  const result = await call([["cosine", "fp16", "overflow"]]);
  assert.deepEqual({
    fixed_fields: calls()[0].includes(`q=${fixedQuery}`),
    text_match_media_type: calls()[0].includes("Accept: application/vnd.github.text-match+json"),
  }, { fixed_fields: true, text_match_media_type: true });
  assert.equal(result.status, "complete");
  assert.deepEqual(result.candidates.map(candidate => candidate.number), [69512]);
  assert.equal(calls().length, 3);
});

test("returns measured numeric-repository comment and issue fragment provenance", async () => {
  const fragment = "Here, w1*w2 overflows in fp16, and subsequent division results in 0.";
  const matches = [
    { object_url: "https://api.github.com/repositories/65600975/issues/69512", object_type: "Issue", property: "title", fragment: "Inconsistent behavior of cosine similarity" },
    { object_url: "https://api.github.com/repositories/65600975/issues/69512", object_type: "Issue", property: "body", fragment: "cosine similarity outputs 0 in fp16" },
    { object_url: "https://api.github.com/repositories/65600975/issues/comments/988220757", object_type: "IssueComment", property: "body", fragment, author_association: "MEMBER", user: { login: "unverified" } },
  ];
  const { call } = session({
    search: { [query("DTensor", "backward")]: search([issue(69512, { text_matches: matches })]), [query("DTensor")]: search([]) },
    issues: { 69512: issue(69512) },
  });
  const result = await call();
  assert.deepEqual(result.candidates[0].matching_fragments, [
    { source: "comment", field: "body", url: "https://github.com/pytorch/pytorch/issues/69512#issuecomment-988220757", fragment, fragment_truncated: false },
    { source: "issue", field: "title", url: "https://github.com/pytorch/pytorch/issues/69512", fragment: matches[0].fragment, fragment_truncated: false },
    { source: "issue", field: "body", url: "https://github.com/pytorch/pytorch/issues/69512", fragment: matches[1].fragment, fragment_truncated: false },
  ]);
  assert.equal(result.candidates[0].matching_fragments_truncated, false);
  assert.equal(result.status, "complete");
  assert.equal(calls().length, 3);
});

test("bounds and deduplicates fragments across queries while preferring comment diagnostics", async () => {
  const numericComment = { object_url: "https://api.github.com/repositories/65600975/issues/comments/123", object_type: "IssueComment", property: "body", fragment: "diagnosis ".repeat(100) };
  const textualComment = { ...numericComment, object_url: "https://api.github.com/repos/pytorch/pytorch/issues/comments/123" };
  const issueMatches = number => ["title", "body"].map(property => ({ object_url: `https://api.github.com/repos/pytorch/pytorch/issues/${number}`, object_type: "Issue", property, fragment: `matched ${property}` }));
  const rows = Array.from({ length: 6 }, (_, index) => issue(index + 1, { text_matches: issueMatches(index + 1) }));
  const otherComment = { ...numericComment, object_url: "https://api.github.com/repositories/65600975/issues/comments/124", fragment: "second diagnostic" };
  const clippedComment = { ...numericComment, fragment: numericComment.fragment.slice(0, 400) };
  const second = rows.map(row => ({
    ...row, text_matches: [textualComment, otherComment, ...issueMatches(row.number), numericComment, clippedComment],
  }));
  const { call } = session({
    search: { [query("first")]: search(rows), [query("second")]: search(second) },
    issues: Object.fromEntries(rows.map(row => [row.number, row])),
  });
  const result = await call([["first"], ["second"]]);
  const hydrated = result.candidates[0];
  assert.equal(hydrated.matching_fragments.length, 3);
  assert.deepEqual(hydrated.matching_fragments.map(match => match.source), ["comment", "comment", "issue"]);
  assert.equal(hydrated.matching_fragments[0].fragment, numericComment.fragment.slice(0, 400));
  assert.equal(hydrated.matching_fragments[0].fragment_truncated, true);
  assert.equal(hydrated.matching_fragments[1].fragment_truncated, false);
  assert.equal(hydrated.matching_fragments_truncated, true);
  assert.deepEqual(hydrated.matched_queries, [0, 1]);
  const unhydrated = result.unhydrated_candidates[0];
  assert.equal(unhydrated.number, 6);
  assert.deepEqual(unhydrated.matching_fragments, [{ ...hydrated.matching_fragments[0], url: "https://github.com/pytorch/pytorch/issues/6#issuecomment-123" }]);
  assert.equal(unhydrated.matching_fragments_truncated, true);
  assert.ok([...result.candidates, ...result.unhydrated_candidates].every(candidate => candidate.matching_fragments.every(match => match.fragment.length <= 400)));
  assert.equal(calls().length, 7);
});

test("ignores unknown or malformed text-match metadata without changing coverage status", async () => {
  const valid = { object_url: "https://api.github.com/repos/pytorch/pytorch/issues/comments/123", object_type: "IssueComment", property: "body", fragment: "comment diagnostic" };
  const invalid = [
    null, {}, { ...valid, object_type: "PullRequest" }, { ...valid, property: "title" },
    { ...valid, fragment: 123 }, { ...valid, fragment: "" }, { ...valid, object_url: null },
    ...[
      "https://api.github.com/repos/other/repo/issues/comments/123",
      "https://api.github.com.evil.test/repos/pytorch/pytorch/issues/comments/123",
      "https://evil.test/repos/pytorch/pytorch/issues/comments/123",
      "http://api.github.com/repos/pytorch/pytorch/issues/comments/123",
      "https://api.github.com:443/repos/pytorch/pytorch/issues/comments/123",
      "https://user@api.github.com/repos/pytorch/pytorch/issues/comments/123",
      "https://api.github.com/repos/pytorch/pytorch/issues/comments/123?redirect=evil",
      "https://api.github.com/repos/pytorch/pytorch/issues/comments/123#fragment",
      "https://api.github.com/repos/pytorch/pytorch/issues/comments/0",
      "https://api.github.com/repos/pytorch/pytorch/issues/comments/9007199254740992",
      "https://api.github.com/repos/pytorch/pytorch/issues/comments/123\n",
      "https://api.github.com/repositories/0/issues/comments/123",
      "https://api.github.com/repositories/9007199254740992/issues/comments/123",
      "https://api.github.com/repos/pytorch/pytorch/issues/10/../comments/123",
    ].map(object_url => ({ ...valid, object_url })),
    { ...valid, object_type: "Issue", object_url: "https://api.github.com/repositories/65600975/issues/11" },
    { ...valid, object_type: "Issue", object_url: "https://api.github.com/repos/pytorch/pytorch/issues/11" },
    { ...valid, object_type: "Issue", object_url: "https://api.github.com/repos/pytorch/pytorch/issues/comments/123" },
    { ...valid, object_url: "https://api.github.com/repositories/65600975/issues/10" },
  ];
  const rows = [issue(10, { text_matches: [...invalid, valid] }), issue(11), issue(12, { text_matches: null }), issue(13, { text_matches: {} })];
  const { call } = session({
    search: { [query("DTensor", "backward")]: search(rows), [query("DTensor")]: search([]) },
    issues: Object.fromEntries(rows.map(row => [row.number, row])),
  });
  const result = await call();
  assert.equal(result.status, "complete");
  assert.deepEqual(result.partial_failures, []);
  assert.deepEqual(result.candidates[0].matching_fragments, [{ source: "comment", field: "body", url: "https://github.com/pytorch/pytorch/issues/10#issuecomment-123", fragment: valid.fragment, fragment_truncated: false }]);
  assert.ok(result.candidates.slice(1).every(candidate => candidate.matching_fragments.length === 0));
  assert.ok(result.candidates.every(candidate => candidate.matching_fragments_truncated === false));
  assert.equal(calls().length, 6);
});

test("returns hydrated open and closed evidence and deduplicates query matches", async () => {
  const closed = issue(11, { state: "closed", state_reason: "completed", body: "Fixed in a later version." });
  const { call } = session({
    search: {
      [query("DTensor", "backward")]: search([issue(10), issue(11), issue(10)]),
      [query("placement")]: search([closed]),
    },
    issues: { 10: issue(10), 11: closed },
  });
  const result = await call([["DTensor", "backward"], ["placement"]]);
  assert.equal(result.source_issue, 42);
  assert.equal(result.repository, "pytorch/pytorch");
  assert.equal(result.status, "complete");
  assert.deepEqual(result.partial_failures, []);
  assert.deepEqual(result.candidates.map(candidate => candidate.number), [10, 11]);
  assert.deepEqual(result.candidates[1].matched_queries, [0, 1]);
  assert.equal(result.candidates[1].state, "closed");
  assert.equal(result.candidates[1].state_reason, "completed");
  assert.equal(result.candidates[1].body, closed.body);
  assert.equal(result.candidates[1].url, closed.html_url);
  assert.deepEqual(result.candidates[0].labels, ["module: dtensor"]);
  assert.equal(result.candidates[0].body_truncated, false);
  assert.equal(result.candidates[0].state_reason, null);
  assert.equal(calls().length, 4);
  assert.ok(calls().slice(0, 2).every(args => args.includes("per_page=10")));
});

test("excludes source, PRs, and foreign identities before and after hydration", async () => {
  const { call } = session({
    search: { [query("DTensor", "backward")]: search([
      issue(42), issue(77), issue(1, { pull_request: {} }),
      issue(2, { url: "https://api.github.com/repos/other/repo/issues/2" }),
      issue(3, { repository_url: "https://api.github.com/repos/other/repo" }),
      issue(4), issue(5), issue(6), issue(7),
    ]), [query("DTensor")]: search([]) },
    issues: {
      4: issue(4, { pull_request: {} }),
      5: issue(5, { number: 42 }),
      6: issue(6, { html_url: "https://github.com/other/repo/issues/6" }),
      7: issue(7),
    },
  });
  const result = await call();
  assert.deepEqual(result.candidates.map(candidate => candidate.number), [7]);
  assert.equal(result.status, "partial");
  assert.equal(result.partial_failures.length, 3);
  assert.equal(calls().length, 6);
});

test("enforces search, hydration, body, label and parallel session caps", async () => {
  const items = Array.from({ length: 12 }, (_, index) => issue(index + 1));
  const large = issue(1, { body: "x".repeat(5000), title: "t".repeat(500), labels: Array.from({ length: 25 }, () => ({ name: "l".repeat(200) })) });
  const { call } = session({
    search: { [query("a")]: search(items, { total_count: 40 }), [query("b", "c")]: search(items, { total_count: 40 }), [query("b")]: search(items, { total_count: 40 }) },
    issues: { 1: large, 2: issue(2), 3: issue(3), 4: issue(4), 5: issue(5) },
  });
  const results = await Promise.allSettled([call([["a"], ["b", "c"]]), call([["a"], ["b", "c"]]), call([["a"]])]);
  assert.equal(results.filter(result => result.status === "fulfilled").length, 2);
  assert.match(results.find(result => result.status === "rejected").reason.message, /budget/);
  for (const { value } of results.filter(result => result.status === "fulfilled")) {
    assert.equal(value.status, "partial");
    assert.equal(value.candidates.length, 5);
    assert.equal(value.candidates[0].body_length, 5000);
    assert.equal(value.candidates[0].body.length, 3000);
    assert.equal(value.candidates[0].body_truncated, true);
    assert.equal(value.candidates[0].title_truncated, true);
    assert.equal(value.candidates[0].labels_truncated, true);
    assert.equal(value.candidates[0].labels.length, 20);
    assert.ok(value.candidates[0].labels.every(label => label.length <= 100));
    assert.equal(value.hydration_truncated, true);
    assert.deepEqual(value.unhydrated_candidates.map(candidate => candidate.number), [6, 7, 8, 9, 10]);
    assert.deepEqual(value.unhydrated_candidates[0].matched_queries, [0, 1, 2]);
    assert.ok(value.queries.every(result => result.returned_count === 10 && result.truncated));
    assert.ok(value.candidates.every(candidate => candidate.number <= 10));
  }
  assert.equal(calls().filter(args => args.includes("search/issues")).length, 6);
  assert.equal(calls().filter(args => !args.includes("search/issues")).length, 10);
});

test("interleaves query ranks and retains titles for unhydrated candidates", async () => {
  const first = Array.from({ length: 7 }, (_, index) => issue(index + 1));
  const second = Array.from({ length: 7 }, (_, index) => issue(index + 8));
  second[2].title = "t".repeat(500);
  const { call } = session({
    search: { [query("broad")]: search(first), [query("specific")]: search(second) },
    issues: Object.fromEntries([...first, ...second].map(item => [item.number, item])),
  });
  const result = await call([["broad"], ["specific"]]);
  assert.deepEqual(result.candidates.map(candidate => candidate.number), [1, 8, 2, 9, 3]);
  assert.equal(result.unhydrated_candidates[0].number, 10);
  assert.equal(result.unhydrated_candidates[0].title, "t".repeat(256));
  assert.equal(result.unhydrated_candidates[0].title_truncated, true);
  assert.deepEqual(result.unhydrated_candidates[0].matched_queries, [1]);
  assert.equal(calls().length, 7);
});

test("rejects search syntax and invalid term groups without spending network budget", async () => {
  const validTerms = ["torch.compile", "_index_put/C++", "fp16 forward"];
  const { call, tools } = session({
    search: { [query("DTensor", "backward")]: search([]), [query("DTensor")]: search([]), [query(...validTerms)]: search([]), [query(validTerms[0], validTerms[2])]: search([]) }, issues: {},
  });
  for (const term of ["repo:other/repo", 'backward" OR "foo', "OR", "a OR b", "AND", "NOT", "a\nb", "a\tb", "a\\b", "(foo)", "*", "-repo", "bad\u007f", "bad\u202e", ""]) {
    await assert.rejects(call([[term]]), /literal|Validation/);
  }
  for (const groups of [[], [[], ["a"]], [["a", "b", "c", "d", "e"]], [["a"], ["b"], ["c"], ["d"]], [["a".repeat(81)]]]) {
    await assert.rejects(call(groups), /Validation|queries|terms/);
  }
  await assert.rejects(call([["a".repeat(80), "b".repeat(80), "c".repeat(80), "d".repeat(80)]]), /query exceeds/);
  const tool = tools.get("find_duplicate_issues");
  assert.throws(() => validateToolArguments(tool, { name: tool.name, arguments: { queries: [{ terms: ["a"] }], repository: "other/repo", source_issue: 10 } }), /Validation/);
  assert.deepEqual(calls(), []);
  assert.equal((await call()).status, "complete");
  assert.equal(calls().length, 2);
  assert.equal((await call([validTerms])).status, "complete");
  assert.ok(calls()[2].includes(`q=${query(...validTerms)}`));
  assert.equal(calls().length, 4);
});

test("preserves partial failures, incomplete results, malformed responses, and genuine empty results", async () => {
  const { call } = session({
    search: {
      [query("bad")]: { error: "GitHub rate limit exceeded" },
      [query("good")]: search([issue(8), issue(9)], { incomplete_results: true }),
      [query("good", "malformed")]: { raw: "not json" },
    },
    issues: { 8: { error: "not found" }, 9: issue(9) },
  });
  const result = await call([["bad"], ["good", "malformed"]]);
  assert.equal(result.status, "partial");
  assert.equal(result.partial_failures.length, 3);
  assert.deepEqual(result.candidates.map(candidate => candidate.number), [9]);
  assert.equal(result.queries[2].incomplete_results, true);
  assert.equal(result.queries[0].status, "error");
  const empty = session({ search: { [query("DTensor", "backward")]: search([]), [query("DTensor")]: search([]) }, issues: {} });
  const emptyResult = await empty.call();
  assert.equal(emptyResult.status, "complete");
  assert.deepEqual(emptyResult.candidates, []);
  assert.deepEqual(emptyResult.partial_failures, []);
});

test("propagates caller cancellation as partial failure without launching remaining requests", async () => {
  const { call } = session({
    search: { [query("slow")]: search([issue(10)], { delayMs: 2000 }) }, issues: {},
  });
  const result = await call([["slow"], ["later", "context"]], AbortSignal.timeout(300));
  assert.equal(result.status, "partial");
  assert.deepEqual(result.candidates, []);
  assert.equal(result.partial_failures.length, 3);
  assert.deepEqual(result.queries.map(query => query.origin), ["provided", "provided", "broadened"]);
  assert.ok(result.queries.every(query => query.status === "error"));
  assert.equal(calls().length, 1);
});

test("search errors consume the call budget and missing source binding does not break legacy reads", async () => {
  const { call } = session({ search: { [query("DTensor", "backward")]: { error: "search unavailable" }, [query("DTensor")]: { error: "search unavailable" } }, issues: {} });
  assert.equal((await call()).status, "partial");
  assert.equal((await call()).status, "partial");
  await assert.rejects(call(), /budget/);
  assert.equal(calls().length, 4);
  for (const source of [null, "0", "-1", "1.5", "42junk", "9007199254740992"]) {
    const invalid = session({ search: {}, issues: { 10: issue(10) } }, source);
    await assert.rejects(invalid.call(), /TRIAGE_SOURCE_ISSUE/);
    const read = await invalid.tools.get("get_issue").execute("read", { issue_number: 10 });
    assert.equal(JSON.parse(read.content[0].text).number, 10);
  }
});
