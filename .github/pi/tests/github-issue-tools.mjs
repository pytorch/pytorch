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
const temp = mkdtempSync(join(tmpdir(), "pi-issue-tools-"));
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
} else if (args[0] === "api" && args.length === 2) {
  route = fixture.reads[args[1]];
}
if (!route) { console.error("unexpected gh request: " + JSON.stringify(args)); process.exit(1); }
if (route.error) { console.error(route.error); process.exit(1); }
process.stdout.write(JSON.stringify(route));
`);
chmodSync(join(temp, "gh"), 0o755);
after(() => {
  process.env.PATH = originalPath;
  delete process.env.GH_FIXTURE;
  delete process.env.GH_CALLS;
  rmSync(temp, { recursive: true, force: true });
});

const query = terms => `repo:pytorch/pytorch is:issue ${terms}`;
const issue = (number, overrides = {}) => ({
  number,
  url: `https://api.github.com/repos/pytorch/pytorch/issues/${number}`,
  html_url: `https://github.com/pytorch/pytorch/issues/${number}`,
  title: `DTensor backward failure ${number}`,
  body: "A reproducible DTensor backward failure, with its traceback.",
  state: "open",
  labels: [{ name: "module: dtensor" }],
  ...overrides,
});

function session(fixture, source = "42") {
  writeFileSync(fixtureFile, JSON.stringify(fixture));
  writeFileSync(callsFile, "");
  if (source === null) delete process.env.TRIAGE_SOURCE_ISSUE;
  else process.env.TRIAGE_SOURCE_ISSUE = source;
  const tools = new Map();
  extension({ registerTool(tool) { tools.set(tool.name, tool); } });
  return async (name, args) => {
    const tool = tools.get(name);
    assert.ok(tool, `${name} must be registered`);
    const validated = validateToolArguments(tool, { name, arguments: args });
    const result = await tool.execute("test-call", validated, AbortSignal.timeout(10000));
    return JSON.parse(result.content[0].text);
  };
}

function calls() {
  return readFileSync(callsFile, "utf8").split("\n").filter(Boolean).map(line => JSON.parse(line));
}

test("supports general GitHub queries followed by candidate body and comment reads", async () => {
  const terms = '"cosine_similarity" in:title,body,comments label:"module: nn"';
  const candidate = issue(69512, { state: "closed", body: "The intermediate product overflows in fp16." });
  const call = session({
    search: { [query(terms)]: { items: [candidate] } },
    reads: {
      "repos/pytorch/pytorch/issues/69512": candidate,
      "repos/pytorch/pytorch/issues/69512/comments?per_page=50": [{ user: { login: "maintainer" }, body: "The final result is representable." }],
    },
  });
  assert.deepEqual(await call("search_issues", { query: terms }), [
    { number: 69512, title: candidate.title, state: "closed", labels: ["module: dtensor"] },
  ]);
  assert.equal((await call("get_issue", { issue_number: 69512 })).body, candidate.body);
  assert.equal((await call("get_issue_comments", { issue_number: 69512 }))[0].body, "The final result is representable.");
  assert.deepEqual(calls(), [
    ["api", "-X", "GET", "search/issues", "-f", `q=${query(terms)}`, "-f", "per_page=10"],
    ["api", "repos/pytorch/pytorch/issues/69512"],
    ["api", "repos/pytorch/pytorch/issues/69512/comments?per_page=50"],
  ]);
});

test("filters source, replay, PR and foreign identities independently of the query", async () => {
  const terms = "DTensor";
  const call = session({
    search: { [query(terms)]: { items: [
      issue(42), issue(77), issue(1, { pull_request: {} }),
      issue(2, { url: "https://api.github.com/repos/other/repo/issues/2" }), // @lint-ignore
      issue(3, { repository_url: "https://api.github.com/repos/other/repo" }), // @lint-ignore
      issue(4, { html_url: "https://github.com/other/repo/issues/4" }), // @lint-ignore
      issue(5, { url: "https://api.github.com/repos/pytorch/pytorch/issues/6" }), // @lint-ignore
      issue(-1), issue("10"),
      issue(7, { url: "https://api.github.com/repos/PyTorch/PyTorch/issues/7" }), // @lint-ignore
    ] } }, reads: {},
  });
  assert.deepEqual((await call("search_issues", { query: terms })).map(row => row.number), [7]);
  assert.equal(calls().length, 1);
});

test("rejects scope overrides and unquoted OR before calling GitHub", async () => {
  const unsafe = [
    "repo:other/repo", "-repo:pytorch/pytorch", 'repo:"other/repo"',
    "org:other", "user:other", "DTensor OR CUDA", "DTensor or CUDA",
    '(DTensor OR CUDA)', '"unfinished', "DTensor\nCUDA", "DTensor\u202eCUDA",
  ];
  const quoted = '"expected x OR y" label:"module: nn"';
  const call = session({
    search: Object.fromEntries([...unsafe, quoted].map(terms => [query(terms), { items: [] }])),
    reads: {},
  });
  for (const terms of unsafe) {
    await assert.rejects(call("search_issues", { query: terms }), /workflow repository/);
  }
  assert.deepEqual(calls(), [], "scope-changing queries must not reach GitHub");
  assert.deepEqual(await call("search_issues", { query: quoted }), []);
  assert.equal(calls().length, 1);
});

test("search works without source binding and distinguishes API failures from empty results", async () => {
  const call = session({
    search: {
      [query("empty")]: { items: [] },
      [query("error")]: { error: "GitHub rate limit exceeded" },
      [query("DTensor")]: { items: [issue(10), issue(77)] },
    }, reads: {},
  }, null);
  assert.deepEqual(await call("search_issues", { query: "empty" }), []);
  await assert.rejects(call("search_issues", { query: "error" }), /GitHub rate limit exceeded/);
  assert.deepEqual((await call("search_issues", { query: "DTensor" })).map(row => row.number), [10]);
  assert.equal(calls().length, 3);
});
