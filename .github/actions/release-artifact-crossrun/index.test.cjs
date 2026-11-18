'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const p = require('./protocol.cjs');
const c = require('./index.cjs');

function binding(kind, runId) {
  const value = {run_id: runId, attempt: 1, head: String(runId % 10).repeat(40), job_id: runId + 10, workflow_id: 300, pr: 205999, artifacts: {}};
  const kinds = kind === 'preflight' ? ['preflight'] : Object.keys(p.MEMBERS);
  for (const [index, key] of kinds.entries()) value.artifacts[key] = {id: runId * 10 + index, name: p.artifactName(value, key), digest: kind === 'preflight' ? 'e'.repeat(64) : p.digest(p.archive(key))};
  if (kind === 'preflight') Object.assign(value, {deleted_by: 'd4l3k', delete_status: 204, deleted_get_status: 404, remaining_artifacts: 0});
  return value;
}

function config(phase) {
  return {phase, expected_pr: 205999, actor_ids: {ptdbot: 11, d4l3k: 12}, rest_preflight: phase === 'preflight' ? null : binding('preflight', 101), producer: phase === 'consumer' ? binding('producer', 102) : null};
}

function earlier(value) {
  const repo = {id: p.REPOSITORY_ID, full_name: p.REPOSITORY};
  const headRepo = {id: p.HEAD_REPOSITORY_ID, full_name: p.HEAD_REPOSITORY};
  const run = {id: value.run_id, run_attempt: value.attempt, workflow_id: value.workflow_id, path: p.WORKFLOW, event: 'pull_request', head_sha: value.head, head_branch: p.BRANCH, repository: repo, head_repository: headRepo, status: 'completed', conclusion: 'success', actor: {login: 'ptdbot', id: 11}, triggering_actor: {login: 'd4l3k', id: 12}, pull_requests: [{number: value.pr, head: {sha: value.head, ref: p.BRANCH, repo: headRepo}, base: {ref: 'main', repo}}]};
  const job = {id: value.job_id, run_id: value.run_id, run_attempt: value.attempt, head_sha: value.head, workflow_name: c.WORKFLOW_NAME, name: c.JOB, status: 'completed', conclusion: 'success'};
  const artifacts = Object.values(value.artifacts).map(item => ({...item, digest: `sha256:${item.digest}`, expired: false, size_in_bytes: 200, workflow_run: {id: value.run_id, repository_id: p.REPOSITORY_ID, head_repository_id: p.HEAD_REPOSITORY_ID, head_branch: p.BRANCH, head_sha: value.head}}));
  return {run, job, inventory: {total_count: artifacts.length, artifacts}};
}

function changed(value, route, replacement) {
  const copy = structuredClone(value);
  let target = copy;
  for (const key of route.slice(0, -1)) target = target[key];
  target[route.at(-1)] = replacement;
  return copy;
}

test('reviewed activation requires both ordinary deletion authorities and literal producer bindings', () => {
  for (const phase of ['preflight', 'producer', 'consumer']) assert.equal(c.configuration(config(phase)).phase, phase);
  for (const [route, replacement] of [
    [['phase'], 'disabled'], [['phase'], 'other'], [['expected_pr'], 0], [['actor_ids', 'ptdbot'], 0], [['actor_ids', 'd4l3k'], 11],
    [['rest_preflight'], null], [['rest_preflight', 'delete_status'], 200], [['rest_preflight', 'deleted_get_status'], 200], [['rest_preflight', 'deleted_by'], 'other'], [['rest_preflight', 'remaining_artifacts'], 1], [['rest_preflight', 'pr'], 205998], [['rest_preflight', 'artifacts', 'preflight', 'id'], 0], [['rest_preflight', 'artifacts', 'preflight', 'name'], 'unowned'], [['rest_preflight', 'artifacts', 'preflight', 'digest'], 'bad'],
    [['producer'], null], [['producer', 'run_id'], 101], [['producer', 'workflow_id'], 301], [['producer', 'artifacts', 'embedded', 'id'], 0], [['producer', 'artifacts', 'embedded', 'name'], 'other'], [['producer', 'artifacts', 'embedded', 'digest'], '0'.repeat(64)]
  ]) assert.throws(() => c.configuration(changed(config('consumer'), route, replacement)), route.join('.'));
  const duplicate = config('consumer');
  duplicate.producer.artifacts.leading.id = duplicate.producer.artifacts.ordinary.id;
  assert.throws(() => c.configuration(duplicate));
  assert.throws(() => c.configuration({...config('consumer'), arbitrary_run: 100}));
  assert.throws(() => c.configuration({...config('preflight'), producer: binding('producer', 102)}));
});

test('earlier-run authority binds exact repository, actor IDs, PR, head, workflow, attempt and completed job', () => {
  const cfg = config('consumer');
  const base = earlier(cfg.producer);
  c.validateEarlierRun(base.run, base.job, cfg.producer, cfg);
  const mutations = [
    ['run', ['id'], 103], ['run', ['run_attempt'], 2], ['run', ['workflow_id'], 301], ['run', ['path'], 'other.yml'], ['run', ['event'], 'push'], ['run', ['head_sha'], 'a'.repeat(40)], ['run', ['head_branch'], 'main'], ['run', ['repository', 'id'], 1], ['run', ['repository', 'full_name'], 'other/repo'], ['run', ['head_repository', 'id'], 1], ['run', ['head_repository', 'full_name'], 'other/repo'], ['run', ['status'], 'in_progress'], ['run', ['conclusion'], 'failure'], ['run', ['actor', 'login'], 'd4l3k'], ['run', ['actor', 'id'], 12], ['run', ['triggering_actor', 'id'], 11], ['run', ['pull_requests'], []], ['run', ['pull_requests', 0, 'number'], 205998], ['run', ['pull_requests', 0, 'head', 'sha'], 'a'.repeat(40)], ['run', ['pull_requests', 0, 'head', 'ref'], 'main'], ['run', ['pull_requests', 0, 'head', 'repo', 'id'], 1], ['run', ['pull_requests', 0, 'base', 'ref'], 'release/2.15'], ['run', ['pull_requests', 0, 'base', 'repo', 'id'], 1],
    ['job', ['id'], 113], ['job', ['run_id'], 103], ['job', ['run_attempt'], 2], ['job', ['head_sha'], 'a'.repeat(40)], ['job', ['workflow_name'], 'other'], ['job', ['name'], 'other'], ['job', ['status'], 'in_progress'], ['job', ['conclusion'], 'failure']
  ];
  for (const [group, route, replacement] of mutations) {
    const data = {...base, [group]: changed(base[group], route, replacement)};
    assert.throws(() => c.validateEarlierRun(data.run, data.job, cfg.producer, cfg), `${group}.${route.join('.')}`);
  }
});

test('public artifact metadata requires the allowlisted service ID, name, digest and workflow provenance', () => {
  const cfg = config('consumer');
  const expected = cfg.producer.artifacts.embedded;
  const item = earlier(cfg.producer).inventory.artifacts.find(value => value.id === expected.id);
  c.validateArtifact(item, expected, cfg.producer);
  for (const [route, replacement] of [
    [['id'], expected.id + 1], [['name'], 'other'], [['digest'], `sha256:${'0'.repeat(64)}`], [['expired'], true], [['size_in_bytes'], 0], [['size_in_bytes'], 65537], [['workflow_run', 'id'], 103], [['workflow_run', 'repository_id'], 1], [['workflow_run', 'head_repository_id'], 1], [['workflow_run', 'head_branch'], 'main'], [['workflow_run', 'head_sha'], 'a'.repeat(40)]
  ]) assert.throws(() => c.validateArtifact(changed(item, route, replacement), expected, cfg.producer), route.join('.'));
});

test('cross-run REST preflight uses only GET404 and an empty exact-run inventory', async () => {
  const cfg = config('producer');
  const fixture = earlier(cfg.rest_preflight);
  const calls = [];
  const request = async (endpoint, status = 200) => {
    calls.push([endpoint, status]);
    if (endpoint.includes('/attempts/')) return fixture.run;
    if (endpoint.includes('/jobs/')) return fixture.job;
    if (endpoint.includes('/artifacts?')) return {total_count: 0, artifacts: []};
    assert.equal(endpoint, `repos/${p.REPOSITORY}/actions/artifacts/${cfg.rest_preflight.artifacts.preflight.id}`);
    assert.equal(status, 404);
    return null;
  };
  await c.verifyEarlier(cfg.rest_preflight, cfg, true, request);
  assert.equal(calls.length, 4);
  await assert.rejects(c.verifyEarlier(cfg.rest_preflight, cfg, true, async (endpoint, status) => endpoint.includes('/artifacts?') ? fixture.inventory : request(endpoint, status)));
  assert(calls.every(([endpoint]) => endpoint.startsWith('repos/pytorch/pytorch/actions/')));
});

test('consumer resolves only literal artifacts and rejects substitutions, extra inventory or missing IDs', async () => {
  const cfg = config('consumer');
  const fixture = earlier(cfg.producer);
  const request = async (endpoint, status = 200) => {
    assert.equal(status, 200);
    if (endpoint.includes('/attempts/')) return fixture.run;
    if (endpoint.includes('/jobs/')) return fixture.job;
    if (endpoint.includes('/artifacts?')) return fixture.inventory;
    return fixture.inventory.artifacts.find(item => endpoint.endsWith(`/${item.id}`));
  };
  await c.verifyEarlier(cfg.producer, cfg, false, request);
  for (const mutate of [
    item => ({...item, total_count: 4}),
    item => ({total_count: 2, artifacts: item.artifacts.slice(1)}),
    item => ({...item, artifacts: item.artifacts.map((value, i) => i ? value : {...value, id: 999})})
  ]) await assert.rejects(c.verifyEarlier(cfg.producer, cfg, false, async (endpoint, status) => endpoint.includes('/artifacts?') ? mutate(fixture.inventory) : request(endpoint, status)));
});

test('the ordinary preflight never invokes custom ZIP creation and requires exact SDK job-owned listing', async () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'artifact-crossrun-rest-'));
  const previous = {RUNNER_TEMP: process.env.RUNNER_TEMP, GITHUB_OUTPUT: process.env.GITHUB_OUTPUT};
  process.env.RUNNER_TEMP = temporary;
  process.env.GITHUB_OUTPUT = path.join(temporary, 'outputs');
  const context = {repository: p.REPOSITORY, run_id: 123, attempt: 1, phase: 'preflight'};
  p.initialize(context);
  const ids = {workflowRunBackendId: '11111111-1111-1111-1111-111111111111', workflowJobRunBackendId: '22222222-2222-2222-2222-222222222222'};
  const calls = [];
  const service = {ids, StringValue: {create: x => x}, upload: async (name, files, root, options) => {
    calls.push('ordinary-upload');
    assert.equal(name, p.artifactName(context, 'preflight'));
    assert.deepEqual(fs.readFileSync(files[0]), p.MARKER);
    assert.equal(path.dirname(files[0]), root);
    assert.deepEqual(options, {retentionDays: 1, compressionLevel: 0});
    return {id: 789, digest: 'e'.repeat(64)};
  }, client: {ListArtifacts: async request => {
    calls.push('list');
    assert.deepEqual(request, {...ids, nameFilter: {value: p.artifactName(context, 'preflight')}});
    return {artifacts: [{...ids, name: p.artifactName(context, 'preflight'), databaseId: '789'}]};
  }}};
  try {
    assert.equal(c.completed(context), false);
    await c.ordinaryPreflight(context, service);
    assert.deepEqual(calls, ['ordinary-upload', 'list']);
    assert.equal(c.completed(context), true);
    assert.equal(c.ready(context), false);
    const receipt = path.join(p.fixtureRoot(context), 'ready.json');
    fs.writeFileSync(receipt, JSON.stringify(context), {flag: 'wx'});
    assert.equal(c.ready(context), true);
    fs.writeFileSync(receipt, JSON.stringify({...context, run_id: 999}));
    assert.throws(() => c.ready(context));
    assert.equal(p.readState(context).preflight_deleted, false);
  } finally {
    for (const [key, value] of Object.entries(previous)) value === undefined ? delete process.env[key] : process.env[key] = value;
    fs.rmSync(temporary, {recursive: true, force: true});
  }
});
