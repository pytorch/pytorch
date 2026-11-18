'use strict';

const assert = require('node:assert/strict');
const {execFileSync, spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const p = require('./protocol.cjs');
const ACTIVATION = require('./activation.json');

const JOB = 'exact-owned-crossrun';
const LEGACY_COMMIT = '65a9edc5881444af0b9093a5e628f2fe47ea3b2e';
const WORKFLOW_NAME = 'CI security artifact cross-run canary';
const MAX_PR_COMMITS = 250;
const COMMITS_PER_PAGE = 100;

function exactKeys(value, keys) {
  assert(value && typeof value === 'object' && !Array.isArray(value));
  assert.deepEqual(Object.keys(value).sort(), [...keys].sort());
}

function validateBinding(binding, kind, expectedPr) {
  const keys = ['run_id', 'attempt', 'head', 'job_id', 'workflow_id', 'pr', 'artifacts'];
  if (kind === 'preflight') keys.push('deleted_by', 'delete_status', 'deleted_get_status', 'remaining_artifacts');
  exactKeys(binding, keys);
  for (const key of ['run_id', 'attempt', 'job_id', 'workflow_id']) p.positiveInteger(binding[key]);
  assert.equal(binding.pr, expectedPr);
  assert.match(binding.head, /^[0-9a-f]{40}$/);
  if (kind === 'preflight') {
    assert(['ptdbot', 'd4l3k'].includes(binding.deleted_by));
    assert.equal(binding.delete_status, 204);
    assert.equal(binding.deleted_get_status, 404);
    assert.equal(binding.remaining_artifacts, 0);
  }
  const kinds = kind === 'preflight' ? ['preflight'] : Object.keys(p.MEMBERS);
  exactKeys(binding.artifacts, kinds);
  const ids = new Set();
  for (const name of kinds) {
    const item = binding.artifacts[name];
    exactKeys(item, ['id', 'name', 'digest']);
    ids.add(p.positiveInteger(item.id));
    assert.equal(item.name, p.artifactName(binding, name));
    assert.match(item.digest, /^[0-9a-f]{64}$/);
    if (kind === 'producer') assert.equal(item.digest, p.digest(p.archive(name)));
  }
  assert.equal(ids.size, kinds.length);
  return binding;
}

function configuration(value = ACTIVATION) {
  exactKeys(value, ['phase', 'expected_pr', 'actor_ids', 'rest_preflight', 'producer']);
  assert(['preflight', 'producer', 'consumer'].includes(value.phase), 'PR0 is inert');
  p.positiveInteger(value.expected_pr);
  exactKeys(value.actor_ids, ['ptdbot', 'd4l3k']);
  p.positiveInteger(value.actor_ids.ptdbot);
  p.positiveInteger(value.actor_ids.d4l3k);
  assert.notEqual(value.actor_ids.ptdbot, value.actor_ids.d4l3k);
  if (value.phase === 'preflight') {
    assert.equal(value.rest_preflight, null);
    assert.equal(value.producer, null);
  } else {
    validateBinding(value.rest_preflight, 'preflight', value.expected_pr);
    if (value.phase === 'producer') assert.equal(value.producer, null);
    else {
      validateBinding(value.producer, 'producer', value.expected_pr);
      assert.notEqual(value.producer.run_id, value.rest_preflight.run_id);
      assert.notEqual(value.producer.head, value.rest_preflight.head);
      assert.equal(value.producer.workflow_id, value.rest_preflight.workflow_id);
    }
  }
  return value;
}

async function api(endpoint, expectedStatus = 200) {
  assert.match(endpoint, /^repos\/pytorch\/pytorch\/(actions\/(runs|jobs|artifacts|workflows)\/|pulls\/)[A-Za-z0-9_/?=&.-]+$/);
  const token = process.env['INPUT_GITHUB-TOKEN'];
  assert(token);
  const response = await fetch(`https://api.github.com/${endpoint}`, {redirect: 'error', headers: {
    Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'
  }});
  assert.equal(response.status, expectedStatus, 'Owned GitHub metadata request failed');
  return expectedStatus === 200 ? response.json() : null;
}

function actor(value, config) {
  assert(value && ['ptdbot', 'd4l3k'].includes(value.login));
  assert.equal(value.id, config.actor_ids[value.login]);
}

function validateJob(job, context, runnerName) {
  assert.equal(job.id, context.job_id);
  assert.equal(job.run_id, context.run_id);
  assert.equal(job.run_attempt, context.attempt);
  assert.equal(job.head_sha, context.head);
  assert.equal(job.workflow_name, WORKFLOW_NAME);
  assert.equal(job.name, JOB);
  if (runnerName !== undefined) assert.equal(job.runner_name, runnerName);
  return job;
}

async function guardRuntime(config, cleanup = false, request = api) {
  const env = process.env;
  assert.equal(env.GITHUB_JOB, JOB);
  const event = JSON.parse(fs.readFileSync(env.GITHUB_EVENT_PATH, 'utf8'));
  const runId = p.positiveInteger(env.GITHUB_RUN_ID);
  const attempt = p.positiveInteger(env.GITHUB_RUN_ATTEMPT);
  const [run, pr, jobs] = await Promise.all([
    request(`repos/${p.REPOSITORY}/actions/runs/${runId}/attempts/${attempt}`),
    request(`repos/${p.REPOSITORY}/pulls/${config.expected_pr}`),
    request(`repos/${p.REPOSITORY}/actions/runs/${runId}/attempts/${attempt}/jobs?per_page=100`)
  ]);
  const context = p.validateIdentity(env, event, run, pr, config.expected_pr, cleanup);
  for (const item of [event.sender, event.pull_request.user, run.actor, run.triggering_actor, pr.user]) actor(item, config);
  assert.equal(run.actor.login, 'ptdbot');
  assert(Array.isArray(jobs.jobs) && jobs.total_count === jobs.jobs.length && jobs.total_count <= 100);
  const matching = jobs.jobs.filter(item => item.name === JOB);
  assert.equal(matching.length, 1);
  Object.assign(context, {job_id: p.positiveInteger(matching[0].id), workflow_id: p.positiveInteger(run.workflow_id), phase: config.phase});
  validateJob(matching[0], context, env.RUNNER_NAME);
  const workflow = await request(`repos/${p.REPOSITORY}/actions/workflows/${context.workflow_id}`);
  assert.equal(workflow.id, context.workflow_id);
  assert.equal(workflow.path, p.WORKFLOW);
  assert.equal(workflow.name, WORKFLOW_NAME);
  if (config.rest_preflight) assert.equal(context.workflow_id, config.rest_preflight.workflow_id);
  assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], {cwd: env.GITHUB_WORKSPACE, encoding: 'utf8'}).trim(), context.head);
  if (config.phase === 'consumer') {
    assert.notEqual(context.run_id, config.producer.run_id);
    assert.notEqual(context.head, config.producer.head);
    Object.assign(context, {artifact_run_id: config.producer.run_id, artifact_attempt: config.producer.attempt});
  }
  return context;
}

function currentPr(pr, context, config) {
  assert.equal(pr.number, config.expected_pr);
  assert.equal(pr.user.login, 'ptdbot');
  actor(pr.user, config);
  assert.equal(pr.state, 'open');
  assert.equal(pr.draft, true);
  assert.equal(pr.head.repo.id, p.HEAD_REPOSITORY_ID);
  assert.equal(pr.head.repo.full_name, p.HEAD_REPOSITORY);
  assert.equal(pr.head.ref, p.BRANCH);
  assert.equal(pr.head.sha, context.head);
  assert.equal(pr.base.repo.id, p.REPOSITORY_ID);
  assert.equal(pr.base.repo.full_name, p.REPOSITORY);
  assert.equal(pr.base.ref, 'main');
  assert(Number.isSafeInteger(pr.commits) && pr.commits > 0 && pr.commits <= MAX_PR_COMMITS);
  return pr.commits;
}

function checkAncestry(heads, current) {
  const options = {cwd: process.env.GITHUB_WORKSPACE, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']};
  assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], options).trim(), current);
  assert.equal(execFileSync('git', ['rev-parse', '--is-shallow-repository'], options).trim(), 'false', 'Phase history checkout must not be shallow');
  for (const [index, head] of heads.entries()) {
    assert.match(head, /^[0-9a-f]{40}$/);
    assert.notEqual(head, current, 'Earlier phase must precede current head');
    execFileSync('git', ['merge-base', '--is-ancestor', head, current], options);
    if (index) execFileSync('git', ['merge-base', '--is-ancestor', heads[index - 1], head], options);
  }
}

async function phaseHistory(config, context, request = api, ancestry = checkAncestry) {
  assert.equal(context.pr, config.expected_pr);
  assert.match(context.head, /^[0-9a-f]{40}$/);
  const endpoint = `repos/${p.REPOSITORY}/pulls/${config.expected_pr}`;
  const total = currentPr(await request(endpoint), context, config);
  const commits = new Set();
  let last;
  for (let page = 1; commits.size < total; page++) {
    assert(page <= Math.ceil(MAX_PR_COMMITS / COMMITS_PER_PAGE));
    const items = await request(`${endpoint}/commits?per_page=${COMMITS_PER_PAGE}&page=${page}`);
    assert(Array.isArray(items));
    assert.equal(items.length, Math.min(COMMITS_PER_PAGE, total - commits.size));
    for (const item of items) {
      assert.match(item.sha, /^[0-9a-f]{40}$/);
      assert(!commits.has(item.sha), 'Duplicate PR commit');
      commits.add(item.sha);
      last = item.sha;
    }
  }
  assert.equal(last, context.head, 'PR commit list does not end at current head');
  assert.equal(currentPr(await request(endpoint), context, config), total, 'PR history changed while reading');
  const heads = [config.rest_preflight?.head, config.producer?.head].filter(Boolean);
  assert.equal(new Set(heads).size, heads.length);
  for (const head of heads) assert(commits.has(head), 'Earlier phase head missing from current PR');
  ancestry(heads, context.head);
  return {pr: config.expected_pr, current_head: context.head, heads};
}

function validateEarlierRun(run, job, binding, config, history) {
  assert.equal(history.pr, config.expected_pr);
  assert.match(history.current_head, /^[0-9a-f]{40}$/);
  assert(history.heads.includes(binding.head), 'Earlier run lacks append-only PR history');
  assert.equal(run.id, binding.run_id);
  assert.equal(run.run_attempt, binding.attempt);
  assert.equal(run.workflow_id, binding.workflow_id);
  assert.equal(run.path, p.WORKFLOW);
  assert.equal(run.event, 'pull_request');
  assert.equal(run.head_sha, binding.head);
  assert.equal(run.head_branch, p.BRANCH);
  assert.equal(run.repository.id, p.REPOSITORY_ID);
  assert.equal(run.repository.full_name, p.REPOSITORY);
  assert.equal(run.head_repository.id, p.HEAD_REPOSITORY_ID);
  assert.equal(run.head_repository.full_name, p.HEAD_REPOSITORY);
  assert.equal(run.status, 'completed');
  assert.equal(run.conclusion, 'success');
  actor(run.actor, config);
  actor(run.triggering_actor, config);
  assert.equal(run.actor.login, 'ptdbot');
  assert(Array.isArray(run.pull_requests));
  if (run.pull_requests.length) {
    const matches = run.pull_requests.filter(pr => pr.number === binding.pr);
    assert.equal(matches.length, 1);
    const pr = matches[0];
    assert.equal(pr.head.sha, binding.head);
    assert.equal(pr.head.ref, p.BRANCH);
    assert.equal(pr.head.repo.id, p.HEAD_REPOSITORY_ID);
    assert.equal(pr.base.ref, 'main');
    assert.equal(pr.base.repo.id, p.REPOSITORY_ID);
  }
  validateJob(job, binding);
  assert.equal(job.status, 'completed');
  assert.equal(job.conclusion, 'success');
}

function validateArtifact(item, expected, binding) {
  assert.equal(item.id, expected.id);
  assert.equal(item.name, expected.name);
  assert.equal(item.digest, `sha256:${expected.digest}`);
  assert.equal(item.expired, false);
  assert(Number.isSafeInteger(item.size_in_bytes) && item.size_in_bytes > 0 && item.size_in_bytes <= 65536);
  const run = item.workflow_run;
  assert.equal(run.id, binding.run_id);
  assert.equal(run.repository_id, p.REPOSITORY_ID);
  assert.equal(run.head_repository_id, p.HEAD_REPOSITORY_ID);
  assert.equal(run.head_branch, p.BRANCH);
  assert.equal(run.head_sha, binding.head);
}

async function verifyEarlier(binding, config, deleted, context, request = api, ancestry = checkAncestry) {
  const history = await phaseHistory(config, context, request, ancestry);
  const [run, job, inventory] = await Promise.all([
    request(`repos/${p.REPOSITORY}/actions/runs/${binding.run_id}/attempts/${binding.attempt}`),
    request(`repos/${p.REPOSITORY}/actions/jobs/${binding.job_id}`),
    request(`repos/${p.REPOSITORY}/actions/runs/${binding.run_id}/artifacts?per_page=100`)
  ]);
  validateEarlierRun(run, job, binding, config, history);
  const expected = Object.values(binding.artifacts);
  assert(Array.isArray(inventory.artifacts));
  assert.equal(inventory.total_count, inventory.artifacts.length);
  assert.equal(inventory.total_count, deleted ? 0 : expected.length);
  for (const item of expected) {
    const object = await request(`repos/${p.REPOSITORY}/actions/artifacts/${item.id}`, deleted ? 404 : 200);
    if (deleted) continue;
    validateArtifact(object, item, binding);
    const matches = inventory.artifacts.filter(entry => entry.id === item.id);
    assert.equal(matches.length, 1);
    validateArtifact(matches[0], item, binding);
  }
}

function output(name, value) {
  assert.match(name, /^[a-z0-9_]+$/);
  assert(!/[\r\n]/.test(String(value)));
  fs.appendFileSync(process.env.GITHUB_OUTPUT, `${name}=${value}\n`);
}

async function ordinaryPreflight(context, service) {
  const state = p.readState(context);
  assert.equal(Object.keys(state.artifacts).length, 0);
  const file = path.join(p.fixtureRoot(context), 'rest-preflight', 'ordinary.txt');
  fs.mkdirSync(path.dirname(file));
  fs.writeFileSync(file, p.MARKER, {flag: 'wx'});
  const item = {name: p.artifactName(context, 'preflight'), phase: 'pending'};
  state.artifacts.preflight = item;
  p.writeState(context, state);
  const uploaded = await service.upload(item.name, [file], path.dirname(file), {retentionDays: 1, compressionLevel: 0});
  Object.assign(item, {phase: 'finalized', id: p.positiveInteger(uploaded.id), digest: uploaded.digest});
  assert.match(item.digest, /^[0-9a-f]{64}$/);
  p.writeState(context, state);
  const found = await p.listed(service, item.name);
  assert.equal(found.length, 1);
  assert.equal(p.positiveInteger(found[0].databaseId), item.id);
}

function completed(context) {
  const state = p.readState(context);
  if (context.phase === 'preflight') return Object.keys(state.artifacts).length === 1 && state.artifacts.preflight?.phase === 'finalized';
  if (context.phase !== 'producer' || !state.preflight_deleted) return false;
  return Object.keys(state.artifacts).length === 4 && Object.keys(p.MEMBERS).every(kind => state.artifacts[kind]?.phase === 'finalized');
}

function ready(context) {
  const file = path.join(p.fixtureRoot(context), 'ready.json');
  if (!fs.existsSync(file)) return false;
  assert(fs.lstatSync(file).isFile() && fs.realpathSync(file) === file);
  assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')), context);
  return completed(context);
}

async function protocolOperation(mode, context, config) {
  const service = p.protocol(p.verifyDependencies(true));
  fs.writeFileSync(path.join(p.fixtureRoot(context), 'protocol-identity.json'), JSON.stringify(service.ids), {mode: 0o600});
  if (mode === 'protocol-preflight') {
    assert.equal(config.phase, 'preflight');
    return ordinaryPreflight(context, service);
  }
  if (mode === 'protocol-produce') {
    assert.equal(config.phase, 'producer');
    await verifyEarlier(config.rest_preflight, config, true, context);
    return p.produce(context, service);
  }
  assert.equal(mode, 'protocol-cleanup');
  return p.cleanup(context, service);
}

async function isolated(mode, context) {
  const child = spawn(process.execPath, [__filename, `protocol-${mode}`], {stdio: 'ignore', env: process.env});
  const status = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve({code, signal}));
  });
  const state = p.readState(context);
  const identityFile = path.join(p.fixtureRoot(context), 'protocol-identity.json');
  const backend = fs.existsSync(identityFile) ? JSON.parse(fs.readFileSync(identityFile, 'utf8')) : null;
  if (backend) {
    exactKeys(backend, ['workflowRunBackendId', 'workflowJobRunBackendId']);
    for (const value of Object.values(backend)) assert.match(value, /^[0-9a-f-]{36}$/i);
  }
  console.log(JSON.stringify({operation: mode, context, node: process.version, node_executable: process.execPath, sdk_commit: p.SDK_COMMIT, sdk_version: '2.3.2', status, backend, preflight_deleted: state.preflight_deleted, artifacts: state.artifacts, observations: p.readProtocolObservations(context)}));
  assert.equal(status.signal, null);
  assert.equal(status.code, 0, `Isolated ${mode} failed`);
}

async function main(mode) {
  const config = configuration();
  const context = await guardRuntime(config, mode === 'finish' || mode === 'protocol-cleanup');
  if (mode === 'guard') {
    if (config.rest_preflight) await verifyEarlier(config.rest_preflight, config, true, context);
    if (config.phase === 'consumer') await verifyEarlier(config.producer, config, false, context);
    p.initialize(context);
    console.log(JSON.stringify({operation: 'guard', context, node: process.version, node_executable: process.execPath, consumers: {legacy: LEGACY_COMMIT, modern: p.MODERN_COMMIT}, consumer_process_runtime: 'not observed'}));
    output('phase', config.phase);
    if (config.producer) output('producer_run_id', config.producer.run_id);
    if (config.phase === 'consumer') {
      const state = p.readState(context);
      state.preflight_deleted = true;
      for (const [kind, item] of Object.entries(config.producer.artifacts)) state.artifacts[kind] = {...item, phase: 'finalized'};
      p.writeState(context, state);
    }
    return;
  }
  if (mode === 'dependencies') return p.verifyDependencies(false);
  if (mode.startsWith('protocol-')) return protocolOperation(mode, context, config);
  if (mode === 'preflight' || mode === 'produce') {
    assert.equal(config.phase, mode === 'produce' ? 'producer' : 'preflight');
    p.verifyDependencies(true);
    await isolated(mode, context);
    assert(completed(context));
    fs.writeFileSync(path.join(p.fixtureRoot(context), 'ready.json'), JSON.stringify(context), {flag: 'wx', mode: 0o600});
    return;
  }
  if (mode === 'verify') {
    assert.equal(config.phase, 'consumer');
    await verifyEarlier(config.producer, config, false, context);
    return p.verifyExtraction(context);
  }
  assert.equal(mode, 'finish');
  const retain = config.phase !== 'consumer' && ready(context);
  if (config.phase !== 'consumer' && !retain && Object.keys(p.readState(context).artifacts).length) await isolated('cleanup', context);
  console.log(JSON.stringify({operation: 'finish', context, retained_artifacts: retain ? p.readState(context).artifacts : {}}));
  fs.rmSync(p.fixtureRoot(context), {recursive: true});
}

module.exports = {JOB, LEGACY_COMMIT, WORKFLOW_NAME, MAX_PR_COMMITS, COMMITS_PER_PAGE, configuration, validateBinding, validateJob, currentPr, checkAncestry, phaseHistory, validateEarlierRun, validateArtifact, verifyEarlier, ordinaryPreflight, completed, ready, exactKeys};
if (require.main === module) main(process.argv[2] || process.env.INPUT_MODE).catch(() => {
  console.error('Cross-run artifact canary failed closed; raw service errors and credentials are not logged.');
  process.exitCode = 1;
});
