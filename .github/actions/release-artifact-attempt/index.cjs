'use strict';

const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const p = require('./protocol.cjs');
const ACTIVATION = require('./activation.json');

const JOB = 'exact-owned-attempt';
const WORKFLOW_NAME = 'CI security ordinary artifact attempt audit e94e6bc53f124f34';
const SOURCE_COMMIT = '93fdc6758e5513edcd0d86a09e313b5aca05bd64';
const SOURCE_DIRECTORY = '.artifact-attempt-source';

function configuration(value = ACTIVATION) {
  assert.deepEqual(Object.keys(value).sort(), ['actor_ids', 'expected_pr', 'producer_attempt', 'workflow_id']);
  for (const key of ['expected_pr', 'workflow_id', 'producer_attempt']) p.positiveInteger(value[key]);
  assert.deepEqual(value.actor_ids, {ptdbot: 283013143, d4l3k: 909104});
  return value;
}

async function api(endpoint, expectedStatus = 200) {
  assert.match(endpoint, /^repos\/pytorch\/pytorch\/(actions\/(runs|jobs|artifacts|workflows)\/|pulls\/)[A-Za-z0-9_/?=&.-]+$/);
  const response = await fetch(`https://api.github.com/${endpoint}`, {redirect: 'error', headers: {
    Authorization: `Bearer ${process.env['INPUT_GITHUB-TOKEN']}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'
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
}

async function guard(config, cleanup = false, request = api) {
  const env = process.env;
  assert.equal(env.GITHUB_JOB, JOB);
  const runId = p.positiveInteger(env.GITHUB_RUN_ID);
  const attempt = p.positiveInteger(env.GITHUB_RUN_ATTEMPT);
  assert([config.producer_attempt, config.producer_attempt + 1].includes(attempt), 'Unexpected attempt');
  const event = JSON.parse(fs.readFileSync(env.GITHUB_EVENT_PATH, 'utf8'));
  const [run, pr, jobs, workflow] = await Promise.all([
    request(`repos/${p.REPOSITORY}/actions/runs/${runId}/attempts/${attempt}`),
    request(`repos/${p.REPOSITORY}/pulls/${config.expected_pr}`),
    request(`repos/${p.REPOSITORY}/actions/runs/${runId}/attempts/${attempt}/jobs?per_page=100`),
    request(`repos/${p.REPOSITORY}/actions/workflows/${config.workflow_id}`)
  ]);
  const context = p.validateIdentity(env, event, run, pr, config.expected_pr, cleanup);
  for (const item of [event.sender, event.pull_request.user, run.actor, run.triggering_actor, pr.user]) actor(item, config);
  assert.equal(run.workflow_id, config.workflow_id);
  assert.equal(workflow.id, config.workflow_id);
  assert.equal(workflow.path, p.WORKFLOW);
  assert.equal(workflow.name, WORKFLOW_NAME);
  assert(Array.isArray(jobs.jobs) && jobs.total_count === jobs.jobs.length && jobs.total_count <= 100);
  const matches = jobs.jobs.filter(item => item.name === JOB);
  assert.equal(matches.length, 1);
  Object.assign(context, {job_id: p.positiveInteger(matches[0].id), workflow_id: config.workflow_id, producer_attempt: config.producer_attempt, phase: attempt === config.producer_attempt ? 'producer' : 'consumer'});
  validateJob(matches[0], context, env.RUNNER_NAME);
  assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], {cwd: env.GITHUB_WORKSPACE, encoding: 'utf8'}).trim(), context.head);
  return context;
}

function validateArtifact(item, context, expectedId) {
  const id = p.positiveInteger(item.id);
  if (expectedId !== undefined) assert.equal(id, expectedId, 'Artifact ID changed');
  const name = p.artifactName({...context, artifact_attempt: context.producer_attempt}, 'ordinary');
  const bytes = p.archive('ordinary');
  assert.equal(item.name, name);
  assert.equal(item.digest, `sha256:${p.digest(bytes)}`);
  assert.equal(item.size_in_bytes, bytes.length);
  assert.equal(item.expired, false);
  const run = item.workflow_run;
  assert.equal(run.id, context.run_id);
  assert.equal(run.repository_id, p.REPOSITORY_ID);
  assert.equal(run.head_repository_id, p.HEAD_REPOSITORY_ID);
  assert.equal(run.head_branch, p.BRANCH);
  assert.equal(run.head_sha, context.head);
  return {id, name, digest: p.digest(bytes), size: bytes.length};
}

async function priorArtifact(config, context, request = api, expectedId) {
  assert.equal(context.attempt, config.producer_attempt + 1);
  const [run, jobs, inventory] = await Promise.all([
    request(`repos/${p.REPOSITORY}/actions/runs/${context.run_id}/attempts/${config.producer_attempt}`),
    request(`repos/${p.REPOSITORY}/actions/runs/${context.run_id}/attempts/${config.producer_attempt}/jobs?per_page=100`),
    request(`repos/${p.REPOSITORY}/actions/runs/${context.run_id}/artifacts?per_page=100`)
  ]);
  assert.equal(run.id, context.run_id);
  assert.equal(run.run_attempt, config.producer_attempt);
  assert.equal(run.workflow_id, config.workflow_id);
  assert.equal(run.path, p.WORKFLOW);
  assert.equal(run.event, 'pull_request');
  assert.equal(run.head_sha, context.head);
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
  assert(Array.isArray(jobs.jobs) && jobs.total_count === jobs.jobs.length && jobs.total_count <= 100);
  const selected = jobs.jobs.filter(item => item.name === JOB);
  assert.equal(selected.length, 1);
  validateJob(selected[0], {...context, attempt: config.producer_attempt, job_id: selected[0].id});
  assert.equal(selected[0].status, 'completed');
  assert.equal(selected[0].conclusion, 'success');
  assert(Array.isArray(inventory.artifacts));
  assert.equal(inventory.total_count, 1, 'Unexpected run artifacts');
  assert.equal(inventory.artifacts.length, 1);
  const result = validateArtifact(inventory.artifacts[0], context, expectedId);
  validateArtifact(await request(`repos/${p.REPOSITORY}/actions/artifacts/${result.id}`), context, result.id);
  return result;
}

async function redirectHost(artifact) {
  const response = await fetch(`https://api.github.com/repos/${p.REPOSITORY}/actions/artifacts/${artifact.id}/zip`, {redirect: 'manual', headers: {
    Authorization: `Bearer ${process.env['INPUT_GITHUB-TOKEN']}`, Accept: 'application/vnd.github+json'
  }});
  assert.equal(response.status, 302, 'Expected owned artifact redirect');
  const url = new URL(response.headers.get('location'));
  assert.equal(url.protocol, 'https:');
  assert(!url.username && !url.password && (!url.port || url.port === '443'));
  assert(/^[a-z0-9-]+\.blob\.core\.windows\.net$/.test(url.hostname) || /^[a-z0-9.-]+\.actions\.githubusercontent\.com$/.test(url.hostname));
  return url.hostname;
}

async function consume(config, context) {
  const artifact = await priorArtifact(config, context);
  const root = p.fixtureRoot(context);
  const input = {context, artifact, source_commit: SOURCE_COMMIT};
  fs.writeFileSync(path.join(root, 'consumer-input.json'), JSON.stringify(input), {flag: 'wx', mode: 0o600});
  await require('./tunnel.cjs').runIsolated(root, await redirectHost(artifact), path.join(root, 'venv', 'bin', 'python'), path.join(__dirname, 'consume.py'), path.join(process.env.GITHUB_WORKSPACE, SOURCE_DIRECTORY), process.env['INPUT_GITHUB-TOKEN']);
  const result = JSON.parse(fs.readFileSync(path.join(root, 'consumer-result.json'), 'utf8'));
  assert.deepEqual(result, {artifact_id: artifact.id, digest: artifact.digest, size: artifact.size, producer_attempt: config.producer_attempt, consumer_attempt: context.attempt, ordinary_control: true, mismatch_download: true, torchci_rejected: true, torchci_entered_helpers: [], observed_download_ids: [artifact.id, artifact.id]});
  assert.deepEqual(await priorArtifact(config, context, api, artifact.id), artifact);
  console.log(JSON.stringify({scope: 'owned same-run unchanged helper; controller must compare its pre-rerun artifact ID', context, result}));
}

function completed(context) {
  const state = p.readState(context);
  return state.preflight_deleted && state.artifacts.preflight?.phase === 'deleted' && state.artifacts.ordinary?.phase === 'finalized' && Object.keys(state.artifacts).length === 2;
}

async function finish(config, context) {
  const root = p.fixtureRoot(context);
  const state = p.readState(context);
  const ready = path.join(root, 'producer-ready.json');
  const retain = context.phase === 'producer' && fs.existsSync(ready) && completed(context) && JSON.stringify(JSON.parse(fs.readFileSync(ready, 'utf8'))) === JSON.stringify(context);
  let failed = false;
  try {
    if (context.phase === 'producer' && !retain && Object.values(state.artifacts).some(item => item.phase !== 'deleted')) await p.cleanup(context, p.protocol(p.verifyDependencies(true)));
  } catch { failed = true; }
  console.log(JSON.stringify({scope: 'owned artifact lifecycle', context, cleanup_failed: failed, retain_for_controller: retain || context.phase === 'consumer', artifacts: p.readState(context).artifacts, observations: p.readProtocolObservations(context)}));
  assert.equal(fs.realpathSync(root), root);
  fs.rmSync(root, {recursive: true});
  assert.equal(failed, false, 'Independent controller cleanup is required');
}

async function main() {
  const config = configuration();
  const mode = process.env.INPUT_MODE;
  assert(['guard', 'dependencies', 'produce', 'consume', 'finish'].includes(mode));
  const context = await guard(config, mode === 'finish');
  if (mode === 'guard') {
    p.initialize(context);
    p.output('phase', context.phase);
    return;
  }
  if (mode === 'dependencies') return p.verifyDependencies(false);
  if (mode === 'produce') {
    assert.equal(context.phase, 'producer');
    await p.produce(context, p.protocol(p.verifyDependencies(true)));
    assert(completed(context));
    const item = p.readState(context).artifacts.ordinary;
    assert.equal(item.digest, p.digest(p.archive('ordinary')));
    fs.writeFileSync(path.join(p.fixtureRoot(context), 'producer-ready.json'), JSON.stringify(context), {flag: 'wx', mode: 0o600});
    console.log(JSON.stringify({scope: 'controller must authenticate before rerun', context, artifact: {id: item.id, name: item.name, digest: item.digest, size: p.archive('ordinary').length}}));
    return;
  }
  if (mode === 'consume') {
    assert.equal(context.phase, 'consumer');
    return consume(config, context);
  }
  return finish(config, context);
}

module.exports = {JOB, WORKFLOW_NAME, SOURCE_COMMIT, SOURCE_DIRECTORY, configuration, validateJob, guard, validateArtifact, priorArtifact, completed};
if (require.main === module) main().catch(() => { console.error('::error::Exact-owned artifact attempt audit failed closed'); process.exitCode = 1; });
