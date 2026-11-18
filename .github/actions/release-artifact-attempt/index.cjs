'use strict';

const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const p = require('./protocol.cjs');
const ACTIVATION = require('./activation.json');

const PRODUCER_JOB = 'exact-owned-producer';
const CONSUMER_JOB = 'exact-owned-consumer';
const CHECK_CI_JOB = 'check-ci / check-ci';
const WORKFLOW_NAME = 'CI security ordinary artifact attempt audit e94e6bc53f124f34';
const SOURCE_COMMIT = '93fdc6758e5513edcd0d86a09e313b5aca05bd64';
const SOURCE_DIRECTORY = '.artifact-attempt-source';
const ACTION_DIRECTORY = '.github/actions/release-artifact-attempt';
const CHECK_CI = '.github/workflows/_check-ci.yml';
const CHECKOUT_STEP = 'Checkout exact-owned head';
const GUARD_STEP = 'Verify exact-owned draft PR and bound attempt';
const HOLD_STEP = 'Await exact-job controller rerun';
const HOLD_MARKER = 'awaiting exact-job controller rerun';
const CONSUMER_STEPS = ['Checkout unchanged downloader source', 'Install exact isolated Python inputs', 'Observe unchanged same-run attempt selection', 'Clean local consumer fixture'];
const PRODUCER_STEPS = ['Checkout immutable artifact SDK lock', 'Verify dependency inputs', 'Install locked artifact runtime', 'Prove deletion and upload the ordinary archive', 'Retain for controller or clean failed producer'];
const STAGES = new Set(['configuration', 'identity', 'selected-source', 'source-jobs', 'artifact-before', 'relay', 'artifact-after', 'hold', 'dependencies', 'produce', 'finish']);
let stage = 'configuration';

function markStage(next) {
  assert(STAGES.has(next));
  stage = next;
  console.log(JSON.stringify({scope: 'owned artifact attempt stage', stage}));
}

function configuration(value = ACTIVATION) {
  assert.deepEqual(Object.keys(value).sort(), ['actor_ids', 'expected_pr', 'producer_attempt', 'workflow_id']);
  for (const key of ['expected_pr', 'workflow_id', 'producer_attempt']) p.positiveInteger(value[key]);
  assert.deepEqual(value.actor_ids, {ptdbot: 283013143, d4l3k: 909104});
  return value;
}

async function api(endpoint, expectedStatus = 200) {
  assert.match(endpoint, /^repos\/pytorch\/pytorch\/(actions\/(runs|jobs|artifacts|workflows)\/|pulls\/|commits\/|git\/trees\/)[A-Za-z0-9_/?=&.-]+$/);
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

function phaseFor(job, attempt, producerAttempt) {
  if (job === PRODUCER_JOB && attempt === producerAttempt) return 'produce';
  if (job === CONSUMER_JOB && attempt === producerAttempt) return 'hold';
  if (job === CONSUMER_JOB && attempt === producerAttempt + 1) return 'consume';
  assert.fail('Unexpected job/attempt tuple');
}

function validateJob(job, context, runnerName) {
  assert.equal(job.id, context.job_id);
  p.positiveInteger(job.id);
  assert.equal(job.run_id, context.run_id);
  assert.equal(job.run_attempt, context.attempt);
  assert.equal(job.head_sha, context.head);
  assert.equal(job.workflow_name, WORKFLOW_NAME);
  assert.equal(job.name, context.job_name);
  p.positiveInteger(job.runner_id);
  assert.equal(job.runner_group_id, 0);
  assert.equal(job.runner_group_name, 'GitHub Actions');
  assert.deepEqual(job.labels, [context.job_name === CHECK_CI_JOB ? 'ubuntu-24.04' : 'ubuntu-22.04']);
  assert.equal(typeof job.runner_name, 'string');
  assert(job.runner_name.length > 0 && job.runner_name.length <= 100);
  if (runnerName !== undefined) assert.equal(job.runner_name, runnerName);
}

function validateSelectedWorkflow(run, context) {
  assert.deepEqual(run.referenced_workflows, [{path: `${p.REPOSITORY}/${CHECK_CI}@${context.merge_sha}`, sha: context.merge_sha, ref: `refs/pull/${context.pr}/merge`}]);
}

function localSource(workspace = process.env.GITHUB_WORKSPACE) {
  const git = args => execFileSync('git', args, {cwd: workspace, encoding: 'utf8'}).trim();
  const head = git(['rev-parse', 'HEAD']);
  const entries = git(['ls-tree', `HEAD:${ACTION_DIRECTORY}`]).split('\n').map(line => {
    const match = /^100644 blob ([0-9a-f]{40})\t([a-zA-Z0-9_.-]+)$/.exec(line);
    assert(match, 'Unexpected action tree entry');
    assert.equal(git(['hash-object', path.join(ACTION_DIRECTORY, match[2])]), match[1], 'Working action source changed');
    return match[2];
  });
  assert.deepEqual(fs.readdirSync(path.join(workspace, ACTION_DIRECTORY)).sort(), entries.sort());
  const result = {head, action: git(['rev-parse', `HEAD:${ACTION_DIRECTORY}`])};
  for (const [key, file] of [['workflow', p.WORKFLOW], ['check_ci', CHECK_CI]]) {
    result[key] = git(['rev-parse', `HEAD:${file}`]);
    assert.equal(git(['hash-object', file]), result[key], 'Working workflow source changed');
  }
  return result;
}

async function selectedSource(run, context, request = api, source = localSource()) {
  validateSelectedWorkflow(run, context);
  assert.equal(source.head, context.head);
  const commit = await request(`repos/${p.REPOSITORY}/commits/${context.merge_sha}`);
  assert.equal(commit.sha, context.merge_sha);
  assert.equal(commit.parents.length, 2);
  assert.equal(commit.parents[1].sha, context.head, 'Selected merge has a different head parent');
  assert.match(commit.parents[0].sha, /^[0-9a-f]{40}$/);
  async function entry(tree, name, type) {
    assert.match(tree, /^[0-9a-f]{40}$/);
    const value = await request(`repos/${p.REPOSITORY}/git/trees/${tree}`);
    assert.equal(value.sha, tree);
    assert.equal(value.truncated, false);
    assert(Array.isArray(value.tree) && value.tree.length <= 10000);
    const matches = value.tree.filter(item => item.path === name);
    assert.equal(matches.length, 1);
    assert.equal(matches[0].type, type);
    assert.equal(matches[0].mode, type === 'tree' ? '040000' : '100644');
    assert.match(matches[0].sha, /^[0-9a-f]{40}$/);
    return matches[0].sha;
  }
  const github = await entry(commit.commit.tree.sha, '.github', 'tree');
  const workflows = await entry(github, 'workflows', 'tree');
  const actions = await entry(github, 'actions', 'tree');
  assert.equal(await entry(workflows, path.posix.basename(p.WORKFLOW), 'blob'), source.workflow, 'Selected workflow source differs');
  assert.equal(await entry(workflows, '_check-ci.yml', 'blob'), source.check_ci, 'Selected approval source differs');
  assert.equal(await entry(actions, 'release-artifact-attempt', 'tree'), source.action, 'Selected action source differs');
}

function jobList(value) {
  assert(Array.isArray(value.jobs) && value.total_count === value.jobs.length && value.total_count <= 100);
  assert.equal(new Set(value.jobs.map(job => job.id)).size, value.jobs.length);
  assert(value.jobs.every(job => [CHECK_CI_JOB, PRODUCER_JOB, CONSUMER_JOB].includes(job.name)), 'Unexpected workflow job');
  assert.equal(new Set(value.jobs.map(job => job.name)).size, value.jobs.length);
  return value.jobs;
}

function onlyJob(jobs, name) {
  const found = jobs.filter(job => job.name === name);
  assert.equal(found.length, 1, 'Expected one exact-owned job');
  return found[0];
}

function completedJob(job, context, name, conclusion) {
  validateJob(job, {...context, job_id: p.positiveInteger(job.id), job_name: name});
  assert.equal(job.status, 'completed');
  assert.equal(job.conclusion, conclusion);
}

function exactSteps(job, expected) {
  assert(Array.isArray(job.steps));
  assert.deepEqual(job.steps.map(item => item.name), expected.map(item => item[0]), 'Unexpected job steps');
  let previous = 0;
  for (let i = 0; i < expected.length; i++) {
    const item = job.steps[i];
    assert(p.positiveInteger(item.number) > previous);
    previous = item.number;
    assert.equal(item.status, 'completed');
    assert.equal(item.conclusion, expected[i][1], 'Unexpected step conclusion');
  }
}

function successfulSteps(names) { return names.map(name => [name, 'success']); }

function validateSourceJobs(value, context) {
  const jobs = jobList(value);
  assert.equal(jobs.length, 3);
  const check = onlyJob(jobs, CHECK_CI_JOB);
  const producer = onlyJob(jobs, PRODUCER_JOB);
  const hold = onlyJob(jobs, CONSUMER_JOB);
  completedJob(check, context, CHECK_CI_JOB, 'success');
  completedJob(producer, context, PRODUCER_JOB, 'success');
  completedJob(hold, context, CONSUMER_JOB, 'failure');
  exactSteps(check, successfulSteps(['Set up job', 'Check for disabled CI', 'Complete job']));
  exactSteps(producer, successfulSteps(['Set up job', CHECKOUT_STEP, GUARD_STEP, ...PRODUCER_STEPS, 'Post Checkout immutable artifact SDK lock', `Post ${CHECKOUT_STEP}`, 'Complete job']));
  exactSteps(hold, [...successfulSteps(['Set up job', CHECKOUT_STEP, GUARD_STEP]), [HOLD_STEP, 'failure'], ...CONSUMER_STEPS.map(name => [name, 'skipped']), ...successfulSteps([`Post ${CHECKOUT_STEP}`, 'Complete job'])]);
  return {check, producer, hold};
}

function validateSourceRun(config, context, run) {
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
  assert.equal(run.conclusion, 'failure');
  actor(run.actor, config);
  actor(run.triggering_actor, config);
  assert.equal(run.actor.login, 'ptdbot');
  validateSelectedWorkflow(run, context);
}

async function sourceAttempt(config, context, request = api) {
  markStage('source-jobs');
  const [run, jobs] = await Promise.all([
    request(`repos/${p.REPOSITORY}/actions/runs/${context.run_id}/attempts/${config.producer_attempt}`),
    request(`repos/${p.REPOSITORY}/actions/runs/${context.run_id}/attempts/${config.producer_attempt}/jobs?per_page=100`)
  ]);
  validateSourceRun(config, context, run);
  return validateSourceJobs(jobs, {...context, attempt: config.producer_attempt});
}

function rerunTarget(config, context, run, jobs, requestedEndpoint) {
  validateSourceRun(config, context, run);
  const source = validateSourceJobs(jobs, {...context, attempt: config.producer_attempt});
  const endpoint = `repos/${p.REPOSITORY}/actions/jobs/${source.hold.id}/rerun`;
  if (requestedEndpoint !== undefined) assert.equal(requestedEndpoint, endpoint, 'Only the exact consumer-job rerun is permitted');
  return {method: 'POST', endpoint, expected_status: 201, body: {enable_debug_logging: false}, producer_job_id: source.producer.id, consumer_job_id: source.hold.id};
}

function validateCurrentJobs(value, context, source) {
  const jobs = jobList(value);
  const current = onlyJob(jobs, CONSUMER_JOB);
  validateJob(current, context, process.env.RUNNER_NAME);
  assert.notEqual(current.id, source.hold.id, 'Hold job was not rerun');
  for (const job of jobs) {
    if (job.name === CONSUMER_JOB) continue;
    const original = job.name === PRODUCER_JOB ? source.producer : source.check;
    for (const key of ['id', 'run_id', 'run_attempt', 'head_sha', 'workflow_name', 'name', 'status', 'conclusion', 'runner_id', 'runner_name', 'runner_group_id', 'runner_group_name', 'labels']) assert.deepEqual(job[key], original[key], 'Unexpected prerequisite rerun');
  }
  return current;
}

async function guard(config, cleanup = false, request = api, verifySource = selectedSource) {
  markStage('identity');
  const env = process.env;
  const runId = p.positiveInteger(env.GITHUB_RUN_ID);
  const attempt = p.positiveInteger(env.GITHUB_RUN_ATTEMPT);
  const phase = phaseFor(env.GITHUB_JOB, attempt, config.producer_attempt);
  const event = JSON.parse(fs.readFileSync(env.GITHUB_EVENT_PATH, 'utf8'));
  const [run, pr, response, workflow] = await Promise.all([
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
  const jobs = jobList(response);
  const current = onlyJob(jobs, env.GITHUB_JOB);
  Object.assign(context, {job_id: p.positiveInteger(current.id), job_name: env.GITHUB_JOB, workflow_id: config.workflow_id, producer_attempt: config.producer_attempt, phase});
  validateJob(current, context, env.RUNNER_NAME);
  markStage('selected-source');
  await verifySource(run, context, request);
  if (phase === 'consume') {
    const source = await sourceAttempt(config, context, request);
    validateCurrentJobs(response, context, source);
    Object.assign(context, {producer_job_id: source.producer.id, hold_job_id: source.hold.id, check_ci_job_id: source.check.id});
    return context;
  }
  completedJob(onlyJob(jobs, CHECK_CI_JOB), context, CHECK_CI_JOB, 'success');
  if (phase === 'hold') completedJob(onlyJob(jobs, PRODUCER_JOB), context, PRODUCER_JOB, 'success');
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
  assert.equal(context.phase, 'consume');
  assert.equal(context.attempt, config.producer_attempt + 1);
  const source = await sourceAttempt(config, context, request);
  assert.equal(source.producer.id, context.producer_job_id);
  assert.equal(source.hold.id, context.hold_job_id);
  assert.equal(source.check.id, context.check_ci_job_id);
  const [jobs, inventory] = await Promise.all([
    request(`repos/${p.REPOSITORY}/actions/runs/${context.run_id}/attempts/${context.attempt}/jobs?per_page=100`),
    request(`repos/${p.REPOSITORY}/actions/runs/${context.run_id}/artifacts?per_page=100`)
  ]);
  validateCurrentJobs(jobs, context, source);
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
  markStage('artifact-before');
  const artifact = await priorArtifact(config, context);
  const root = p.fixtureRoot(context);
  const input = {context, artifact, source_commit: SOURCE_COMMIT};
  fs.writeFileSync(path.join(root, 'consumer-input.json'), JSON.stringify(input), {flag: 'wx', mode: 0o600});
  markStage('relay');
  await require('./tunnel.cjs').runIsolated(root, await redirectHost(artifact), path.join(root, 'venv', 'bin', 'python'), path.join(__dirname, 'consume.py'), path.join(process.env.GITHUB_WORKSPACE, SOURCE_DIRECTORY), process.env['INPUT_GITHUB-TOKEN']);
  const result = JSON.parse(fs.readFileSync(path.join(root, 'consumer-result.json'), 'utf8'));
  assert.deepEqual(result, {artifact_id: artifact.id, digest: artifact.digest, size: artifact.size, producer_attempt: config.producer_attempt, consumer_attempt: context.attempt, ordinary_control: true, mismatch_download: true, torchci_rejected: true, torchci_entered_helpers: [], observed_download_ids: [artifact.id, artifact.id]});
  markStage('artifact-after');
  assert.deepEqual(await priorArtifact(config, context, api, artifact.id), artifact);
  console.log(JSON.stringify({scope: 'owned same-run unchanged helper; controller must compare its pre-rerun artifact ID', context, result}));
}

function completed(context) {
  const state = p.readState(context);
  return state.preflight_deleted && state.artifacts.preflight?.phase === 'deleted' && state.artifacts.ordinary?.phase === 'finalized' && Object.keys(state.artifacts).length === 2;
}

async function finish(context) {
  assert(['produce', 'consume'].includes(context.phase));
  const root = p.fixtureRoot(context);
  const state = p.readState(context);
  const ready = path.join(root, 'producer-ready.json');
  const retain = context.phase === 'produce' && fs.existsSync(ready) && completed(context) && JSON.stringify(JSON.parse(fs.readFileSync(ready, 'utf8'))) === JSON.stringify(context);
  let failed = false;
  try {
    if (context.phase === 'produce' && !retain && Object.values(state.artifacts).some(item => item.phase !== 'deleted')) await p.cleanup(context, p.protocol(p.verifyDependencies(true)));
  } catch { failed = true; }
  console.log(JSON.stringify({scope: 'owned artifact lifecycle', context, cleanup_failed: failed, retain_for_controller: retain || context.phase === 'consume', artifacts: p.readState(context).artifacts, observations: p.readProtocolObservations(context)}));
  assert.equal(fs.realpathSync(root), root);
  fs.rmSync(root, {recursive: true});
  assert.equal(failed, false, 'Independent controller cleanup is required');
}

async function execute(mode, config, context) {
  assert.equal(phaseFor(context.job_name, context.attempt, config.producer_attempt), context.phase);
  if (mode === 'guard') {
    if (context.phase !== 'hold') p.initialize(context);
    p.output('phase', context.phase);
    return 0;
  }
  if (mode === 'hold') {
    assert.equal(context.phase, 'hold');
    markStage('hold');
    console.log(HOLD_MARKER);
    return 1;
  }
  if (mode === 'dependencies') {
    assert.equal(context.phase, 'produce');
    markStage('dependencies');
    p.verifyDependencies(false);
    return 0;
  }
  if (mode === 'produce') {
    assert.equal(context.phase, 'produce');
    markStage('produce');
    await p.produce(context, p.protocol(p.verifyDependencies(true)));
    assert(completed(context));
    const item = p.readState(context).artifacts.ordinary;
    assert.equal(item.digest, p.digest(p.archive('ordinary')));
    fs.writeFileSync(path.join(p.fixtureRoot(context), 'producer-ready.json'), JSON.stringify(context), {flag: 'wx', mode: 0o600});
    console.log(JSON.stringify({scope: 'controller must authenticate before exact consumer-job rerun', context, artifact: {id: item.id, name: item.name, digest: item.digest, size: p.archive('ordinary').length}}));
    return 0;
  }
  if (mode === 'consume') {
    assert.equal(context.phase, 'consume');
    await consume(config, context);
    return 0;
  }
  assert.equal(mode, 'finish');
  markStage('finish');
  await finish(context);
  return 0;
}

async function main() {
  const config = configuration();
  const mode = process.env.INPUT_MODE;
  assert(['guard', 'dependencies', 'produce', 'hold', 'consume', 'finish'].includes(mode));
  const context = await guard(config, mode === 'finish');
  process.exitCode = await execute(mode, config, context);
}

module.exports = {PRODUCER_JOB, CONSUMER_JOB, CHECK_CI_JOB, WORKFLOW_NAME, SOURCE_COMMIT, SOURCE_DIRECTORY, ACTION_DIRECTORY, CHECK_CI, CHECKOUT_STEP, GUARD_STEP, HOLD_STEP, HOLD_MARKER, CONSUMER_STEPS, PRODUCER_STEPS, markStage, configuration, phaseFor, validateJob, validateSelectedWorkflow, localSource, selectedSource, jobList, validateSourceJobs, validateSourceRun, sourceAttempt, rerunTarget, validateCurrentJobs, guard, validateArtifact, priorArtifact, completed, execute};
if (require.main === module) main().catch(() => { console.error(`::error::Exact-owned artifact attempt audit failed closed at ${stage}`); process.exitCode = 1; });
