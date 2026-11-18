'use strict';

const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const p = require('./protocol.cjs');
const a = require('./index.cjs');
const tunnel = require('./tunnel.cjs');

const CONFIG = {expected_pr: 123, workflow_id: 456, producer_attempt: 2, actor_ids: {ptdbot: 283013143, d4l3k: 909104}};
const context = {run_id: 100, attempt: 2, producer_attempt: 2, workflow_id: 456, job_id: 200, job_name: a.PRODUCER_JOB, head: 'a'.repeat(40), merge_sha: 'b'.repeat(40), pr: 123, actor: 'ptdbot', triggering_actor: 'd4l3k', phase: 'produce', repository: p.REPOSITORY};

function workflowContract() {
  const workspace = execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim();
  const program = `import json,pathlib,runpy,yaml\np=pathlib.Path(${JSON.stringify(p.WORKFLOW)})\nr=runpy.run_path('.github/scripts/ensure_actions_will_cancel.py')\nprint(json.dumps({'workflow':yaml.load(p.read_text(),Loader=yaml.BaseLoader),'action':yaml.load(pathlib.Path('.github/actions/release-artifact-attempt/action.yml').read_text(),Loader=yaml.BaseLoader),'selected':r['should_check'](p),'expected_group':r['EXPECTED_GROUP_STANDARD']}))`;
  return JSON.parse(execFileSync('uv', ['run', '--no-project', '--python', path.join(workspace, '.venv', 'bin', 'python'), 'python', '-I', '-c', program], {cwd: workspace, encoding: 'utf8'}));
}

async function fixture(callback) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'artifact-attempt-test-'));
  const old = {...process.env};
  process.env.RUNNER_TEMP = root;
  process.env.GITHUB_OUTPUT = path.join(root, 'output');
  try { return await callback(root); } finally {
    for (const key of Object.keys(process.env)) if (!(key in old)) delete process.env[key];
    Object.assign(process.env, old);
    fs.rmSync(root, {recursive: true});
  }
}

function service() {
  const ids = {workflowRunBackendId: '11111111-1111-1111-1111-111111111111', workflowJobRunBackendId: '22222222-2222-2222-2222-222222222222'};
  const objects = new Map();
  const calls = [];
  let next = 10;
  const result = {ids, objects, calls, StringValue: {create: x => x}, expiration: x => ({days: x}),
    upload: async name => { const id = next++; objects.set(name, {...ids, name, databaseId: id}); calls.push(['upload', name]); return {id, digest: 'c'.repeat(64)}; },
    uploadBlob: async (_url, stream) => { const pieces = []; for await (const x of stream) pieces.push(x); const b = Buffer.concat(pieces); calls.push(['blob', b]); return {uploadSize: b.length, sha256Hash: p.digest(b)}; },
    client: {
      ListArtifacts: async request => ({artifacts: [...objects.values()].filter(x => x.name === request.nameFilter.value)}),
      DeleteArtifact: async request => { const obj = objects.get(request.name); objects.delete(request.name); calls.push(['delete', request.name]); return {ok: !!obj, artifactId: String(obj?.databaseId ?? 0)}; },
      CreateArtifact: async request => { calls.push(['create', request.name]); return {ok: true, signedUploadUrl: 'unit-only'}; },
      FinalizeArtifact: async request => { const id = next++; objects.set(request.name, {...ids, name: request.name, databaseId: id}); calls.push(['finalize', request.name]); return {ok: true, artifactId: String(id)}; }
    }};
  return result;
}

test('activation binds the allocated draft and rejects zero identities', () => {
  assert.deepEqual(a.configuration(), {expected_pr: 205523, workflow_id: 422814113, producer_attempt: 2, actor_ids: CONFIG.actor_ids});
  assert.equal(p.EXPECTED_PR, 205523);
  assert.deepEqual(a.configuration(CONFIG), CONFIG);
  for (const key of ['expected_pr', 'workflow_id', 'producer_attempt']) assert.throws(() => a.configuration({...CONFIG, [key]: 0}));
  assert.throws(() => a.configuration({...CONFIG, actor_ids: {...CONFIG.actor_ids, ptdbot: 1}}));
  const yaml = fs.readFileSync(path.join(__dirname, '../../workflows/ci-security-ordinary-artifact-attempt-e94e6bc53f124f34.yml'), 'utf8');
  assert.equal(yaml.match(/pull_request\.number == 205523/g).length, 3);
  assert.equal(/pull_request\.number == 0(?:\D|$)/.test(yaml), false);
  assert(yaml.includes('uses: ./.github/workflows/_check-ci.yml'));
  assert(!yaml.includes('id-token: write'));
});

test('actual workflow, action, SDK source and concurrency contracts agree', async () => fixture(root => {
  const {workflow, action, selected, expected_group} = workflowContract();
  assert.equal(workflow.name, a.WORKFLOW_NAME);
  assert.deepEqual(workflow.on, {pull_request: {types: ['opened', 'synchronize', 'reopened']}});
  assert.equal(selected, true);
  assert.deepEqual(workflow.concurrency, {group: expected_group, 'cancel-in-progress': 'true'});
  assert.deepEqual(workflow.permissions, {contents: 'read', actions: 'read'});
  assert.equal(action.runs.using, 'node20');
  assert.equal(action.runs.main, 'index.cjs');
  assert(fs.statSync(path.join(__dirname, action.runs.main)).isFile());
  assert.deepEqual(Object.keys(workflow.jobs), ['check-ci', a.PRODUCER_JOB, a.CONSUMER_JOB]);
  assert.equal(workflow.jobs[a.PRODUCER_JOB].needs, 'check-ci');
  assert.deepEqual(workflow.jobs[a.CONSUMER_JOB].needs, ['check-ci', a.PRODUCER_JOB]);
  assert.equal(Object.values(workflow.jobs).some(job => (Array.isArray(job.needs) ? job.needs : [job.needs]).includes(a.CONSUMER_JOB)), false);
  const steps = workflow.jobs[a.PRODUCER_JOB].steps;
  const consumer = workflow.jobs[a.CONSUMER_JOB].steps;
  assert.deepEqual(steps.map(step => step.name), [a.CHECKOUT_STEP, a.GUARD_STEP, ...a.PRODUCER_STEPS]);
  assert.deepEqual(consumer.map(step => step.name), [a.CHECKOUT_STEP, a.GUARD_STEP, a.HOLD_STEP, ...a.CONSUMER_STEPS]);
  assert.equal(consumer.find(step => step.name === a.HOLD_STEP)['continue-on-error'], undefined);
  assert.equal(consumer.some(step => ['dependencies', 'produce'].includes(step.with?.mode) || step.with?.repository === 'actions/download-artifact' || /npm ci/.test(step.run || '')), false);
  const sdk = steps.find(step => step.with?.repository === 'actions/download-artifact');
  assert.equal(sdk.with.ref, p.SDK_COMMIT);
  assert.equal(sdk.with.path, p.SDK_DIRECTORY);
  assert.equal(steps.find(step => step.name === 'Install locked artifact runtime')['working-directory'], sdk.with.path);
  const source = consumer.find(step => step.with?.repository === p.REPOSITORY);
  assert.equal(source.with.ref, a.SOURCE_COMMIT);
  assert.equal(source.with.path, a.SOURCE_DIRECTORY);
  for (const mode of ['dependencies', 'produce']) assert.equal(steps.find(step => step.with?.mode === mode).if, "steps.guard.outputs.phase == 'produce'");
  for (const mode of ['hold', 'consume']) assert.equal(consumer.find(step => step.with?.mode === mode).if, `steps.guard.outputs.phase == '${mode}'`);
  assert.equal(consumer.find(step => step.with?.mode === 'finish').if, "always() && steps.guard.outcome == 'success' && steps.guard.outputs.phase == 'consume'");
  for (const step of [...steps, ...consumer].filter(step => step.with?.mode)) assert.equal(step.uses, './.github/actions/release-artifact-attempt');
  const workspace = execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim();
  const sdkSource = path.join(workspace, 'agent_space', 'artifact-attempt-sdk-source');
  assert(fs.existsSync(sdkSource), 'Provision the immutable SDK source checkout first');
  fs.symlinkSync(sdkSource, path.join(root, sdk.with.path), 'dir');
  process.env.GITHUB_WORKSPACE = root;
  assert.equal(p.verifyDependencies(false), path.join(root, sdk.with.path, 'node_modules', '@actions', 'artifact'));
}));

test('single ordinary ZIP is deterministic and cannot enter production prefixes', () => {
  const b = p.archive('ordinary');
  assert.deepEqual(b, p.archive('ordinary'));
  assert.equal(b.readUInt32LE(0), 0x04034b50);
  assert.equal(b.readUInt16LE(8), 0);
  assert.equal(b.subarray(30, 30 + b.readUInt16LE(26)).toString(), 'canary.txt');
  assert.equal(b.readUInt32LE(14), p.crc32(p.MARKER));
  assert.equal(b.readUInt32LE(b.length - 22), 0x06054b50);
  assert.equal(b.readUInt16LE(b.length - 14), 1);
  assert.equal(p.crc32(Buffer.from('123456789')), 0xcbf43926);
  assert.throws(() => p.archive('embedded'));
  for (const kind of ['ordinary', 'preflight']) {
    const name = p.artifactName(context, kind);
    assert(name.includes('runattempt2'));
    for (const prefix of ['sccache-stats', 'test-jsons', 'test-reports', 'torchci-reports', 'usage-log']) assert(!name.startsWith(prefix));
  }
});

test('SDK lock and source hashes reject mutations', async () => fixture(root => {
  fs.writeFileSync(path.join(root, 'x'), 'original');
  const hashes = {x: p.digest(Buffer.from('original'))};
  p.verifyHashes(root, hashes);
  fs.writeFileSync(path.join(root, 'x'), 'changed');
  assert.throws(() => p.verifyHashes(root, hashes));
  assert.equal(p.SDK_COMMIT, 'd3f86a106a0bac45b974a628896c90dbdf5c8093');
}));

test('producer confirms preflight deletion before any experiment create', async () => fixture(async () => {
  p.initialize(context);
  const s = service();
  await p.produce(context, s);
  assert(a.completed(context));
  assert(s.calls.findIndex(x => x[0] === 'delete') < s.calls.findIndex(x => x[0] === 'create'));
  assert.deepEqual(s.calls.find(x => x[0] === 'blob')[1], p.archive('ordinary'));
  assert.equal(s.objects.size, 1);
  await p.cleanup(context, s);
  assert.equal(s.objects.size, 0);
}));

test('negative delete response cannot unlock producer', async () => fixture(async () => {
  p.initialize(context);
  const s = service();
  s.client.DeleteArtifact = async () => ({ok: false, artifactId: '0'});
  await assert.rejects(p.produce(context, s), {code: 'ERR_ASSERTION', operator: 'strictEqual', actual: false, expected: true});
  assert.equal(p.readState(context).preflight_deleted, false);
  assert.equal(s.calls.some(x => x[0] === 'create'), false);
}));

test('mismatched successful deletion ID cannot unlock producer', async () => fixture(async () => {
  p.initialize(context);
  const s = service();
  s.client.DeleteArtifact = async request => { s.objects.delete(request.name); return {ok: true, artifactId: '999'}; };
  await assert.rejects(p.produce(context, s), {code: 'ERR_ASSERTION', operator: 'strictEqual', actual: 999, expected: 10});
  assert.equal(p.readState(context).preflight_deleted, false);
  assert.equal(s.calls.some(x => x[0] === 'create'), false);
}));

test('bounded deletion polling admits genuine delayed absence', async () => fixture(async () => {
  const s = service();
  const item = {...s.ids, name: 'unit-owned', databaseId: 41};
  s.objects.set(item.name, item);
  let deleted = false, polls = 0, clock = 0;
  s.client.DeleteArtifact = async () => { deleted = true; return {ok: true, artifactId: '41'}; };
  s.client.ListArtifacts = async () => ({artifacts: deleted && ++polls === 3 ? [] : [item]});
  await p.removeArtifact(s, {name: item.name, id: 41, phase: 'finalized'}, true, () => {}, {now: () => clock, wait: async n => { clock += n; }});
  assert.equal(polls, 3);
}));

test('persistent deletion visibility fails at finite bound', async () => fixture(async () => {
  const s = service();
  const item = {...s.ids, name: 'unit-owned', databaseId: 41};
  s.client.DeleteArtifact = async () => ({ok: true, artifactId: '41'});
  s.client.ListArtifacts = async () => ({artifacts: [item]});
  let clock = 0, waits = 0;
  await assert.rejects(p.removeArtifact(s, {name: item.name, id: 41, phase: 'finalized'}, true, () => {}, {now: () => clock, wait: async n => { clock += n; waits++; }}), /visibility deadline/);
  assert(waits <= p.DELETE_VISIBILITY_MAX_POLLS);
}));

test('post-delete owner and ID substitution fails immediately', async () => fixture(async () => {
  for (const [field, substitute] of [['name', 'other'], ['workflowRunBackendId', 'other'], ['workflowJobRunBackendId', 'other'], ['databaseId', 99]]) {
    const s = service();
    const item = {...s.ids, name: 'unit-owned', databaseId: 41};
    const original = item[field];
    let deleted = false;
    s.client.DeleteArtifact = async () => { deleted = true; return {ok: true, artifactId: '41'}; };
    s.client.ListArtifacts = async () => ({artifacts: [deleted ? {...item, [field]: substitute} : item]});
    await assert.rejects(p.removeArtifact(s, {name: item.name, id: 41, phase: 'finalized'}, true), {code: 'ERR_ASSERTION', operator: 'strictEqual', actual: substitute, expected: original});
  }
}));

function finishedSteps(names, failure, skipped = []) {
  return names.map((name, i) => ({name, number: i + 1, status: 'completed', conclusion: name === failure ? 'failure' : skipped.includes(name) ? 'skipped' : 'success'}));
}

function job(name, id, attempt, ctx = context, conclusion = 'success') {
  return {id, run_id: ctx.run_id, run_attempt: attempt, head_sha: ctx.head, workflow_name: a.WORKFLOW_NAME, name, status: 'completed', conclusion, runner_name: 'unit-runner', runner_id: 555, runner_group_id: 0, runner_group_name: 'GitHub Actions', labels: [name === a.CHECK_CI_JOB ? 'ubuntu-24.04' : 'ubuntu-22.04']};
}

function metadata(ctx = {...context, attempt: 3, job_id: 202, job_name: a.CONSUMER_JOB, phase: 'consume', producer_job_id: 200, hold_job_id: 201, check_ci_job_id: 199}) {
  const actor = {login: 'ptdbot', id: CONFIG.actor_ids.ptdbot};
  const run = {id: ctx.run_id, run_attempt: 2, workflow_id: 456, path: p.WORKFLOW, event: 'pull_request', head_sha: ctx.head, head_branch: p.BRANCH, repository: {id: p.REPOSITORY_ID, full_name: p.REPOSITORY}, head_repository: {id: p.HEAD_REPOSITORY_ID, full_name: p.HEAD_REPOSITORY}, status: 'completed', conclusion: 'failure', actor, triggering_actor: {login: 'd4l3k', id: CONFIG.actor_ids.d4l3k}, referenced_workflows: [{path: `${p.REPOSITORY}/${a.CHECK_CI}@${ctx.merge_sha}`, sha: ctx.merge_sha, ref: `refs/pull/${ctx.pr}/merge`}]};
  const check = {...job(a.CHECK_CI_JOB, 199, 2, ctx), steps: finishedSteps(['Set up job', 'Check for disabled CI', 'Complete job'])};
  const producer = {...job(a.PRODUCER_JOB, 200, 2, ctx), steps: finishedSteps(['Set up job', a.CHECKOUT_STEP, a.GUARD_STEP, ...a.PRODUCER_STEPS, 'Post Checkout immutable artifact SDK lock', `Post ${a.CHECKOUT_STEP}`, 'Complete job'])};
  const hold = {...job(a.CONSUMER_JOB, 201, 2, ctx, 'failure'), steps: finishedSteps(['Set up job', a.CHECKOUT_STEP, a.GUARD_STEP, a.HOLD_STEP, ...a.CONSUMER_STEPS, `Post ${a.CHECKOUT_STEP}`, 'Complete job'], a.HOLD_STEP, a.CONSUMER_STEPS)};
  const current = {...job(a.CONSUMER_JOB, ctx.job_id, 3, ctx, null), status: 'in_progress'};
  const artifact = {id: 99, name: p.artifactName({...ctx, artifact_attempt: 2}, 'ordinary'), digest: `sha256:${p.digest(p.archive('ordinary'))}`, size_in_bytes: p.archive('ordinary').length, expired: false, workflow_run: {id: ctx.run_id, repository_id: p.REPOSITORY_ID, head_repository_id: p.HEAD_REPOSITORY_ID, head_branch: p.BRANCH, head_sha: ctx.head}};
  const currentRun = {...run, run_attempt: 3, previous_attempt_url: `https://api.github.com/repos/${p.REPOSITORY}/actions/runs/${ctx.run_id}/attempts/2`, run_started_at: '2026-11-18T18:39:11Z'};
  Object.assign(ctx, {source_jobs_digest: a.sourceFingerprint({check, producer, hold}), current_run_started_at: currentRun.run_started_at});
  return {ctx, run, currentRun, check, producer, hold, current, artifact};
}

function priorJobs(data) { return {total_count: 3, jobs: [data.check, data.producer, data.hold]}; }
function currentJobs(data) { return {total_count: 3, jobs: [data.check, data.producer, data.current]}; }

function requestFixture(data) {
  return async endpoint => {
    if (endpoint.endsWith('/attempts/2')) return data.run;
    if (endpoint.endsWith('/attempts/3')) return data.currentRun;
    if (endpoint.endsWith('/attempts/2/jobs?per_page=100')) return priorJobs(data);
    if (endpoint.endsWith('/attempts/3/jobs?per_page=100')) return currentJobs(data);
    if (endpoint.endsWith('/artifacts?per_page=100')) return {total_count: data.extra ? 2 : 1, artifacts: data.extra ? [data.artifact, data.extra] : [data.artifact]};
    if (endpoint.endsWith(`/artifacts/${data.artifact.id}`)) return data.detail ?? data.artifact;
    throw new Error('Unexpected unit request');
  };
}

test('N+1 authenticates successful producer and sole failed hold', async () => {
  const data = metadata();
  const result = await a.priorArtifact(CONFIG, data.ctx, requestFixture(data));
  assert.equal(result.id, 99);
  assert.equal(result.name, `${p.PREFIX}runattempt2-canary.zip`);
  assert.deepEqual(await a.priorArtifact(CONFIG, data.ctx, requestFixture(data), 99), result);
  await assert.rejects(a.priorArtifact(CONFIG, data.ctx, requestFixture(data), 100), /Artifact ID changed/);
});

test('extra archives and changed artifact metadata reject', async () => {
  const data = metadata();
  await assert.rejects(a.priorArtifact(CONFIG, data.ctx, requestFixture({...data, extra: {...data.artifact, id: 100}})), /Unexpected run artifacts/);
  const absent = async endpoint => endpoint.endsWith('/artifacts?per_page=100') ? {total_count: 0, artifacts: []} : requestFixture(data)(endpoint);
  await assert.rejects(a.priorArtifact(CONFIG, data.ctx, absent), /Unexpected run artifacts/);
  for (const [key, value] of [['id', 0], ['name', 'test-reports-runattempt2-x'], ['digest', `sha256:${'0'.repeat(64)}`], ['size_in_bytes', 1], ['expired', true]]) {
    const changed = structuredClone(data); changed.artifact[key] = value;
    await assert.rejects(a.priorArtifact(CONFIG, changed.ctx, requestFixture(changed)));
  }
  for (const key of ['id', 'repository_id', 'head_repository_id', 'head_branch', 'head_sha']) {
    const changed = structuredClone(data); changed.artifact.workflow_run[key] = 'wrong';
    await assert.rejects(a.priorArtifact(CONFIG, changed.ctx, requestFixture(changed)));
  }
  const substituted = structuredClone(data); substituted.detail = {...substituted.artifact, id: 100};
  await assert.rejects(a.priorArtifact(CONFIG, substituted.ctx, requestFixture(substituted)), /Artifact ID changed/);
});

test('wrong source attempt, run, head and workflow reject', async () => {
  for (const [key, value] of [['id', 101], ['run_attempt', 1], ['workflow_id', 457], ['path', 'other'], ['event', 'push'], ['head_sha', 'c'.repeat(40)], ['head_branch', 'other'], ['status', 'in_progress'], ['conclusion', 'success']]) {
    const changed = metadata(); changed.run[key] = value;
    await assert.rejects(a.priorArtifact(CONFIG, changed.ctx, requestFixture(changed)));
  }
  const data = metadata();
  await assert.rejects(a.priorArtifact(CONFIG, {...data.ctx, attempt: 4}, requestFixture(data)));
});

test('every unapproved source failure and unexpected hold operation rejects', async () => {
  const mutations = [d => { d.check.conclusion = 'skipped'; }, d => { d.producer.conclusion = 'failure'; }, d => { d.producer.status = 'queued'; }, d => { d.hold.conclusion = 'success'; }, d => { d.hold.steps.find(s => s.name === a.GUARD_STEP).conclusion = 'failure'; }, d => { d.hold.steps.find(s => s.name === a.HOLD_STEP).conclusion = 'success'; }, d => { d.hold.steps.find(s => s.name === a.CONSUMER_STEPS[0]).conclusion = 'success'; }, d => { d.hold.steps.push({name: 'unexpected upload', number: 99, status: 'completed', conclusion: 'success'}); }, d => { d.producer.runner_group_name = 'self-hosted'; }, d => { d.hold.labels = ['self-hosted']; }, d => { d.run.referenced_workflows[0].sha = 'c'.repeat(40); }];
  for (const mutate of mutations) { const data = metadata(); mutate(data); await assert.rejects(a.priorArtifact(CONFIG, data.ctx, requestFixture(data))); }
  const data = metadata();
  assert.throws(() => a.validateSourceJobs({total_count: 4, jobs: [...priorJobs(data).jobs, {...data.hold, id: 999, name: 'other'}]}, {...data.ctx, attempt: 2}));
});

test('numeric source and consumer job substitutions reject', async () => {
  for (const role of ['check', 'producer', 'hold', 'current']) {
    const data = metadata(); data[role].id = 999;
    await assert.rejects(a.priorArtifact(CONFIG, data.ctx, requestFixture(data)));
  }
  const data = metadata();
  assert.throws(() => a.validateSourceJobs({total_count: 3, jobs: [data.check, data.producer, {...data.hold, id: data.producer.id}]}, {...data.ctx, attempt: 2}));
});

test('partial rerun retains prerequisite IDs and never permits all-jobs selection', () => {
  const data = metadata();
  const source = a.validateSourceJobs(priorJobs(data), {...data.ctx, attempt: 2});
  assert.equal(a.validateCurrentJobs(currentJobs(data), data.ctx, source, data.currentRun).id, 202);
  assert.equal(a.validateCurrentJobs({total_count: 1, jobs: [data.current]}, data.ctx, source, data.currentRun).id, 202);
  for (const field of ['id', 'run_attempt', 'runner_id', 'head_sha']) {
    const changed = structuredClone(data); changed.producer[field] = field === 'head_sha' ? 'c'.repeat(40) : 999;
    assert.throws(() => a.validateCurrentJobs(currentJobs(changed), data.ctx, source, data.currentRun), /Unexpected prerequisite rerun|strictly equal|Invalid job chronology|must be of type/);
  }
  assert.throws(() => a.validateCurrentJobs({total_count: 1, jobs: [{...data.current, id: 201}]}, {...data.ctx, job_id: 201}, source, data.currentRun), /Hold job was not rerun/);
  const target = a.rerunTarget(CONFIG, data.ctx, data.run, priorJobs(data));
  assert.deepEqual(target, {method: 'POST', endpoint: `repos/${p.REPOSITORY}/actions/jobs/201/rerun`, expected_status: 201, body: {enable_debug_logging: false}, producer_job_id: 200, consumer_job_id: 201, source_jobs: source, source_jobs_digest: a.sourceFingerprint(source)});
  for (const endpoint of [`repos/${p.REPOSITORY}/actions/runs/100/rerun`, `repos/${p.REPOSITORY}/actions/runs/100/rerun-failed-jobs`, `repos/${p.REPOSITORY}/actions/jobs/200/rerun`]) assert.throws(() => a.rerunTarget(CONFIG, data.ctx, data.run, priorJobs(data), endpoint), /Only the exact consumer-job/);
});

test('fixed role tuples and inert hold cannot invoke uploader or consumer', async () => fixture(async root => {
  assert.equal(a.phaseFor(a.PRODUCER_JOB, 2, 2), 'produce');
  assert.equal(a.phaseFor(a.CONSUMER_JOB, 2, 2), 'hold');
  assert.equal(a.phaseFor(a.CONSUMER_JOB, 3, 2), 'consume');
  for (const name of [a.PRODUCER_JOB, a.CONSUMER_JOB, 'other']) for (const attempt of [1, 2, 3, 4]) {
    if ((name === a.PRODUCER_JOB && attempt === 2) || (name === a.CONSUMER_JOB && [2, 3].includes(attempt))) continue;
    assert.throws(() => a.phaseFor(name, attempt, 2), /Unexpected job\/attempt tuple/);
  }
  const hold = {...context, job_name: a.CONSUMER_JOB, job_id: 201, phase: 'hold'};
  await a.execute('guard', CONFIG, hold);
  assert.equal(fs.existsSync(p.fixtureRoot(hold)), false);
  assert.equal(await a.execute('hold', CONFIG, hold), 1);
  assert.equal(fs.existsSync(p.fixtureRoot(hold)), false);
  assert.equal(fs.readdirSync(root).join(','), 'output');
  for (const mode of ['dependencies', 'produce', 'consume', 'finish']) await assert.rejects(a.execute(mode, CONFIG, hold));
  const consumer = metadata().ctx;
  for (const mode of ['dependencies', 'produce', 'hold']) await assert.rejects(a.execute(mode, CONFIG, consumer));
  assert.throws(() => a.markStage('unbounded secret diagnostic'));
  await a.execute('guard', CONFIG, consumer);
  assert.deepEqual(p.readState(consumer).artifacts, {});
  await a.execute('finish', CONFIG, consumer);
  assert.equal(fs.existsSync(p.fixtureRoot(consumer)), false);

}));

test('private and special addresses cannot reach the HTTPS relay', () => {
  for (const value of ['0.0.0.0', '10.1.2.3', '100.64.0.1', '100.127.255.255', '127.0.0.1', '169.254.169.254', '172.16.1.1', '172.31.255.255', '192.0.0.9', '192.0.2.1', '192.31.196.1', '192.52.193.1', '192.88.99.1', '192.168.1.1', '192.175.48.1', '198.18.0.1', '198.19.255.255', '198.51.100.1', '203.0.113.1', '224.0.0.1', '240.0.0.1', '255.255.255.255', '::', '::1', '0:0:0:0:0:0:0:1', '::ffff:127.0.0.1', '0:0:0:0:0:ffff:7f00:1', '::ffff:140.82.112.6', 'fd00::1', 'fe80::1', 'ff00::1', '2606:50c0:8000::154', '0177.0.0.1', 'not-ip']) assert.equal(tunnel.eligibleIPv4Address(value), false, value);
  for (const value of ['100.63.255.255', '100.128.0.0', '172.15.255.255', '172.32.0.0', '198.17.255.255', '198.20.0.0', '140.82.112.6', '20.60.0.1']) assert.equal(tunnel.eligibleIPv4Address(value), true, value);
});

test('original source hash pins match immutable Git objects', () => {
  const consumer = fs.readFileSync(path.join(__dirname, 'consume.py'), 'utf8');
  for (const file of ['tools/stats/upload_stats_lib.py', 'tools/stats/upload_artifacts.py']) {
    const bytes = execFileSync('git', ['show', `${a.SOURCE_COMMIT}:${file}`]);
    assert(consumer.includes(p.digest(bytes)));
  }
});

test('runtime guard binds the owned draft, actors, workflow, runner and exact tuples', async () => fixture(async root => {
  const workspace = execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim();
  const head = execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  const user = {login: 'ptdbot', id: CONFIG.actor_ids.ptdbot};
  const upstream = {id: p.REPOSITORY_ID, full_name: p.REPOSITORY};
  const fork = {id: p.HEAD_REPOSITORY_ID, full_name: p.HEAD_REPOSITORY};
  const pr = {number: 123, user, state: 'open', draft: true, head: {repo: fork, ref: p.BRANCH, sha: head}, base: {repo: upstream, ref: 'main'}};
  const event = {action: 'synchronize', number: 123, repository: upstream, sender: user, pull_request: pr};
  const eventFile = path.join(root, 'event.json');
  fs.writeFileSync(eventFile, JSON.stringify(event));
  Object.assign(process.env, {GITHUB_WORKSPACE: workspace, GITHUB_EVENT_PATH: eventFile, GITHUB_EVENT_NAME: 'pull_request', GITHUB_REPOSITORY: p.REPOSITORY, GITHUB_REPOSITORY_ID: String(p.REPOSITORY_ID), GITHUB_ACTOR: 'ptdbot', GITHUB_TRIGGERING_ACTOR: 'd4l3k', GITHUB_REF: 'refs/pull/123/merge', GITHUB_HEAD_REF: p.BRANCH, GITHUB_BASE_REF: 'main', GITHUB_SHA: 'b'.repeat(40), GITHUB_RUN_ID: '100', RUNNER_NAME: 'unit-runner'});
  const verifySource = async (run, ctx) => { a.validateSelectedWorkflow(run, ctx); assert.equal(ctx.head, head); };
  for (const [name, attempt, phase] of [[a.PRODUCER_JOB, 2, 'produce'], [a.CONSUMER_JOB, 2, 'hold'], [a.CONSUMER_JOB, 3, 'consume']]) {
    Object.assign(process.env, {GITHUB_JOB: name, GITHUB_RUN_ATTEMPT: String(attempt)});
    const data = metadata({...metadata().ctx, head});
    const run = {...(attempt === 3 ? data.currentRun : data.run), run_attempt: attempt, status: 'in_progress', conclusion: null};
    const current = phase === 'produce' ? data.producer : phase === 'hold' ? data.hold : data.current;
    const workflow = {id: 456, path: p.WORKFLOW, name: a.WORKFLOW_NAME};
    const request = async endpoint => endpoint.endsWith(`/attempts/${attempt}`) ? run : endpoint.endsWith('/pulls/123') ? pr : endpoint.endsWith(`/attempts/${attempt}/jobs?per_page=100`) ? (attempt === 2 ? priorJobs(data) : currentJobs(data)) : endpoint.endsWith('/workflows/456') ? workflow : requestFixture(data)(endpoint);
    const result = await a.guard(CONFIG, false, request, verifySource);
    assert.equal(result.phase, phase);
    for (const [object, key, value] of [[run, 'workflow_id', 999], [current, 'run_id', 999], [current, 'runner_name', 'foreign-runner'], [workflow, 'id', 999]]) {
      const original = object[key]; object[key] = value;
      await assert.rejects(a.guard(CONFIG, false, request, verifySource)); object[key] = original;
    }
    const oldHead = pr.head.sha; pr.head.sha = 'c'.repeat(40);
    await assert.rejects(a.guard(CONFIG, false, request, verifySource)); pr.head.sha = oldHead;
  }
  process.env.GITHUB_RUN_ATTEMPT = '4';
  await assert.rejects(a.guard(CONFIG, false, async () => { throw new Error('Must not request'); }), /Unexpected job\/attempt tuple/);
}));

test('real private namespace has only loopback and rejects unapproved CONNECT', async () => fixture(async root => {
  const python = execFileSync('uv', ['python', 'find', '--managed-python', '3.13.13'], {encoding: 'utf8'}).trim();
  const script = path.join(root, 'network-control.py');
  fs.writeFileSync(path.join(root, 'outer-netns'), fs.readlinkSync('/proc/self/ns/net'));
  fs.writeFileSync(script, `import os, pathlib, socket, urllib.request\nr=pathlib.Path(${JSON.stringify(root)})\nif os.readlink('/proc/self/ns/net') == (r/'outer-netns').read_text(): raise RuntimeError('same netns')\nif {name for _, name in socket.if_nameindex()} != {'lo'}: raise RuntimeError('unexpected network interface')\ntry:\n urllib.request.urlopen('https://forbidden.invalid/', timeout=3)\nexcept OSError:\n (r/'network-control-ok').write_text('private namespace and denied CONNECT\\n')\nelse: raise RuntimeError('unexpected connection')\n`);
  const observations = await tunnel.runIsolated(root, 'example.invalid', python, script, root, 'inert-unit-token');
  assert.deepEqual(observations, {admission_denied: 1, dns_lookups: 0, dials: 0});
  assert.equal(fs.readFileSync(path.join(root, 'network-control-ok'), 'utf8'), 'private namespace and denied CONNECT\n');
  assert.equal(fs.existsSync(path.join(root, 'https-relay.sock')), false);
}));

test('unchanged Python helpers import and reject wrong torchci attempt without network', async () => fixture(root => {
  const workspace = execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim();
  const python = path.join(workspace, 'agent_space', 'consumer-venv', 'bin', 'python');
  assert(fs.existsSync(python), 'Provision agent_space/consumer-venv from the exact requirements.txt first');
  const source = path.join(root, 'source');
  for (const name of ['tools/stats/upload_stats_lib.py', 'tools/stats/upload_artifacts.py']) {
    const file = path.join(source, name);
    fs.mkdirSync(path.dirname(file), {recursive: true});
    fs.writeFileSync(file, execFileSync('git', ['show', `${a.SOURCE_COMMIT}:${name}`]));
  }
  const zip = path.join(root, 'ordinary.zip');
  fs.writeFileSync(zip, p.archive('ordinary'));
  const script = `import contextlib,hashlib,importlib.util,io,pathlib,sys\nroot=pathlib.Path(sys.argv[1]);source=root/'source';sys.path.insert(0,str(source))\nfrom tools.stats import upload_artifacts as opposing\nfrom tools.stats import upload_stats_lib as library\nspec=importlib.util.spec_from_file_location('audit_consumer',sys.argv[2]);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)\nfor name,expected in audit.SOURCE_HASHES.items():\n if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected: raise RuntimeError('source mismatch')\nb=(root/'ordinary.zip').read_bytes();audit.verify_zip(root/'ordinary.zip',{'size':len(b),'digest':hashlib.sha256(b).hexdigest()})\nseen=[]\ndef observe(frame,event,arg):\n if event=='call' and frame.f_code in (opposing.upload_torchci_reports.__code__,library.upload_file_to_s3.__code__,library.unzip.__code__): seen.append(frame.f_code.co_name)\nlocal=root/'torchci-reports-runattempt2-200.zip';local.write_bytes(b)\nsys.setprofile(observe)\ntry: opposing.upload_torchci_reports('pytorch/pytorch',100,3,local)\nfinally: sys.setprofile(None)\nif seen!=['upload_torchci_reports'] or local.read_bytes()!=b: raise RuntimeError('wrong opposing path')\nblocked=[];sys.setprofile(audit.observe_requests(library._github_get.__code__,'allowed-list','allowed-download',99,[],(library.unzip.__code__,library.upload_file_to_s3.__code__),blocked))\ntry: opposing.upload_torchci_reports('pytorch/pytorch',100,3,local)\nfinally: sys.setprofile(None)\nif blocked or local.read_bytes()!=b: raise RuntimeError('production observer rejected wrong-attempt control')\nblocked=[];sys.setprofile(audit.observe_requests(library._github_get.__code__,'allowed-list','allowed-download',99,[],(library.unzip.__code__,library.upload_file_to_s3.__code__),blocked))\ntry:\n with contextlib.redirect_stdout(io.StringIO()): opposing.upload_torchci_reports('pytorch/pytorch',100,2,local)\nfinally: sys.setprofile(None)\nif blocked!=['unzip'] or local.read_bytes()!=b: raise RuntimeError('production observer missed matching-attempt extraction')\nobserved=[];sys.setprofile(audit.observe_requests(library._github_get.__code__,'allowed-list','allowed-download',99,observed))\ntry:\n try: library._github_get('denied-before-http')\n except AssertionError as e:\n  if str(e)!='Unexpected original helper URL': raise\n else: raise RuntimeError('URL guard did not reject')\nfinally: sys.setprofile(None)\nif observed: raise RuntimeError('unexpected download observation')\nprint('unchanged-source opposing control passed')\n`;
  const env = {PATH: '/usr/local/bin:/usr/bin:/bin', HOME: root, AWS_EC2_METADATA_DISABLED: 'true', AWS_CONFIG_FILE: '/dev/null', AWS_SHARED_CREDENTIALS_FILE: '/dev/null', PYTHONDONTWRITEBYTECODE: '1'};
  const result = execFileSync('unshare', ['--user', '--map-root-user', '--net', '--', python, '-I', '-B', '-c', script, root, path.join(__dirname, 'consume.py')], {env, encoding: 'utf8', timeout: 30000});
  assert.equal(result.trim(), 'unchanged-source opposing control passed');
}));

test('selected merge authenticates real Git workflow, approval and entire action trees', async () => fixture(async root => {
  const workspace = execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim();
  const head = execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  const source = path.join(root, 'immutable-source');
  const env = {...process.env, GIT_NO_LAZY_FETCH: '1', GIT_NO_REPLACE_OBJECTS: '1', GIT_ALLOW_PROTOCOL: 'file'};
  execFileSync('git', ['clone', '--shared', '--no-checkout', workspace, source], {env, stdio: 'pipe'});
  execFileSync('git', ['config', 'core.sparseCheckout', 'true'], {cwd: source, env});
  fs.writeFileSync(path.join(source, '.git', 'info', 'sparse-checkout'), `/${a.ACTION_DIRECTORY}/\n/${p.WORKFLOW}\n/${a.CHECK_CI}\n`);
  execFileSync('git', ['checkout', '--detach', head], {cwd: source, env, stdio: 'pipe'});
  const identity = a.localSource(source);
  const data = metadata({...metadata().ctx, head});
  const rootTree = execFileSync('git', ['rev-parse', `${head}^{tree}`], {encoding: 'utf8'}).trim();
  const commit = {sha: data.ctx.merge_sha, parents: [{sha: 'c'.repeat(40)}, {sha: head}], commit: {tree: {sha: rootTree}}};
  const request = async endpoint => {
    if (endpoint.endsWith(`/commits/${data.ctx.merge_sha}`)) return commit;
    const match = /\/git\/trees\/([0-9a-f]{40})$/.exec(endpoint);
    assert(match, 'Unexpected source unit request');
    const text = execFileSync('git', ['ls-tree', match[1]], {encoding: 'utf8'}).trim();
    return {sha: match[1], truncated: false, tree: text.split('\n').map(line => { const [header, name] = line.split('\t'); const [mode, type, sha] = header.split(' '); return {mode, type, sha, path: name}; })};
  };
  await a.selectedSource(data.run, data.ctx, request, identity);
  for (const key of ['head', 'workflow', 'check_ci', 'action']) await assert.rejects(a.selectedSource(data.run, data.ctx, request, {...identity, [key]: 'd'.repeat(40)}));
  const old = commit.parents[1].sha; commit.parents[1].sha = 'e'.repeat(40);
  await assert.rejects(a.selectedSource(data.run, data.ctx, request, identity), /different head parent/); commit.parents[1].sha = old;
  const wrong = structuredClone(data.run); wrong.referenced_workflows[0].ref = 'refs/heads/main';
  await assert.rejects(a.selectedSource(wrong, data.ctx, request, identity));
  fs.appendFileSync(path.join(source, a.ACTION_DIRECTORY, 'README.md'), '\nunit source mismatch\n');
  assert.throws(() => a.localSource(source), /Working action source changed/);
}));

function genuineJobs() {
  const fixed = structuredClone(require('./jobs.test.json'));
  const run = fixed.current_run;
  const current = fixed.current_jobs.jobs.find(item => item.name === a.CONSUMER_JOB);
  const ctx = {run_id: run.id, attempt: run.run_attempt, producer_attempt: 2, workflow_id: run.workflow_id, job_id: current.id, job_name: a.CONSUMER_JOB, head: run.head_sha, merge_sha: run.referenced_workflows[0].sha, pr: 205523, phase: 'consume', current_run_started_at: run.run_started_at};
  const config = a.configuration();
  a.validateSourceRun(config, ctx, fixed.source_run);
  const source = a.validateSourceJobs(fixed.source_jobs, {...ctx, attempt: 2});
  Object.assign(ctx, {producer_job_id: source.producer.id, hold_job_id: source.hold.id, check_ci_job_id: source.check.id, source_jobs_digest: a.sourceFingerprint(source)});
  return {...fixed, ctx, config, source, current};
}

function genuineRequest(data) {
  const artifact = {id: 99, name: p.artifactName({...data.ctx, artifact_attempt: 2}, 'ordinary'), digest: `sha256:${p.digest(p.archive('ordinary'))}`, size_in_bytes: p.archive('ordinary').length, expired: false, workflow_run: {id: data.ctx.run_id, repository_id: p.REPOSITORY_ID, head_repository_id: p.HEAD_REPOSITORY_ID, head_branch: p.BRANCH, head_sha: data.ctx.head}};
  return async endpoint => {
    if (endpoint.endsWith('/attempts/2')) return data.source_run;
    if (endpoint.endsWith('/attempts/3')) return data.current_run;
    if (endpoint.endsWith('/attempts/2/jobs?per_page=100')) return data.source_jobs;
    if (endpoint.endsWith('/attempts/3/jobs?per_page=100')) return data.current_jobs;
    if (endpoint.endsWith('/artifacts?per_page=100')) return {total_count: 1, artifacts: [artifact]};
    if (endpoint.endsWith('/artifacts/99')) return artifact;
    throw new Error('Unexpected genuine-regression request');
  };
}

test('genuine preserved jobs accept exact copies, originals, omissions and mixed views', async () => {
  const data = genuineJobs();
  assert.equal(data.ctx.source_jobs_digest, a.sourceFingerprint(a.rerunTarget(data.config, data.ctx, data.source_run, data.source_jobs).source_jobs));
  for (const oldCheck of [false, true]) for (const oldProducer of [false, true]) {
    const jobs = data.current_jobs.jobs.map(item => item.name === a.CHECK_CI_JOB && oldCheck ? data.source.check : item.name === a.PRODUCER_JOB && oldProducer ? data.source.producer : item);
    assert.equal(a.validateCurrentJobs({total_count: 3, jobs}, data.ctx, data.source, data.current_run).id, 135041148089);
    assert.equal((await a.priorArtifact(data.config, data.ctx, genuineRequest({...data, current_jobs: {total_count: 3, jobs}}))).id, 99);
  }
  assert.equal(a.validateCurrentJobs({total_count: 1, jobs: [data.current]}, data.ctx, data.source, data.current_run).id, data.current.id);
  assert.deepEqual([data.ctx.check_ci_job_id, data.ctx.producer_job_id, data.ctx.hold_job_id], [135038722005, 135038833723, 135039409990]);
  const reordered = Object.fromEntries(Object.entries(data.source).reverse().map(([key, item]) => [key, Object.fromEntries(Object.entries(item).reverse())]));
  assert.equal(a.sourceFingerprint(reordered), data.ctx.source_jobs_digest);
});

test('each copied prerequisite rejects every noncanonical or newly executed shape', () => {
  const baseline = genuineJobs();
  const sourceIds = Object.values(baseline.source).map(item => item.id);
  const changes = [
    ...[0, '135041149791', Number.MAX_SAFE_INTEGER + 1, ...sourceIds, baseline.current.id].map(value => ['id', value]),
    ['run_attempt', 2], ['run_attempt', 4], ['steps', undefined], ['steps', null], ['steps', [{}]],
    ['runner_group_id', 0], ['runner_group_id', undefined], ['runner_group_id', 'null'],
    ['runner_id', 7], ['runner_name', 'other'], ['runner_group_name', 'other'], ['labels', ['ubuntu-24.04', 'extra']],
    ['head_sha', 'f'.repeat(40)], ['head_branch', 'other'], ['run_id', 7], ['run_url', 'https://api.github.com/other'], ['workflow_name', 'other'], ['name', 'other'], ['status', 'in_progress'], ['conclusion', 'failure'],
    ['started_at', undefined], ['started_at', null], ['started_at', 'invalid'], ['started_at', '2026-11-18T18:39:11Z'],
    ['completed_at', undefined], ['completed_at', null], ['completed_at', 'invalid'], ['completed_at', '2026-11-18T18:39:11Z'],
    ['created_at', undefined], ['created_at', null], ['created_at', 'invalid'], ['created_at', '2026-02-30T18:39:15Z'], ['created_at', '2026-11-18T18:39:10Z'],
    ['url', 'https://api.github.com/repos/pytorch/pytorch/actions/jobs/7'], ['html_url', 'https://github.com/pytorch/pytorch/actions/runs/7/job/7'], ['check_run_url', 'https://api.github.com/repos/pytorch/pytorch/check-runs/7'],
    ['node_id', undefined], ['node_id', null], ['node_id', ''], ['node_id', baseline.current.node_id], ...Object.values(baseline.source).map(item => ['node_id', item.node_id]),
    ['unrecognized_field', 'different']
  ];
  for (const name of [a.CHECK_CI_JOB, a.PRODUCER_JOB]) for (const [key, value] of changes) {
    const data = genuineJobs();
    const item = data.current_jobs.jobs.find(job => job.name === name);
    if (value === undefined) delete item[key]; else item[key] = value;
    assert.throws(() => a.validateCurrentJobs(data.current_jobs, data.ctx, data.source, data.current_run), `${name}: ${key}=${JSON.stringify(value)}`);
  }
  const data = genuineJobs();
  const producer = data.current_jobs.jobs.find(item => item.name === a.PRODUCER_JOB);
  Object.assign(producer, {started_at: '2026-11-18T18:39:16Z', completed_at: '2026-11-18T18:40:00Z', runner_id: 7, runner_name: 'new runner', runner_group_id: 0, steps: data.source.producer.steps});
  assert.throws(() => a.validateCurrentJobs(data.current_jobs, data.ctx, data.source, data.current_run));
});

test('copied chronology requires valid original dates and exact current-attempt backlink', () => {
  for (const name of [a.CHECK_CI_JOB, a.PRODUCER_JOB]) {
    for (const [key, value] of [['started_at', undefined], ['started_at', null], ['started_at', 'invalid'], ['started_at', '2026-11-18T18:39:10Z'], ['completed_at', undefined], ['completed_at', null], ['completed_at', 'invalid'], ['completed_at', '2026-11-18T18:39:11Z'], ['completed_at', '2026-11-18T18:39:12Z'], ['completed_at', '2026-11-18T18:30:00Z'], ['url', 'other'], ['html_url', 'other'], ['check_run_url', 'other'], ['run_url', 'other']]) {
      const data = genuineJobs();
      for (const item of [data.source_jobs.jobs.find(job => job.name === name), data.current_jobs.jobs.find(job => job.name === name)]) if (value === undefined) delete item[key]; else item[key] = value;
      assert.throws(() => a.validateCurrentJobs(data.current_jobs, data.ctx, data.source, data.current_run), `${name}: ${key}`);
    }
  }
  for (const [key, value] of [['previous_attempt_url', undefined], ['previous_attempt_url', null], ['previous_attempt_url', 'https://api.github.com/repos/pytorch/pytorch/actions/runs/44370897732/attempts/1'], ['run_started_at', undefined], ['run_started_at', null], ['run_started_at', 'invalid'], ['run_started_at', '2026-11-18T18:35:30Z'], ['run_started_at', '2026-11-18T18:39:16Z'], ['id', 7], ['run_attempt', 4], ['workflow_id', 7], ['path', 'other'], ['head_sha', 'f'.repeat(40)], ['event', 'push'], ['head_branch', 'other']]) {
    const data = genuineJobs();
    if (value === undefined) delete data.current_run[key]; else data.current_run[key] = value;
    assert.throws(() => a.validateCurrentJobs(data.current_jobs, data.ctx, data.source, data.current_run), key);
  }
});

test('consumer rechecks cannot reset changed full original evidence', async () => fixture(async () => {
  for (const name of [a.CHECK_CI_JOB, a.PRODUCER_JOB, a.CONSUMER_JOB]) {
    const data = genuineJobs();
    data.source_jobs.jobs.find(item => item.name === name).unrecognized_field = 'added to source';
    const copy = data.current_jobs.jobs.find(item => item.name === name);
    copy.unrecognized_field = 'added to source';
    await assert.rejects(a.priorArtifact(data.config, data.ctx, genuineRequest(data)), /Source job evidence changed/);
  }
  const data = genuineJobs();
  await a.execute('guard', data.config, data.ctx);
  await assert.rejects(a.execute('consume', data.config, {...data.ctx, source_jobs_digest: 'f'.repeat(64)}));
  await a.execute('finish', data.config, data.ctx);
  for (const name of [a.CHECK_CI_JOB, a.PRODUCER_JOB, a.CONSUMER_JOB]) {
    const changed = genuineJobs(); changed.source_jobs.jobs = changed.source_jobs.jobs.filter(item => item.name !== name); changed.source_jobs.total_count--;
    await assert.rejects(a.priorArtifact(changed.config, changed.ctx, genuineRequest(changed)));
  }
  const changed = genuineJobs(); changed.current_run.run_started_at = '2026-11-18T18:39:12Z';
  await assert.rejects(a.priorArtifact(changed.config, changed.ctx, genuineRequest(changed)), /Current run start changed/);
}));

test('copy support does not weaken current-consumer uniqueness or role checks', () => {
  for (const mutate of [
    d => { d.current.runner_group_id = null; }, d => { d.current.runner_name = ''; }, d => { d.current.id = d.source.producer.id; }, d => { d.current.name = a.PRODUCER_JOB; },
    d => { d.current_jobs.jobs.push({...d.current, id: 7}); d.current_jobs.total_count++; },
    d => { d.current_jobs.jobs.find(item => item.name === a.CHECK_CI_JOB).id = d.current.id; },
    d => { d.current_jobs.jobs.push({...d.current, id: 7, name: 'extra'}); d.current_jobs.total_count++; },
    d => { d.current_jobs.jobs.find(item => item.name === a.PRODUCER_JOB).node_id = d.current_jobs.jobs.find(item => item.name === a.CHECK_CI_JOB).node_id; }
  ]) {
    const data = genuineJobs(); mutate(data);
    assert.throws(() => a.validateCurrentJobs(data.current_jobs, data.ctx, data.source, data.current_run));
  }
});
