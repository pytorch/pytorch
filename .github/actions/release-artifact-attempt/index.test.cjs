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
const context = {run_id: 100, attempt: 2, producer_attempt: 2, workflow_id: 456, job_id: 200, head: 'a'.repeat(40), merge_sha: 'b'.repeat(40), pr: 123, actor: 'ptdbot', triggering_actor: 'd4l3k', phase: 'producer', repository: p.REPOSITORY};

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
  assert.equal(yaml.match(/pull_request\.number == 205523/g).length, 2);
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
  const steps = workflow.jobs[a.JOB].steps;
  const sdk = steps.find(step => step.with?.repository === 'actions/download-artifact');
  assert.equal(sdk.with.ref, p.SDK_COMMIT);
  assert.equal(sdk.with.path, p.SDK_DIRECTORY);
  assert.equal(steps.find(step => step.name === 'Install locked artifact runtime')['working-directory'], sdk.with.path);
  const source = steps.find(step => step.with?.repository === p.REPOSITORY);
  assert.equal(source.with.ref, a.SOURCE_COMMIT);
  assert.equal(source.with.path, a.SOURCE_DIRECTORY);
  for (const mode of ['dependencies', 'produce']) assert.equal(steps.find(step => step.with?.mode === mode).if, "steps.guard.outputs.phase == 'producer'");
  assert.equal(steps.find(step => step.with?.mode === 'consume').if, "steps.guard.outputs.phase == 'consumer'");
  for (const step of steps.filter(step => step.with?.mode)) assert.equal(step.uses, './.github/actions/release-artifact-attempt');
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

function metadata(ctx = {...context, attempt: 3, phase: 'consumer'}) {
  const actor = {login: 'ptdbot', id: CONFIG.actor_ids.ptdbot};
  const run = {id: ctx.run_id, run_attempt: 2, workflow_id: 456, path: p.WORKFLOW, event: 'pull_request', head_sha: ctx.head, head_branch: p.BRANCH, repository: {id: p.REPOSITORY_ID, full_name: p.REPOSITORY}, head_repository: {id: p.HEAD_REPOSITORY_ID, full_name: p.HEAD_REPOSITORY}, status: 'completed', conclusion: 'success', actor, triggering_actor: {login: 'd4l3k', id: CONFIG.actor_ids.d4l3k}};
  const job = {id: 200, run_id: ctx.run_id, run_attempt: 2, head_sha: ctx.head, workflow_name: a.WORKFLOW_NAME, name: a.JOB, status: 'completed', conclusion: 'success'};
  const artifact = {id: 99, name: p.artifactName({...ctx, artifact_attempt: 2}, 'ordinary'), digest: `sha256:${p.digest(p.archive('ordinary'))}`, size_in_bytes: p.archive('ordinary').length, expired: false, workflow_run: {id: ctx.run_id, repository_id: p.REPOSITORY_ID, head_repository_id: p.HEAD_REPOSITORY_ID, head_branch: p.BRANCH, head_sha: ctx.head}};
  return {ctx, run, job, artifact};
}

function requestFixture(data) {
  return async endpoint => {
    if (endpoint.endsWith('/attempts/2')) return data.run;
    if (endpoint.endsWith('/attempts/2/jobs?per_page=100')) return {total_count: 1, jobs: [data.job]};
    if (endpoint.endsWith('/artifacts?per_page=100')) return {total_count: data.extra ? 2 : 1, artifacts: data.extra ? [data.artifact, data.extra] : [data.artifact]};
    if (endpoint.endsWith(`/artifacts/${data.artifact.id}`)) return data.artifact;
    throw new Error('Unexpected unit request');
  };
}

test('N+1 authenticates same-run prior attempt and deterministic bytes', async () => {
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
  for (const [key, value] of [['id', 0], ['name', 'test-reports-runattempt2-x'], ['digest', `sha256:${'0'.repeat(64)}`], ['size_in_bytes', 1], ['expired', true]]) {
    const changed = structuredClone(data); changed.artifact[key] = value;
    await assert.rejects(a.priorArtifact(CONFIG, changed.ctx, requestFixture(changed)));
  }
  for (const key of ['id', 'repository_id', 'head_repository_id', 'head_branch', 'head_sha']) {
    const changed = structuredClone(data); changed.artifact.workflow_run[key] = 'wrong';
    await assert.rejects(a.priorArtifact(CONFIG, changed.ctx, requestFixture(changed)));
  }
});

test('wrong source attempt, run, head and workflow reject', async () => {
  for (const [key, value] of [['id', 101], ['run_attempt', 1], ['workflow_id', 457], ['path', 'other'], ['event', 'push'], ['head_sha', 'c'.repeat(40)], ['head_branch', 'other'], ['status', 'in_progress'], ['conclusion', 'failure']]) {
    const changed = metadata(); changed.run[key] = value;
    await assert.rejects(a.priorArtifact(CONFIG, changed.ctx, requestFixture(changed)));
  }
  const data = metadata();
  await assert.rejects(a.priorArtifact(CONFIG, {...data.ctx, attempt: 4}, requestFixture(data)));
});

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

test('runtime guard binds the owned draft, actors, workflow, job and N/N+1', async () => fixture(async root => {
  const workspace = execFileSync('git', ['rev-parse', '--show-toplevel'], {encoding: 'utf8'}).trim();
  const head = execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8'}).trim();
  const user = {login: 'ptdbot', id: CONFIG.actor_ids.ptdbot};
  const upstream = {id: p.REPOSITORY_ID, full_name: p.REPOSITORY};
  const fork = {id: p.HEAD_REPOSITORY_ID, full_name: p.HEAD_REPOSITORY};
  const pr = {number: 123, user, state: 'open', draft: true, head: {repo: fork, ref: p.BRANCH, sha: head}, base: {repo: upstream, ref: 'main'}};
  const event = {action: 'synchronize', number: 123, repository: upstream, sender: user, pull_request: pr};
  const eventFile = path.join(root, 'event.json');
  fs.writeFileSync(eventFile, JSON.stringify(event));
  Object.assign(process.env, {GITHUB_WORKSPACE: workspace, GITHUB_EVENT_PATH: eventFile, GITHUB_JOB: a.JOB, GITHUB_EVENT_NAME: 'pull_request', GITHUB_REPOSITORY: p.REPOSITORY, GITHUB_REPOSITORY_ID: String(p.REPOSITORY_ID), GITHUB_ACTOR: 'ptdbot', GITHUB_TRIGGERING_ACTOR: 'd4l3k', GITHUB_REF: 'refs/pull/123/merge', GITHUB_HEAD_REF: p.BRANCH, GITHUB_BASE_REF: 'main', GITHUB_SHA: 'b'.repeat(40), GITHUB_RUN_ID: '100', RUNNER_NAME: 'unit-runner'});
  for (const attempt of [2, 3]) {
    process.env.GITHUB_RUN_ATTEMPT = String(attempt);
    const run = {id: 100, run_attempt: attempt, workflow_id: 456, event: 'pull_request', path: p.WORKFLOW, head_sha: head, head_branch: p.BRANCH, repository: upstream, head_repository: fork, actor: user, triggering_actor: {login: 'd4l3k', id: CONFIG.actor_ids.d4l3k}};
    const job = {id: 200 + attempt, run_id: 100, run_attempt: attempt, head_sha: head, workflow_name: a.WORKFLOW_NAME, name: a.JOB, runner_name: 'unit-runner'};
    const workflow = {id: 456, path: p.WORKFLOW, name: a.WORKFLOW_NAME};
    const request = async endpoint => endpoint.endsWith(`/attempts/${attempt}`) ? run : endpoint.endsWith('/pulls/123') ? pr : endpoint.includes('/jobs?') ? {total_count: 1, jobs: [job]} : workflow;
    const result = await a.guard(CONFIG, false, request);
    assert.equal(result.phase, attempt === 2 ? 'producer' : 'consumer');
    for (const object of [run, job, workflow]) {
      const key = object === run ? 'workflow_id' : object === job ? 'run_id' : 'id'; const original = object[key]; object[key] = 999;
      await assert.rejects(a.guard(CONFIG, false, request)); object[key] = original;
    }
    const oldHead = pr.head.sha; pr.head.sha = 'c'.repeat(40);
    await assert.rejects(a.guard(CONFIG, false, request)); pr.head.sha = oldHead;
  }
  process.env.GITHUB_RUN_ATTEMPT = '4';
  await assert.rejects(a.guard(CONFIG, false, async () => { throw new Error('Must not request'); }), /Unexpected attempt/);
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
  const result = execFileSync('unshare', ['--user', '--map-root-user', '--net', '--', python, '-I', '-c', script, root, path.join(__dirname, 'consume.py')], {env, encoding: 'utf8', timeout: 30000});
  assert.equal(result.trim(), 'unchanged-source opposing control passed');
}));
