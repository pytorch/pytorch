'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const canary = require('./protocol.cjs');

function identities(prNumber = 205398) {
  const sha = 'a'.repeat(40);
  const env = {GITHUB_EVENT_NAME: 'pull_request', GITHUB_REPOSITORY: canary.REPOSITORY, GITHUB_REPOSITORY_ID: String(canary.REPOSITORY_ID), GITHUB_ACTOR: 'ptdbot', GITHUB_TRIGGERING_ACTOR: 'ptdbot', GITHUB_REF: `refs/pull/${prNumber}/merge`, GITHUB_HEAD_REF: canary.BRANCH, GITHUB_BASE_REF: 'main', GITHUB_SHA: 'b'.repeat(40), GITHUB_RUN_ID: '123', GITHUB_RUN_ATTEMPT: '1'};
  const pr = {number: prNumber, user: {login: 'ptdbot'}, state: 'open', draft: true, head: {repo: {id: canary.HEAD_REPOSITORY_ID, full_name: canary.HEAD_REPOSITORY}, ref: canary.BRANCH, sha}, base: {repo: {id: canary.REPOSITORY_ID, full_name: canary.REPOSITORY}, ref: 'main'}};
  const event = {action: 'synchronize', number: prNumber, repository: {id: canary.REPOSITORY_ID, full_name: canary.REPOSITORY}, sender: {login: 'ptdbot'}, pull_request: structuredClone(pr)};
  const run = {id: 123, run_attempt: 1, event: 'pull_request', path: canary.WORKFLOW, head_sha: sha, head_branch: canary.BRANCH, repository: {id: canary.REPOSITORY_ID, full_name: canary.REPOSITORY}, head_repository: {id: canary.HEAD_REPOSITORY_ID, full_name: canary.HEAD_REPOSITORY}, actor: {login: 'ptdbot'}, triggering_actor: {login: 'ptdbot'}};
  return {env, event, run, pr};
}

function withFixture(callback) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'artifact-crossrun-unit-'));
  const previous = {RUNNER_TEMP: process.env.RUNNER_TEMP, GITHUB_OUTPUT: process.env.GITHUB_OUTPUT};
  process.env.RUNNER_TEMP = temporary;
  process.env.GITHUB_OUTPUT = path.join(temporary, 'outputs');
  const context = {repository: canary.REPOSITORY, run_id: 123, attempt: 1, head: 'a'.repeat(40), merge_sha: 'b'.repeat(40), pr: 205398, actor: 'ptdbot', triggering_actor: 'ptdbot'};
  canary.initialize(context);
  return Promise.resolve().then(() => callback(context, temporary)).finally(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
    fs.rmSync(temporary, {recursive: true, force: true});
  });
}

test('only the allocated PR producer is active and the default PR0 remains denied', () => {
  assert.equal(canary.EXPECTED_PR, 0);
  const {env, event, run, pr} = identities(205419);
  assert.throws(() => canary.validateIdentity(env, event, run, pr));
  assert.equal(canary.validateIdentity(env, event, run, pr, 205419).pr, 205419);
  assert.throws(() => canary.validateIdentity(env, event, run, pr, 205398));
  const activation = require('./activation.json');
  assert.deepEqual(activation, {phase: 'producer', expected_pr: 205419, actor_ids: {ptdbot: 283013143, d4l3k: 909104}, rest_preflight: {run_id: 44210618534, attempt: 2, head: '4784376f2eeaa58f1927676d2c0be1f53133ef85', job_id: 134488060725, workflow_id: 422034765, pr: 205419, deleted_by: 'd4l3k', delete_status: 204, deleted_get_status: 404, remaining_artifacts: 0, artifacts: {preflight: {id: 14385473342, name: 'release-artifact-crossrun-44210618534-2-preflight', digest: 'b080dcdf4d6082c9d30fe62dcd93984b6b8745bce2f85f89cfd1bc5bd0fafa5f'}}}, producer: null});
  assert.deepEqual(require('./index.cjs').configuration(), activation);
  assert.throws(() => require('./index.cjs').configuration({...activation, phase: 'disabled', expected_pr: 0}));
  assert.throws(() => require('./index.cjs').configuration({...activation, expected_pr: 0}));
  assert.throws(() => require('./index.cjs').configuration({...activation, rest_preflight: null}));
  assert.throws(() => require('./index.cjs').configuration({...activation, phase: 'preflight'}));
  assert.throws(() => require('./index.cjs').configuration({...activation, phase: 'consumer'}));
  const workflow = fs.readFileSync(path.resolve(__dirname, '../../workflows/security-artifact-crossrun.yml'), 'utf8');
  assert.equal(canary.WORKFLOW, '.github/workflows/security-artifact-crossrun.yml');
  assert.equal((workflow.match(/github\.event\.pull_request\.number == 205419\b/g) || []).length, 2);
  assert.doesNotMatch(workflow, /github\.event\.pull_request\.number == 0\b/);
  assert.match(workflow, /uses: \.\/\.github\/workflows\/_check-ci\.yml/);
  assert.match(workflow, /github\.repository_owner == 'pytorch'/);
  assert.doesNotMatch(workflow, /actions: write|workflow_dispatch|pull_request_target|id-token|secrets\.|environment:/);
  assert(workflow.includes(`actions/download-artifact@${require('./index.cjs').LEGACY_COMMIT}`));
  assert(workflow.includes(`actions/download-artifact@${canary.MODERN_COMMIT}`));
  assert.equal((workflow.match(/run-id: \$\{\{ steps.guard.outputs.producer_run_id \}\}/g) || []).length, 6);
});

test('owned identities bind repo, author, branch, run, attempt and exact head', () => {
  const base = identities();
  assert.equal(canary.validateIdentity(base.env, base.event, base.run, base.pr, 205398).run_id, 123);
  for (const [group, route, value] of [
    ['env', ['GITHUB_EVENT_NAME'], 'push'], ['env', ['GITHUB_REPOSITORY'], 'other/pytorch'],
    ['env', ['GITHUB_REPOSITORY_ID'], '1'], ['env', ['GITHUB_ACTOR'], 'd4l3k'], ['env', ['GITHUB_TRIGGERING_ACTOR'], 'other'],
    ['env', ['GITHUB_REF'], 'refs/heads/main'], ['env', ['GITHUB_HEAD_REF'], 'main'], ['env', ['GITHUB_BASE_REF'], 'release/2.15'], ['env', ['GITHUB_SHA'], 'not-a-sha'], ['env', ['GITHUB_RUN_ID'], '124'], ['env', ['GITHUB_RUN_ATTEMPT'], '2'],
    ['event', ['action'], 'closed'], ['event', ['number'], 205399], ['event', ['repository', 'id'], 1], ['event', ['repository', 'full_name'], 'other/pytorch'], ['event', ['sender', 'login'], 'other'],
    ['event', ['pull_request', 'number'], 205399], ['event', ['pull_request', 'user', 'login'], 'other'], ['event', ['pull_request', 'draft'], false], ['event', ['pull_request', 'state'], 'closed'], ['event', ['pull_request', 'head', 'repo', 'id'], 1], ['event', ['pull_request', 'head', 'repo', 'full_name'], 'other/pytorch'], ['event', ['pull_request', 'head', 'ref'], 'main'], ['event', ['pull_request', 'head', 'sha'], 'c'.repeat(40)], ['event', ['pull_request', 'base', 'repo', 'full_name'], 'other/pytorch'], ['event', ['pull_request', 'base', 'ref'], 'release/2.15'],
    ['run', ['id'], 124], ['run', ['run_attempt'], 2], ['run', ['event'], 'push'], ['run', ['path'], 'other.yml'], ['run', ['head_sha'], 'c'.repeat(40)], ['run', ['head_branch'], 'main'], ['run', ['repository', 'id'], 1], ['run', ['head_repository', 'id'], 1], ['run', ['head_repository', 'full_name'], 'other/pytorch'], ['run', ['actor', 'login'], 'other'], ['run', ['triggering_actor', 'login'], 'other'],
    ['pr', ['number'], 205399], ['pr', ['user', 'login'], 'other'], ['pr', ['state'], 'closed'], ['pr', ['draft'], false], ['pr', ['head', 'repo', 'id'], 1], ['pr', ['head', 'repo', 'full_name'], 'other/pytorch'], ['pr', ['head', 'ref'], 'main'], ['pr', ['head', 'sha'], 'c'.repeat(40)], ['pr', ['base', 'repo', 'full_name'], 'other/pytorch'], ['pr', ['base', 'ref'], 'release/2.15']
  ]) {
    const changed = structuredClone(base);
    let target = changed[group];
    for (const key of route.slice(0, -1)) target = target[key];
    target[route.at(-1)] = value;
    assert.throws(() => canary.validateIdentity(changed.env, changed.event, changed.run, changed.pr, 205398), `${group}.${route.join('.')}`);
  }
  const approved = structuredClone(base);
  approved.env.GITHUB_TRIGGERING_ACTOR = 'd4l3k';
  approved.run.triggering_actor.login = 'd4l3k';
  assert.equal(canary.validateIdentity(approved.env, approved.event, approved.run, approved.pr, 205398).head, base.pr.head.sha);
  const advanced = structuredClone(base);
  advanced.pr.head.sha = 'c'.repeat(40);
  advanced.pr.state = 'closed';
  advanced.pr.draft = false;
  assert.equal(canary.validateIdentity(advanced.env, advanced.event, advanced.run, advanced.pr, 205398, true).head, base.pr.head.sha);
});

test('ZIPs are deterministic, fixed-name, regular stored entries', () => {
  for (const [kind, name] of Object.entries(canary.MEMBERS)) {
    const data = canary.archive(kind);
    assert.deepEqual(data, canary.archive(kind));
    assert.equal(data.readUInt32LE(0), 0x04034b50);
    assert.equal(data.readUInt16LE(6), 0);
    assert.equal(data.readUInt16LE(8), 0);
    assert.equal(data.readUInt32LE(14), canary.crc32(canary.MARKER));
    assert.equal(data.readUInt32LE(18), canary.MARKER.length);
    assert.equal(data.readUInt32LE(22), canary.MARKER.length);
    assert.equal(data.subarray(30, 30 + data.readUInt16LE(26)).toString(), name);
    const central = 30 + Buffer.byteLength(name) + canary.MARKER.length;
    assert.equal(data.readUInt32LE(central), 0x02014b50);
    assert.equal(data.readUInt32LE(central + 38), 0);
    assert.equal(data.readUInt32LE(data.length - 22), 0x06054b50);
    assert.equal(data.readUInt16LE(data.length - 14), 1);
  }
  assert.throws(() => canary.archive('arbitrary'));
  assert.equal(canary.crc32(Buffer.from('123456789')), 0xcbf43926);
});

test('hash verification rejects changed dependency bytes', () => withFixture((_context, root) => {
  const file = path.join(root, 'dependency');
  fs.writeFileSync(file, 'known source');
  const hashes = {dependency: canary.digest(Buffer.from('known source'))};
  canary.verifyHashes(root, hashes);
  fs.appendFileSync(file, ' changed');
  assert.throws(() => canary.verifyHashes(root, hashes));
}));

function contractService(failPreflightDelete = false) {
  const ids = {workflowRunBackendId: '11111111-1111-1111-1111-111111111111', workflowJobRunBackendId: '22222222-2222-2222-2222-222222222222'};
  const objects = new Map();
  const calls = [];
  let nextId = 10;
  let uploadedBytes;
  const service = {ids, calls, objects, StringValue: {create: value => value}, expiration: days => ({days}),
    upload: async (name, files, root, options) => {
      assert.equal(files.length, 1);
      assert.deepEqual(fs.readFileSync(files[0]), canary.MARKER);
      assert.equal(path.dirname(files[0]), root);
      assert.deepEqual(options, {retentionDays: 1, compressionLevel: 0});
      const id = ++nextId;
      objects.set(name, {...ids, name, databaseId: String(id)});
      calls.push('standard-upload');
      return {id, digest: canary.digest(canary.MARKER)};
    },
    uploadBlob: async (url, stream) => {
      assert.equal(url, 'unit-test-only');
      const buffers = [];
      for await (const data of stream) buffers.push(data);
      uploadedBytes = Buffer.concat(buffers);
      calls.push('blob');
      return {uploadSize: uploadedBytes.length, sha256Hash: canary.digest(uploadedBytes)};
    },
    client: {
      ListArtifacts: async request => {
        assert.equal(request.workflowRunBackendId, ids.workflowRunBackendId);
        assert.equal(request.workflowJobRunBackendId, ids.workflowJobRunBackendId);
        const object = objects.get(request.nameFilter.value);
        return {artifacts: object ? [object] : []};
      },
      DeleteArtifact: async request => {
        if (failPreflightDelete && request.name.endsWith('-preflight')) throw new Error('unit preflight denial');
        const object = objects.get(request.name);
        assert(object);
        objects.delete(request.name);
        calls.push('delete');
        return {ok: true, artifactId: object.databaseId};
      },
      CreateArtifact: async request => {
        assert.equal(request.version, 4);
        assert.deepEqual(request.expiresAt, {days: 1});
        assert(!objects.has(request.name));
        assert(![...objects.keys()].some(name => name.endsWith('-preflight')));
        calls.push('create');
        return {ok: true, signedUploadUrl: 'unit-test-only'};
      },
      FinalizeArtifact: async request => {
        assert.equal(request.size, String(uploadedBytes.length));
        assert.equal(request.hash.value, `sha256:${canary.digest(uploadedBytes)}`);
        const id = ++nextId;
        objects.set(request.name, {...ids, name: request.name, databaseId: String(id)});
        calls.push('finalize');
        return {ok: true, artifactId: String(id)};
      }
    }
  };
  return service;
}

function pollClock(onWait = () => {}) {
  let elapsed = 0;
  const waits = [];
  return {
    waits,
    now: () => elapsed,
    advance: milliseconds => { elapsed += milliseconds; },
    wait: async milliseconds => {
      assert(milliseconds > 0 && milliseconds <= canary.DELETE_VISIBILITY_INTERVAL_MS);
      waits.push(milliseconds);
      elapsed += milliseconds;
      await onWait();
    }
  };
}

test('offline protocol contract requires genuine-deletion stage before raw upload', () => withFixture(async context => {
  const service = contractService();
  await canary.produce(context, service);
  assert.deepEqual(service.calls, ['standard-upload', 'delete', 'create', 'blob', 'finalize', 'create', 'blob', 'finalize', 'create', 'blob', 'finalize']);
  assert.equal(canary.readState(context).preflight_deleted, true);
  assert.equal(service.objects.size, 3);
  assert.deepEqual(canary.readProtocolObservations(context).filter(item => item.kind === 'preflight').map(item => item.stage), ['upload-request', 'upload-response', 'list-request', 'list-response', 'list-request', 'list-response', 'delete-request', 'delete-response', 'list-request', 'list-response', 'delete-confirmed']);
  await canary.cleanup(context, service);
  assert.equal(service.objects.size, 0);
  assert(Object.values(canary.readState(context).artifacts).every(item => item.phase === 'deleted'));
}));

test('offline deletion denial prevents all malformed uploads', () => withFixture(async context => {
  const service = contractService(true);
  await assert.rejects(canary.produce(context, service), /preflight denial/);
  assert.deepEqual(service.calls, ['standard-upload']);
  assert.equal(canary.readState(context).preflight_deleted, false);
}));

test('negative deletion response stops preflight before crafted upload', () => withFixture(async context => {
  const service = contractService();
  const clock = pollClock();
  service.client.DeleteArtifact = async () => {
    service.calls.push('rejected-delete');
    return {ok: false, artifactId: '0', signedUploadUrl: 'unit-only-private-value', detail: 'unit-only-private-value'};
  };
  await assert.rejects(canary.produce(context, service, clock), /did not confirm deletion/);
  assert.deepEqual(clock.waits, []);
  assert.deepEqual(service.calls, ['standard-upload', 'rejected-delete']);
  const state = canary.readState(context);
  assert.equal(state.preflight_deleted, false);
  assert.equal(state.artifacts.preflight.phase, 'finalized');
  assert.equal(service.objects.size, 1);
  const observations = canary.readProtocolObservations(context);
  assert.deepEqual(observations.at(-1), {ok: false, artifact_id: 0, operation: 'produce', kind: 'preflight', stage: 'delete-response'});
  assert(!JSON.stringify(observations).includes('unit-only-private-value'));
}));

test('bounded delayed absence preserves preflight ordering and diagnostic limits', () => withFixture(async context => {
  const service = contractService();
  let pending;
  service.client.DeleteArtifact = async request => {
    const object = service.objects.get(request.name);
    assert(object);
    assert.equal(pending, undefined);
    pending = {name: request.name, waits: 0};
    service.calls.push('delete');
    return {ok: true, artifactId: object.databaseId};
  };
  const clock = pollClock(() => {
    assert(pending);
    if (pending.name.endsWith('-preflight')) assert(!service.calls.includes('create'));
    if (++pending.waits !== canary.DELETE_VISIBILITY_MAX_POLLS - 1) return;
    service.objects.delete(pending.name);
    pending = undefined;
  });
  await canary.produce(context, service, clock);
  assert.equal(canary.readState(context).preflight_deleted, true);
  assert.equal(service.objects.size, 3);
  await canary.cleanup(context, service, clock);
  assert.equal(service.objects.size, 0);
  assert.equal(clock.waits.length, 4 * (canary.DELETE_VISIBILITY_MAX_POLLS - 1));
  assert.equal(canary.readProtocolObservations(context).length, 94);
}));

test('persistent post-delete presence times out before any crafted upload', () => withFixture(async context => {
  const service = contractService();
  const clock = pollClock();
  service.client.DeleteArtifact = async request => {
    service.calls.push('delete');
    return {ok: true, artifactId: service.objects.get(request.name).databaseId};
  };
  await assert.rejects(canary.produce(context, service, clock), /deletion visibility deadline/);
  assert.equal(clock.waits.length, canary.DELETE_VISIBILITY_MAX_POLLS - 1);
  assert(clock.now() <= canary.DELETE_VISIBILITY_TIMEOUT_MS);
  assert.deepEqual(service.calls, ['standard-upload', 'delete']);
  assert.equal(canary.readState(context).preflight_deleted, false);
  assert.equal(service.objects.size, 1);
  assert.deepEqual(canary.readProtocolObservations(context).at(-1), {polls: canary.DELETE_VISIBILITY_MAX_POLLS, reason: 'attempts', operation: 'produce', kind: 'preflight', stage: 'delete-timeout'});
}));

test('post-delete identity substitution is rejected without waiting', async () => {
  for (const [field, value] of [['databaseId', '124'], ['name', 'other'], ['workflowRunBackendId', '33333333-3333-3333-3333-333333333333'], ['workflowJobRunBackendId', '44444444-4444-4444-4444-444444444444']]) {
    const service = contractService();
    const clock = pollClock();
    const object = {...service.ids, name: 'owned', databaseId: '123'};
    const original = object[field];
    service.objects.set('owned', object);
    service.client.DeleteArtifact = async () => {
      object[field] = value;
      service.calls.push('delete');
      return {ok: true, artifactId: '123'};
    };
    await assert.rejects(canary.removeArtifact(service, {name: 'owned', phase: 'finalized', id: 123}, true, () => {}, clock), {code: 'ERR_ASSERTION', operator: 'strictEqual', actual: field === 'databaseId' ? Number(value) : value, expected: field === 'databaseId' ? Number(original) : original});
    assert.deepEqual(clock.waits, []);
    assert.deepEqual(service.calls, ['delete']);
  }
});

test('an absence response after the visibility deadline cannot pass preflight', () => withFixture(async context => {
  const service = contractService();
  const clock = pollClock();
  const list = service.client.ListArtifacts;
  service.client.ListArtifacts = async request => {
    if (service.calls.includes('delete')) clock.advance(canary.DELETE_VISIBILITY_TIMEOUT_MS + 1);
    return list(request);
  };
  await assert.rejects(canary.produce(context, service, clock), /deletion visibility deadline/);
  assert.deepEqual(clock.waits, []);
  assert.deepEqual(service.calls, ['standard-upload', 'delete']);
  assert.equal(canary.readState(context).preflight_deleted, false);
  assert.deepEqual(canary.readProtocolObservations(context).at(-1), {polls: 1, reason: 'deadline', operation: 'produce', kind: 'preflight', stage: 'delete-timeout'});
}));

test('empty preflight listing is observed and fails without a speculative retry', () => withFixture(async context => {
  const service = contractService();
  const list = service.client.ListArtifacts;
  let listCalls = 0;
  service.client.ListArtifacts = async request => ++listCalls === 1 ? {artifacts: []} : list(request);
  await assert.rejects(canary.produce(context, service), {code: 'ERR_ASSERTION', actual: 0, expected: 1});
  assert.equal(listCalls, 1);
  assert.deepEqual(service.calls, ['standard-upload']);
  assert.equal(canary.readState(context).preflight_deleted, false);
  const observation = canary.readProtocolObservations(context).at(-1);
  assert.equal(observation.stage, 'list-response');
  assert.equal(observation.boundary, 'preflight-visible');
  assert.equal(observation.count, 0);
  await canary.cleanup(context, service);
  assert.equal(service.objects.size, 0);
  assert.equal(canary.readProtocolObservations(context).at(-1).stage, 'delete-confirmed');
}));

test('protocol observations reject unrecognized or unsafe fields', () => withFixture(context => {
  const observe = canary.protocolObserver(context, 'produce', 'preflight');
  assert.throws(() => observe('unknown-stage'));
  assert.throws(() => observe('upload-request', {raw: 'unit-only-private-value'}));
  assert.throws(() => observe('delete-response', {ok: false, artifact_id: 'unit-only-private-value'}));
  assert.throws(() => observe('delete-response', {ok: 'true', artifact_id: 123}));
  assert.deepEqual(canary.readProtocolObservations(context), []);
  for (const value of ['https://example.test/private', '-1', '9007199254740992', {}, undefined]) assert.equal(canary.observedId(value), null);
  assert.equal(canary.observedId('0'), 0);
  assert.equal(canary.observedId('123'), 123);
  observe('upload-request');
  const file = path.join(canary.fixtureRoot(context), 'protocol-observations.jsonl');
  fs.appendFileSync(file, JSON.stringify({operation: 'produce', kind: 'preflight', stage: 'upload-request', raw: 'unit-only-private-value'}) + '\n');
  assert.throws(() => canary.readProtocolObservations(context));
}));

test('cleanup retains unfinished state after negative or malformed deletion responses', () => withFixture(async context => {
  for (const phase of ['pending', 'created']) {
    for (const response of [{ok: false, artifactId: '0'}, {ok: false, artifactId: '123'}, {ok: true, artifactId: '0'}, {ok: true, artifactId: '-1'}, {ok: true, artifactId: '9007199254740992'}, {artifactId: '123'}, {ok: true}]) {
      const service = contractService();
      service.client.DeleteArtifact = async () => {
        service.calls.push('rejected-delete');
        return response;
      };
      const state = {schema: 1, context, artifacts: {ordinary: {name: canary.artifactName(context, 'ordinary'), phase}}, preflight_deleted: false};
      canary.writeState(context, state);
      await assert.rejects(canary.cleanup(context, service));
      assert.deepEqual(canary.readState(context), state);
      assert.deepEqual(service.calls, ['rejected-delete']);
    }
  }
}));

test('deletion response must match every known or listed artifact ID', async () => {
  for (const knownId of [undefined, 123]) {
    const service = contractService();
    service.objects.set('owned', {...service.ids, name: 'owned', databaseId: '123'});
    service.client.DeleteArtifact = async () => {
      service.objects.delete('owned');
      return {ok: true, artifactId: '124'};
    };
    const artifact = {name: 'owned', phase: knownId === undefined ? 'created' : 'finalized'};
    if (knownId !== undefined) artifact.id = knownId;
    await assert.rejects(canary.removeArtifact(service, artifact), {code: 'ERR_ASSERTION', actual: 124, expected: 123, operator: 'strictEqual'});
    assert.equal(service.objects.size, 0);
  }
});

test('cleanup absence is idempotent but cannot satisfy deletion preflight', async () => {
  const service = contractService();
  const artifact = {name: 'owned', phase: 'finalized', id: 123};
  await canary.removeArtifact(service, artifact);
  await assert.rejects(canary.removeArtifact(service, artifact, true));
  assert.deepEqual(service.calls, []);
});

test('cleanup refuses an artifact associated with another job', async () => {
  const service = contractService();
  service.objects.set('owned', {...service.ids, workflowJobRunBackendId: '33333333-3333-3333-3333-333333333333', name: 'owned', databaseId: '123'});
  await assert.rejects(canary.removeArtifact(service, {name: 'owned', phase: 'finalized', id: 123}));
  assert.deepEqual(service.calls, []);
});

test('unchanged producer, exact FA3 and modern SDKs consume actual ZIP bytes over loopback HTTP', {skip: !process.env.ARTIFACT_ADMISSION_V4 || !process.env.ARTIFACT_CROSSRUN_V417 || !process.env.ARTIFACT_ADMISSION_V7}, async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'artifact-admission-sdk-'));
  let current;
  const server = http.createServer((request, response) => {
    assert.equal(request.url, '/owned.zip');
    response.writeHead(200, {'Content-Type': 'application/zip', 'Content-Length': current.length});
    response.end(current);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    for (const [version, sdkPath, expected] of [['producer-legacy', process.env.ARTIFACT_ADMISSION_V4, '2.3.2'], ['legacy', process.env.ARTIFACT_CROSSRUN_V417, '2.1.7'], ['modern', process.env.ARTIFACT_ADMISSION_V7, '5.0.1']]) {
      assert.equal(require(path.join(sdkPath, 'package.json')).version, expected);
      if (version === 'producer-legacy') canary.verifyHashes(sdkPath, canary.SDK_HASHES);
      else if (version === 'legacy') canary.verifyHashes(sdkPath, {'package.json': '40c8ce9effb5b2d0ac08cb9f90e01f01f06574424956ec980d2c4d48e4a07436', 'lib/internal/download/download-artifact.js': '677169b5c16e516d0cc8075fcfe7016e3340b614c924b5322270c216b7a97081'});
      else canary.verifyHashes(sdkPath, {'lib/internal/download/download-artifact.js': '9ce6bfa54d59a5271fb91b3374a8d544ef9594f20655c97ee1329ec86e2d463c'});
      const sdk = require(path.join(sdkPath, 'lib/internal/download/download-artifact.js'));
      for (const kind of Object.keys(canary.MEMBERS)) {
        const outer = path.join(root, `${version}-${kind}`);
        const destination = path.join(outer, 'artifacts');
        fs.mkdirSync(destination, {recursive: true});
        fs.writeFileSync(path.join(outer, 'outside.txt'), canary.SENTINEL);
        current = canary.archive(kind);
        const result = await sdk.streamExtractExternal(`http://127.0.0.1:${server.address().port}/owned.zip`, destination);
        if (version === 'legacy') assert.equal(result, undefined);
        else assert.equal(result.sha256Digest, `sha256:${canary.digest(current)}`);
        const escaped = version !== 'modern' && kind === 'embedded';
        assert.deepEqual(fs.readFileSync(path.join(outer, 'outside.txt')), escaped ? canary.MARKER : canary.SENTINEL);
        if (!escaped) {
          const member = kind === 'ordinary' ? 'nested/ordinary.txt' : kind === 'leading' ? 'outside.txt' : 'nested/outside.txt';
          assert.deepEqual(fs.readFileSync(path.join(destination, member)), canary.MARKER);
        }
      }
    }
  } finally {
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(root, {recursive: true, force: true});
  }
});
