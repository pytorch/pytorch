'use strict';

const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {Readable} = require('node:stream');
const {setTimeout: delay} = require('node:timers/promises');

const EXPECTED_PR = 0;
const REPOSITORY = 'pytorch/pytorch';
const REPOSITORY_ID = 65600975;
const HEAD_REPOSITORY = 'ptdbot/pytorch';
const HEAD_REPOSITORY_ID = 1372177501;
const BRANCH = 'd4l3k/release-artifact-crossrun-audit';
const WORKFLOW = '.github/workflows/security-artifact-crossrun.yml';
const SDK_COMMIT = 'd3f86a106a0bac45b974a628896c90dbdf5c8093';
const LEGACY_COMMIT = '65a9edc5881444af0b9093a5e628f2fe47ea3b2e';
const MODERN_COMMIT = '37930b1c2abaa49bbe596cd826c3c89aef350131';
const SDK_DIRECTORY = '.artifact-crossrun-sdk';
const DELETE_VISIBILITY_TIMEOUT_MS = 15000;
const DELETE_VISIBILITY_INTERVAL_MS = 2000;
const DELETE_VISIBILITY_MAX_POLLS = 8;
const DELETE_POLL_RUNTIME = Object.freeze({now: () => performance.now(), wait: delay});
const MARKER = Buffer.from('release-audit-artifact-marker\n');
const SENTINEL = Buffer.from('release-audit-original-sentinel\n');
const MEMBERS = Object.freeze({
  ordinary: 'nested/ordinary.txt',
  leading: '../outside.txt',
  embedded: 'nested/../../outside.txt'
});
const SOURCE_HASHES = Object.freeze({
  'package.json': '5ec236063278fc9eaa3b2e69a79c2e4fd246c05cac819e816a8e2048762700b8',
  'package-lock.json': '0d3bee3757d13a672124cd3e38b00e3f7eb0558dc0064ebb175ec7426b5a9527'
});
const SDK_HASHES = Object.freeze({
  'package.json': '0452d4f819353b9b92bcdac189fc3221d5c16ddf4bacba2ad8e4c23d238212b7',
  'lib/artifact.js': '1badcb30279b273f258ea73fabf9805d886225b0e74b62caa8cbec923890d271',
  'lib/internal/shared/util.js': '721810b668c69128a2869d8ba97f74c06ee215568332b8136db172c28ef66e1d',
  'lib/internal/shared/artifact-twirp-client.js': '9f6b8e666388b91e526fd87201e0fe8958f9896c8b50203946606e690b5951a4',
  'lib/internal/upload/blob-upload.js': 'd9f9b48acf5d3a70c5e1d2637c75702880c6a27f1ecdb551f9098efbc644deb5',
  'lib/internal/upload/upload-artifact.js': '896a4f430a995ef92990cf8d28d3c260b542dcb0302cbfd310b6d5610cefbe9c',
  'lib/internal/upload/retention.js': '3c17499efd4b4287af790c42a1d8bfdf42f0184c6c98904efb75f67aa36149a4',
  'lib/internal/upload/upload-zip-specification.js': 'd06934dd26ecf6488fb12b295e66c28d8ac5b61a1a46f36dda258c0a776a0317',
  'lib/internal/upload/zip.js': '20f8a2e18cb8793c592a0a15822d264c474d40c9c754c3464948a3967e0654d8',
  'lib/generated/index.js': '3e1b0a351100d27ddf166e32e2625ac96757605543cf0833b1070f42b4196df1'
});
const OBSERVATION_SCHEMA = Object.freeze({
  'upload-request': {},
  'upload-response': {artifact_id: 'id', digest_valid: 'boolean'},
  'list-request': {boundary: 'boundary'},
  'list-response': {boundary: 'boundary', count: 'count', names_match: 'boolean', run_ids_match: 'boolean', job_ids_match: 'boolean', ids_valid: 'boolean', artifact_id: 'id'},
  'delete-absent': {require_present: 'boolean'},
  'delete-request': {require_present: 'boolean', known_id: 'id', listed_id: 'id'},
  'delete-response': {ok: 'nullable_boolean', artifact_id: 'id'},
  'delete-timeout': {polls: 'poll_count', reason: 'timeout_reason'},
  'delete-confirmed': {artifact_id: 'id'}
});

function digest(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

function positiveInteger(value) {
  assert.match(String(value), /^[1-9][0-9]*$/);
  const number = Number(value);
  assert(Number.isSafeInteger(number));
  return number;
}

function validateIdentity(env, event, run, pr, expectedPr = EXPECTED_PR, cleanup = false) {
  positiveInteger(expectedPr);
  const runId = positiveInteger(env.GITHUB_RUN_ID);
  const attempt = positiveInteger(env.GITHUB_RUN_ATTEMPT);
  const original = event.pull_request;
  assert.equal(env.GITHUB_EVENT_NAME, 'pull_request');
  assert.equal(env.GITHUB_REPOSITORY, REPOSITORY);
  assert.equal(env.GITHUB_REPOSITORY_ID, String(REPOSITORY_ID));
  assert.equal(env.GITHUB_ACTOR, 'ptdbot');
  assert(['ptdbot', 'd4l3k'].includes(env.GITHUB_TRIGGERING_ACTOR));
  assert.equal(env.GITHUB_REF, `refs/pull/${expectedPr}/merge`);
  assert.equal(env.GITHUB_HEAD_REF, BRANCH);
  assert.equal(env.GITHUB_BASE_REF, 'main');
  assert.match(env.GITHUB_SHA, /^[0-9a-f]{40}$/);
  assert(['opened', 'synchronize', 'reopened'].includes(event.action));
  assert.equal(event.number, expectedPr);
  assert.equal(event.repository.id, REPOSITORY_ID);
  assert.equal(event.repository.full_name, REPOSITORY);
  assert.equal(event.sender.login, 'ptdbot');
  assert.equal(original.number, expectedPr);
  assert.equal(original.user.login, 'ptdbot');
  assert.equal(original.state, 'open');
  assert.equal(original.draft, true);
  assert.equal(original.head.repo.id, HEAD_REPOSITORY_ID);
  assert.equal(original.head.repo.full_name, HEAD_REPOSITORY);
  assert.equal(original.head.ref, BRANCH);
  assert.match(original.head.sha, /^[0-9a-f]{40}$/);
  assert.equal(original.base.repo.id, REPOSITORY_ID);
  assert.equal(original.base.repo.full_name, REPOSITORY);
  assert.equal(original.base.ref, 'main');
  assert.equal(run.id, runId);
  assert.equal(run.run_attempt, attempt);
  assert.equal(run.event, 'pull_request');
  assert.equal(run.path, WORKFLOW);
  assert.equal(run.head_sha, original.head.sha);
  assert.equal(run.head_branch, BRANCH);
  assert.equal(run.repository.id, REPOSITORY_ID);
  assert.equal(run.repository.full_name, REPOSITORY);
  assert.equal(run.head_repository.id, HEAD_REPOSITORY_ID);
  assert.equal(run.head_repository.full_name, HEAD_REPOSITORY);
  assert.equal(run.actor.login, 'ptdbot');
  assert.equal(run.triggering_actor.login, env.GITHUB_TRIGGERING_ACTOR);
  assert.equal(pr.number, expectedPr);
  assert.equal(pr.user.login, 'ptdbot');
  assert.equal(pr.head.repo.id, HEAD_REPOSITORY_ID);
  assert.equal(pr.head.repo.full_name, HEAD_REPOSITORY);
  assert.equal(pr.head.ref, BRANCH);
  assert.equal(pr.base.repo.id, REPOSITORY_ID);
  assert.equal(pr.base.repo.full_name, REPOSITORY);
  assert.equal(pr.base.ref, 'main');
  if (!cleanup) {
    assert.equal(pr.state, 'open');
    assert.equal(pr.draft, true);
    assert.equal(pr.head.sha, original.head.sha);
  }
  return {repository: REPOSITORY, run_id: runId, attempt, head: original.head.sha, merge_sha: env.GITHUB_SHA, pr: expectedPr, actor: env.GITHUB_ACTOR, triggering_actor: env.GITHUB_TRIGGERING_ACTOR};
}

function fixtureRoot(context) {
  const temporary = fs.realpathSync(process.env.RUNNER_TEMP);
  const root = path.join(temporary, `release-artifact-crossrun-${context.run_id}-${context.attempt}`);
  assert.equal(path.dirname(root), temporary);
  return root;
}

function artifactName(context, kind) {
  assert(['preflight', ...Object.keys(MEMBERS)].includes(kind));
  if (kind === 'embedded') return 'flash-attn-3-wheel-cu126-x86_64';
  return `release-artifact-crossrun-${context.artifact_run_id ?? context.run_id}-${context.artifact_attempt ?? context.attempt}-${kind}`;
}

function output(name, value) {
  assert.match(name, /^[a-z0-9_]+$/);
  assert(!/[\r\n]/.test(String(value)));
  fs.appendFileSync(process.env.GITHUB_OUTPUT, `${name}=${value}\n`);
}

function readState(context) {
  const root = fixtureRoot(context);
  assert(fs.lstatSync(root).isDirectory());
  assert.equal(fs.realpathSync(root), root);
  const state = JSON.parse(fs.readFileSync(path.join(root, 'state.json'), 'utf8'));
  assert.deepEqual(state.context, context);
  assert.equal(state.schema, 1);
  assert.deepEqual(Object.keys(state).sort(), ['artifacts', 'context', 'preflight_deleted', 'schema']);
  assert.equal(typeof state.preflight_deleted, 'boolean');
  for (const [kind, artifact] of Object.entries(state.artifacts)) {
    assert.equal(artifact.name, artifactName(context, kind));
    assert(Object.keys(artifact).every(key => ['name', 'phase', 'id', 'digest'].includes(key)));
    if (artifact.id !== undefined) positiveInteger(artifact.id);
    if (artifact.digest !== undefined) assert.match(artifact.digest, /^[0-9a-f]{64}$/);
    assert(['pending', 'created', 'finalized', 'deleted'].includes(artifact.phase));
  }
  return state;
}

function writeState(context, state) {
  const root = fixtureRoot(context);
  assert.deepEqual(state.context, context);
  const temporary = path.join(root, 'state.next.json');
  fs.writeFileSync(temporary, JSON.stringify(state), {mode: 0o600});
  fs.renameSync(temporary, path.join(root, 'state.json'));
}

function observedId(value) {
  if (!['string', 'number'].includes(typeof value) || !/^(0|[1-9][0-9]*)$/.test(String(value))) return null;
  const number = Number(value);
  return Number.isSafeInteger(number) ? number : null;
}

function validateObservation(record) {
  assert(['produce', 'cleanup'].includes(record.operation));
  assert(['preflight', ...Object.keys(MEMBERS)].includes(record.kind));
  assert(Object.hasOwn(OBSERVATION_SCHEMA, record.stage));
  const fields = OBSERVATION_SCHEMA[record.stage];
  assert.deepEqual(Object.keys(record).sort(), ['operation', 'kind', 'stage', ...Object.keys(fields)].sort());
  for (const [name, type] of Object.entries(fields)) {
    const value = record[name];
    if (type === 'id') assert(value === null || (Number.isSafeInteger(value) && value >= 0));
    if (type === 'count') assert(value === null || (Number.isSafeInteger(value) && value >= 0 && value <= 10000));
    if (type === 'boolean') assert.equal(typeof value, 'boolean');
    if (type === 'nullable_boolean') assert(value === null || typeof value === 'boolean');
    if (type === 'boundary') assert(['preflight-visible', 'before-delete', 'after-delete', 'after-finalize'].includes(value));
    if (type === 'poll_count') assert(Number.isSafeInteger(value) && value >= 0 && value <= DELETE_VISIBILITY_MAX_POLLS);
    if (type === 'timeout_reason') assert(['deadline', 'attempts'].includes(value));
  }
  return record;
}

function protocolObserver(context, operation, kind) {
  return (stage, details = {}) => {
    const record = validateObservation({...details, operation, kind, stage});
    fs.appendFileSync(path.join(fixtureRoot(context), 'protocol-observations.jsonl'), `${JSON.stringify(record)}\n`, {mode: 0o600});
  };
}

function readProtocolObservations(context) {
  const file = path.join(fixtureRoot(context), 'protocol-observations.jsonl');
  if (!fs.existsSync(file)) return [];
  assert(fs.lstatSync(file).isFile());
  assert.equal(fs.realpathSync(file), file);
  assert(fs.statSync(file).size <= 65536);
  const records = fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean);
  // Four deletions at the eight-poll limit produce at most 94 records.
  assert(records.length <= 128);
  return records.map(line => validateObservation(JSON.parse(line)));
}

function initialize(context) {
  const root = fixtureRoot(context);
  fs.mkdirSync(root, {mode: 0o700});
  writeState(context, {schema: 1, context, artifacts: {}, preflight_deleted: false});
  for (const version of ['legacy', 'modern']) {
    for (const kind of Object.keys(MEMBERS)) {
      const destination = path.join(root, `${version}-${kind}`, 'artifacts');
      fs.mkdirSync(destination, {recursive: true});
      fs.writeFileSync(path.join(destination, '..', 'outside.txt'), SENTINEL, {flag: 'wx'});
      output(`${version}_${kind}_path`, destination);
    }
  }
  output('fixture_root', root);
  for (const kind of Object.keys(MEMBERS)) output(`${kind}_name`, artifactName(context, kind));
}

function verifyHashes(root, hashes) {
  for (const [relative, expected] of Object.entries(hashes)) {
    assert.equal(digest(fs.readFileSync(path.join(root, relative))), expected, `Dependency mismatch: ${relative}`);
  }
}

function verifyDependencies(installed) {
  const root = path.join(process.env.GITHUB_WORKSPACE, SDK_DIRECTORY);
  assert.equal(execFileSync('git', ['rev-parse', 'HEAD'], {cwd: root, encoding: 'utf8'}).trim(), SDK_COMMIT);
  verifyHashes(root, SOURCE_HASHES);
  const lock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));
  const entry = lock.packages['node_modules/@actions/artifact'];
  assert.equal(entry.version, '2.3.2');
  assert.equal(entry.integrity, 'sha512-uX2Mr5KEPcwnzqa0Og9wOTEKIae6C/yx9P/m8bIglzCS5nZDkcQC/zRWjjoEsyVecL6oQpBx5BuqQj/yuVm0gw==');
  const sdk = path.join(root, 'node_modules', '@actions', 'artifact');
  if (installed) verifyHashes(sdk, SDK_HASHES);
  return sdk;
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function archive(kind) {
  assert(Object.hasOwn(MEMBERS, kind));
  const name = Buffer.from(MEMBERS[kind]);
  const local = Buffer.alloc(30);
  local.writeUInt32LE(0x04034b50, 0);
  local.writeUInt16LE(20, 4);
  local.writeUInt16LE(33, 12);
  local.writeUInt32LE(crc32(MARKER), 14);
  local.writeUInt32LE(MARKER.length, 18);
  local.writeUInt32LE(MARKER.length, 22);
  local.writeUInt16LE(name.length, 26);
  const central = Buffer.alloc(46);
  central.writeUInt32LE(0x02014b50, 0);
  central.writeUInt16LE(20, 4);
  local.copy(central, 6, 4, 30);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(1, 8);
  end.writeUInt16LE(1, 10);
  end.writeUInt32LE(central.length + name.length, 12);
  end.writeUInt32LE(local.length + name.length + MARKER.length, 16);
  return Buffer.concat([local, name, MARKER, central, name, end]);
}

function protocol(sdk) {
  const service = new URL(process.env.ACTIONS_RESULTS_URL);
  assert.equal(service.protocol, 'https:');
  assert(/(^|\.)actions\.githubusercontent\.com$/.test(service.hostname));
  assert(!service.username && !service.password && !service.search && !service.hash);
  const load = relative => require(path.join(sdk, relative));
  const ids = load('lib/internal/shared/util.js').getBackendIdsFromToken();
  for (const id of Object.values(ids)) assert.match(id, /^[0-9a-f-]{36}$/i);
  return {
    ids,
    client: load('lib/internal/shared/artifact-twirp-client.js').internalArtifactTwirpClient(),
    upload: load('lib/internal/upload/upload-artifact.js').uploadArtifact,
    uploadBlob: load('lib/internal/upload/blob-upload.js').uploadZipToBlobStorage,
    expiration: load('lib/internal/upload/retention.js').getExpiration,
    StringValue: load('lib/generated/index.js').StringValue
  };
}

async function listed(service, name, observe = () => {}, boundary = 'before-delete') {
  observe('list-request', {boundary});
  const result = await service.client.ListArtifacts({...service.ids, nameFilter: service.StringValue.create({value: name})});
  const items = Array.isArray(result?.artifacts) ? result.artifacts : null;
  observe('list-response', {
    boundary,
    count: items && items.length <= 10000 ? items.length : null,
    names_match: items !== null && items.every(item => item?.name === name),
    run_ids_match: items !== null && items.every(item => item?.workflowRunBackendId === service.ids.workflowRunBackendId),
    job_ids_match: items !== null && items.every(item => item?.workflowJobRunBackendId === service.ids.workflowJobRunBackendId),
    ids_valid: items !== null && items.every(item => (observedId(item?.databaseId) ?? 0) > 0),
    artifact_id: items?.length === 1 ? observedId(items[0]?.databaseId) : null
  });
  assert(items);
  assert(items.length <= 1, 'Duplicate owned artifact');
  for (const item of items) {
    assert.equal(item.name, name);
    assert.equal(item.workflowRunBackendId, service.ids.workflowRunBackendId);
    assert.equal(item.workflowJobRunBackendId, service.ids.workflowJobRunBackendId);
    positiveInteger(item.databaseId);
  }
  return items;
}

function deletionTimeout(observe, polls, reason) {
  observe('delete-timeout', {polls, reason});
  assert.fail('Owned artifact remains after deletion visibility deadline');
}

async function waitForDeletion(service, name, deletedId, observe, runtime) {
  const deadline = runtime.now() + DELETE_VISIBILITY_TIMEOUT_MS;
  for (let polls = 0; polls < DELETE_VISIBILITY_MAX_POLLS; polls++) {
    const remaining = deadline - runtime.now();
    if (remaining <= 0) deletionTimeout(observe, polls, 'deadline');
    if (polls) await runtime.wait(Math.min(DELETE_VISIBILITY_INTERVAL_MS, remaining));
    if (runtime.now() > deadline) deletionTimeout(observe, polls, 'deadline');
    const found = await listed(service, name, observe, 'after-delete');
    if (found.length) assert.equal(positiveInteger(found[0].databaseId), deletedId, 'Deleted artifact identity changed');
    if (runtime.now() > deadline) deletionTimeout(observe, polls + 1, 'deadline');
    if (!found.length) return;
  }
  deletionTimeout(observe, DELETE_VISIBILITY_MAX_POLLS, 'attempts');
}

async function removeArtifact(service, artifact, requirePresent = false, observe = () => {}, runtime = DELETE_POLL_RUNTIME) {
  const found = await listed(service, artifact.name, observe, 'before-delete');
  const listedId = found.length ? positiveInteger(found[0].databaseId) : undefined;
  if (artifact.id !== undefined && listedId !== undefined) assert.equal(listedId, artifact.id);
  if (!found.length && artifact.phase === 'finalized') {
    observe('delete-absent', {require_present: requirePresent});
    assert.equal(requirePresent, false, 'Deletion preflight requires a present artifact');
    return;
  }
  observe('delete-request', {require_present: requirePresent, known_id: observedId(artifact.id), listed_id: observedId(listedId)});
  const response = await service.client.DeleteArtifact({...service.ids, name: artifact.name});
  observe('delete-response', {ok: typeof response?.ok === 'boolean' ? response.ok : null, artifact_id: observedId(response?.artifactId)});
  assert.equal(response.ok, true, 'Artifact service did not confirm deletion');
  const deletedId = positiveInteger(response.artifactId);
  if (artifact.id !== undefined) assert.equal(deletedId, artifact.id);
  if (listedId !== undefined) assert.equal(deletedId, listedId);
  await waitForDeletion(service, artifact.name, deletedId, observe, runtime);
  observe('delete-confirmed', {artifact_id: deletedId});
}

async function produce(context, service, runtime = DELETE_POLL_RUNTIME) {
  const state = readState(context);
  assert.equal(Object.keys(state.artifacts).length, 0);
  const file = path.join(fixtureRoot(context), 'preflight', 'ordinary.txt');
  fs.mkdirSync(path.dirname(file));
  fs.writeFileSync(file, MARKER, {flag: 'wx'});
  const preflight = {name: artifactName(context, 'preflight'), phase: 'pending'};
  state.artifacts.preflight = preflight;
  writeState(context, state);
  const observe = protocolObserver(context, 'produce', 'preflight');
  observe('upload-request');
  const uploaded = await service.upload(preflight.name, [file], path.dirname(file), {retentionDays: 1, compressionLevel: 0});
  observe('upload-response', {artifact_id: observedId(uploaded?.id), digest_valid: typeof uploaded?.digest === 'string' && /^[0-9a-f]{64}$/.test(uploaded.digest)});
  Object.assign(preflight, {phase: 'finalized', id: positiveInteger(uploaded.id), digest: uploaded.digest});
  assert.match(preflight.digest, /^[0-9a-f]{64}$/);
  writeState(context, state);
  assert.equal((await listed(service, preflight.name, observe, 'preflight-visible')).length, 1);
  await removeArtifact(service, preflight, true, observe, runtime);
  preflight.phase = 'deleted';
  state.preflight_deleted = true;
  writeState(context, state);
  for (const kind of Object.keys(MEMBERS)) {
    assert.equal(state.preflight_deleted, true);
    const bytes = archive(kind);
    const item = {name: artifactName(context, kind), phase: 'pending', digest: digest(bytes)};
    state.artifacts[kind] = item;
    writeState(context, state);
    const created = await service.client.CreateArtifact({...service.ids, name: item.name, version: 4, expiresAt: service.expiration(1)});
    assert.equal(created.ok, true);
    item.phase = 'created';
    writeState(context, state);
    const blob = await service.uploadBlob(created.signedUploadUrl, Readable.from([bytes]));
    assert.equal(blob.uploadSize, bytes.length);
    assert.equal(blob.sha256Hash, item.digest);
    const finalized = await service.client.FinalizeArtifact({...service.ids, name: item.name, size: String(bytes.length), hash: service.StringValue.create({value: `sha256:${item.digest}`})});
    assert.equal(finalized.ok, true);
    item.id = positiveInteger(finalized.artifactId);
    item.phase = 'finalized';
    writeState(context, state);
    const found = await listed(service, item.name, protocolObserver(context, 'produce', kind), 'after-finalize');
    assert.equal(found.length, 1);
    assert.equal(positiveInteger(found[0].databaseId), item.id);
  }
}

function verifyExtraction(context) {
  const state = readState(context);
  assert.equal(state.preflight_deleted, true);
  const results = [];
  for (const version of ['legacy', 'modern']) {
    for (const kind of Object.keys(MEMBERS)) {
      assert.equal(state.artifacts[kind].phase, 'finalized');
      const root = path.join(fixtureRoot(context), `${version}-${kind}`);
      const escaped = version === 'legacy' && kind === 'embedded';
      assert.deepEqual(fs.readFileSync(path.join(root, 'outside.txt')), escaped ? MARKER : SENTINEL);
      const member = kind === 'ordinary' ? 'nested/ordinary.txt' : kind === 'leading' ? 'outside.txt' : 'nested/outside.txt';
      if (!escaped) assert.deepEqual(fs.readFileSync(path.join(root, 'artifacts', member)), MARKER);
      results.push({version, kind, escaped, artifact_id: state.artifacts[kind].id, sha256: state.artifacts[kind].digest});
    }
  }
  console.log(JSON.stringify({scope: 'normally approved fork-PR cross-run artifact selection only', context, node: process.version, node_executable: process.execPath, consumers: {legacy: LEGACY_COMMIT, modern: MODERN_COMMIT}, consumer_process_runtime: 'not observed', results}));
}

async function cleanup(context, service, runtime = DELETE_POLL_RUNTIME) {
  const state = readState(context);
  for (const kind of ['preflight', ...Object.keys(MEMBERS)]) {
    const artifact = state.artifacts[kind];
    if (!artifact || artifact.phase === 'deleted') continue;
    await removeArtifact(service, artifact, false, protocolObserver(context, 'cleanup', kind), runtime);
    artifact.phase = 'deleted';
    writeState(context, state);
  }
}

module.exports = {EXPECTED_PR, REPOSITORY, REPOSITORY_ID, HEAD_REPOSITORY, HEAD_REPOSITORY_ID, BRANCH, WORKFLOW, SDK_COMMIT, MODERN_COMMIT, SOURCE_HASHES, SDK_HASHES, MEMBERS, MARKER, SENTINEL, DELETE_VISIBILITY_TIMEOUT_MS, DELETE_VISIBILITY_INTERVAL_MS, DELETE_VISIBILITY_MAX_POLLS, positiveInteger, validateIdentity, artifactName, archive, crc32, digest, verifyHashes, verifyDependencies, protocol, listed, removeArtifact, initialize, fixtureRoot, readState, writeState, observedId, validateObservation, protocolObserver, readProtocolObservations, produce, cleanup, verifyExtraction};
