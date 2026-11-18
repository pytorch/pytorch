'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const https = require('node:https');
const Module = require('node:module');
const path = require('node:path');
const {execFileSync, fork} = require('node:child_process');
const {Readable} = require('node:stream');
const guard = require('./guard.cjs');
const {abortOwnGroup, registerDownloader} = require('./index.cjs');

const PIN = 'd3f86a106a0bac45b974a628896c90dbdf5c8093'; // actions/download-artifact v4.3.0
const BUNDLE_SHA = '02c2cf24c5b60ae4b3f8d71d3cc85e2e6433dbdf106a38fee7ec3f1c96e2d18e';
const BRANCH = 'd4l3k/gha-infra-artifact-canary-20261117';
const WORKFLOW = '.github/workflows/ci-gha-artifact-canary.yml';
const REPOSITORY = 'pytorch/pytorch';
const REPO_ID = 65600975;
const FORK_ID = 1372177501;
const PR_NUMBER = 205438;
const STARTUP = 'var __webpack_exports__ = __nccwpck_require__(19946);';
const LIMIT = 8192;
const sha256 = value => crypto.createHash('sha256').update(value).digest('hex');
const check = (condition, code = 'CANARY_IDENTITY') => {
  if (!condition) throw new Error(code);
};
const emit = value => process.send?.(value);
const number = value => /^[1-9][0-9]*$/.test(String(value));

function crc32(bytes) {
  let value = 0xffffffff;
  for (const byte of bytes) {
    value ^= byte;
    for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ ((value & 1) ? 0xedb88320 : 0);
  }
  return (value ^ 0xffffffff) >>> 0;
}

function archive(nonce) {
  check(/^[a-f0-9]{32}$/.test(nonce));
  const entries = [
    ['nested/../../escaped.txt', `escape:${nonce}`],
    ['../../leading.txt', `leading:${nonce}`],
    ['nested/inside.txt', `contained:${nonce}`],
  ];
  const locals = [], centrals = [];
  let offset = 0;
  for (const [name, text] of entries) {
    const filename = Buffer.from(name), data = Buffer.from(text), crc = crc32(data);
    const local = Buffer.alloc(30), central = Buffer.alloc(46);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(filename.length, 26);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(filename.length, 28);
    central.writeUInt32LE(offset, 42);
    locals.push(local, filename, data);
    centrals.push(central, filename);
    offset += local.length + filename.length + data.length;
  }
  const directory = Buffer.concat(centrals), end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(directory.length, 12);
  end.writeUInt32LE(offset, 16);
  const result = Buffer.concat([...locals, directory, end]);
  check(result.length <= LIMIT, 'CANARY_SIZE');
  return result;
}

function modules(filename) {
  const original = fs.readFileSync(filename, 'utf8');
  check(sha256(original) === BUNDLE_SHA, 'CANARY_BUNDLE_DIGEST');
  check(original.split(STARTUP).length === 2, 'CANARY_BUNDLE_ENTRY');
  const loaded = new Module(filename, module);
  loaded.filename = filename;
  loaded.paths = Module._nodeModulePaths(path.dirname(filename));
  loaded._compile(original.replace(STARTUP, 'var __webpack_exports__ = __nccwpck_require__;'), filename);
  return loaded.exports;
}

function storageURL(value) {
  const url = new URL(value);
  check(/^[a-z0-9][a-z0-9.-]*\.(blob\.core\.windows\.net|actions\.githubusercontent\.com)$/.test(url.hostname), 'CANARY_STORAGE_HOST');
  guard.checkURL(url, new Set([url.hostname]));
  guard.allowed.add(url.hostname);
  return url;
}

function get(url, maxBytes, headers = {}) {
  return new Promise((resolve, reject) => {
    const request = https.get(url, {headers, rejectUnauthorized: true}, response => {
      const chunks = [];
      let length = 0;
      response.on('data', chunk => {
        length += chunk.length;
        if (length > maxBytes) response.destroy(new Error('CANARY_RESPONSE_SIZE'));
        else chunks.push(chunk);
      });
      response.once('error', reject);
      response.once('end', () => resolve({status: response.statusCode, body: Buffer.concat(chunks)}));
    });
    request.once('error', reject);
  });
}

async function api(endpoint, expected = 200) {
  check(endpoint.startsWith(`/repos/${REPOSITORY}/`), 'CANARY_API_SCOPE');
  const response = await get(`https://api.github.com${endpoint}`, 2 * 1024 * 1024, {
    Accept: 'application/vnd.github+json', Authorization: `Bearer ${process.env['INPUT_GITHUB-TOKEN']}`,
    'User-Agent': 'pytorch-owned-artifact-canary', 'X-GitHub-Api-Version': '2022-11-28',
  });
  check(response.status === expected, 'CANARY_API_STATUS');
  return expected === 200 ? JSON.parse(response.body) : undefined;
}

function eventIdentity(env, event) {
  const pr = event.pull_request;
  check(env.GITHUB_SERVER_URL === 'https://github.com' && env.GITHUB_API_URL === 'https://api.github.com');
  check(env.GITHUB_ACTIONS === 'true' && env.GITHUB_EVENT_NAME === 'pull_request');
  check(env.GITHUB_REPOSITORY === REPOSITORY && Number(env.GITHUB_REPOSITORY_ID) === REPO_ID);
  check(env.GITHUB_ACTOR === 'ptdbot' && env.GITHUB_TRIGGERING_ACTOR === 'd4l3k');
  check(env.GITHUB_RUN_ATTEMPT === '2' && number(env.GITHUB_RUN_ID) && env.GITHUB_JOB === 'probe');
  check(pr?.draft === true && pr.state === 'open' && pr.user?.login === 'ptdbot');
  check(event.repository?.id === REPO_ID && pr.base?.repo?.id === REPO_ID);
  check(pr.head?.repo?.id === FORK_ID && pr.head.repo.full_name === 'ptdbot/pytorch' && pr.head.ref === BRANCH);
  check(pr.number === PR_NUMBER && /^[a-f0-9]{40}$/.test(pr.head.sha) && env['INPUT_EXPECTED-HEAD'] === pr.head.sha);
  check(/^[a-f0-9]{40}$/.test(env.GITHUB_SHA));
  check(env.GITHUB_WORKFLOW_REF === `${REPOSITORY}/${WORKFLOW}@refs/pull/${pr.number}/merge`);
  return {repository: REPOSITORY, repository_id: REPO_ID, fork_id: FORK_ID,
    run_id: env.GITHUB_RUN_ID, run_attempt: 2, triggering_actor: 'd4l3k', job: 'probe', pr: pr.number,
    head_sha: pr.head.sha, merge_sha: env.GITHUB_SHA, workflow: WORKFLOW};
}

function runSourceIdentity(run, identity) {
  check(String(run.id) === identity.run_id && run.event === 'pull_request');
  check(run.path === WORKFLOW && run.head_branch === BRANCH);
  check(run.repository?.id === REPO_ID && run.head_repository?.id === FORK_ID);
  check(run.actor?.login === 'ptdbot' && [identity.head_sha, identity.merge_sha].includes(run.head_sha));
  return run.head_sha;
}

function runIdentity(run, identity) {
  check(run.run_attempt === 2 && run.triggering_actor?.login === 'd4l3k');
  return runSourceIdentity(run, identity);
}

function priorAttemptIdentity(run, identity, approvedHead) {
  check(run.run_attempt === 1 && run.triggering_actor?.login === 'ptdbot' &&
    run.status === 'completed' && run.conclusion === 'action_required', 'CANARY_APPROVAL');
  check(runSourceIdentity(run, identity) === approvedHead, 'CANARY_APPROVAL_HEAD');
}

async function identity(mode) {
  check(!process.env.NODE_OPTIONS && process.env.NODE_TLS_REJECT_UNAUTHORIZED !== '0', 'CANARY_TLS_CONFIG');
  for (const name of ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']) {
    check(!process.env[name], 'CANARY_PROXY_CONFIG');
  }
  check(process.env['INPUT_GITHUB-TOKEN'] && process.env.ACTIONS_RUNTIME_TOKEN, 'CANARY_CREDENTIAL_MISSING');
  const result = eventIdentity(process.env, JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8')));
  check(execFileSync('git', ['rev-parse', 'HEAD'], {encoding: 'utf8', cwd: process.env.GITHUB_WORKSPACE}).trim() === result.head_sha);
  const results = new URL(process.env.ACTIONS_RESULTS_URL);
  check(/^[a-z0-9][a-z0-9.-]*\.actions\.githubusercontent\.com$/.test(results.hostname), 'CANARY_RESULTS_HOST');
  guard.checkURL(results, new Set([results.hostname]));
  check(!results.search && !results.hash, 'CANARY_RESULTS_URL');
  guard.allowed.add(results.hostname);
  process.env.CANARY_RESULTS_HOST = results.hostname;
  guard.allowed.add('raw.githubusercontent.com');
  guard.allowed.add('api.github.com');
  guard.install();
  if (mode !== 'cleanup') {
    const current = await api(`/repos/${REPOSITORY}/pulls/${result.pr}`);
    check(current.state === 'open' && current.draft && current.head?.sha === result.head_sha);
    check(current.head?.repo?.id === FORK_ID && current.head?.ref === BRANCH && current.base?.repo?.id === REPO_ID);
  }
  result.run_head_sha = runIdentity(await api(`/repos/${REPOSITORY}/actions/runs/${result.run_id}/attempts/2`), result);
  priorAttemptIdentity(await api(`/repos/${REPOSITORY}/actions/runs/${result.run_id}/attempts/1`), result, result.run_head_sha);
  return result;
}

function save(state) {
  fs.writeFileSync(path.join(state.root, 'ownership.json'), JSON.stringify(state), {mode: 0o600});
}

function readState(root, identity) {
  const base = fs.realpathSync(process.env.RUNNER_TEMP);
  check(path.dirname(root) === base && /^gha-artifact-canary-[a-f0-9]{32}$/.test(path.basename(root)), 'CANARY_ROOT');
  check(fs.realpathSync(root) === root && !fs.lstatSync(root).isSymbolicLink(), 'CANARY_ROOT');
  const state = JSON.parse(fs.readFileSync(path.join(root, 'ownership.json'), 'utf8'));
  for (const [key, value] of Object.entries(identity)) check(state[key] === value);
  check(state.root === root && path.basename(root) === `gha-artifact-canary-${state.nonce}`);
  check(state.name === `gha-infra-owned-${state.run_id}-${state.nonce}`);
  check(state.digest === sha256(archive(state.nonce)) && state.size === archive(state.nonce).length);
  return state;
}

function removeOwnedRoot(root) {
  const identity = eventIdentity(process.env, JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8')));
  const state = readState(root, identity);
  check(!state.create_attempted || state.cleanup_verified, 'CANARY_LOCAL_CLEANUP');
  fs.rmSync(root, {recursive: true});
  check(!fs.existsSync(root), 'CANARY_LOCAL_CLEANUP');
}

function verifyFiles(state) {
  const dest = path.join(state.root, 'destination');
  check(fs.realpathSync(dest) === dest, 'CANARY_DESTINATION');
  check(fs.readFileSync(path.join(state.root, 'escaped.txt'), 'utf8') === `escape:${state.nonce}`, 'CANARY_ESCAPE');
  check(fs.readFileSync(path.join(dest, 'leading.txt'), 'utf8') === `leading:${state.nonce}`, 'CANARY_LEADING');
  check(fs.readFileSync(path.join(dest, 'nested', 'inside.txt'), 'utf8') === `contained:${state.nonce}`, 'CANARY_CONTAINED');
  for (const filename of ['escaped.txt', 'destination/leading.txt', 'destination/nested/inside.txt']) {
    check(fs.lstatSync(path.join(state.root, filename)).isFile(), 'CANARY_FILE_TYPE');
  }
  assert.deepEqual(fs.readdirSync(state.root).sort(), ['destination', 'download-artifact.cjs', 'download-output', 'escaped.txt', 'ownership.json']);
  assert.deepEqual(fs.readdirSync(dest).sort(), ['leading.txt', 'nested']);
  assert.deepEqual(fs.readdirSync(path.join(dest, 'nested')), ['inside.txt']);
}

function artifactOwner(artifact, state) {
  check(artifact.name === state.name && artifact.workflowRunBackendId === state.workflowRunBackendId &&
    artifact.workflowJobRunBackendId === state.workflowJobRunBackendId, 'CANARY_ARTIFACT_OWNER');
  check(number(artifact.databaseId), 'CANARY_ARTIFACT_ID');
  if (state.id) check(artifact.databaseId === state.id, 'CANARY_ARTIFACT_ID');
  return artifact.databaseId;
}

function matchesArtifact(artifact, state) {
  const id = artifactOwner(artifact, state);
  check(artifact.size === String(state.size) && artifact.digest?.value === `sha256:${state.digest}`, 'CANARY_ARTIFACT_CONTENT');
  return id;
}

function publicArtifactOwner(artifact, state) {
  check(number(artifact.id) && String(artifact.id) === state.id && artifact.name === state.name, 'CANARY_REST_ARTIFACT');
  check(String(artifact.workflow_run?.id) === state.run_id && artifact.workflow_run?.repository_id === REPO_ID &&
    artifact.workflow_run?.head_repository_id === FORK_ID && artifact.workflow_run?.head_branch === BRANCH &&
    artifact.workflow_run?.head_sha === state.run_head_sha, 'CANARY_REST_RUN');
}

function matchesPublicArtifact(artifact, state) {
  publicArtifactOwner(artifact, state);
  check(artifact.size_in_bytes === state.size && artifact.digest === `sha256:${state.digest}` &&
    artifact.expired === false, 'CANARY_REST_CONTENT');
}

async function loadToolkit(state) {
  const filename = path.join(state.root, 'download-artifact.cjs');
  if (!fs.existsSync(filename)) {
    const response = await get(`https://raw.githubusercontent.com/actions/download-artifact/${PIN}/dist/index.js`, 6 * 1024 * 1024);
    check(response.status === 200 && sha256(response.body) === BUNDLE_SHA, 'CANARY_BUNDLE_DIGEST');
    fs.writeFileSync(filename, response.body, {mode: 0o600, flag: 'wx'});
  }
  const requireModule = modules(filename);
  const ids = requireModule(63062).getBackendIdsFromToken();
  for (const key of ['workflowRunBackendId', 'workflowJobRunBackendId']) {
    check(/^[a-f0-9-]{36}$/i.test(ids[key]), 'CANARY_BACKEND_ID');
    if (state[key]) check(state[key] === ids[key], 'CANARY_BACKEND_ID');
  }
  Object.assign(state, ids);
  save(state);
  return {filename, ids, generated: requireModule(49960),
    client: requireModule(12312).internalArtifactTwirpClient({maxAttempts: 1}),
    upload: requireModule(7246).uploadZipToBlobStorage};
}

async function list(toolkit, state) {
  const result = await toolkit.client.ListArtifacts({...toolkit.ids, nameFilter: {value: state.name}});
  check(result.artifacts.length <= 1, 'CANARY_ARTIFACT_AMBIGUOUS');
  return result.artifacts;
}

async function cleanup(state, toolkit) {
  if (!state.create_attempted) return;
  check(state.creation_scope_verified, 'CANARY_CLEANUP_SCOPE');
  if (!state.deleted) {
    // The pre-create empty-name check and saved exact job/name remain valid
    // even if finalization succeeds but later discovery is unavailable.
    let response;
    try {
      response = await toolkit.client.ListArtifacts({...toolkit.ids, nameFilter: {value: state.name}});
    } catch {
      response = undefined;
    }
    if (response) {
      check(response.artifacts.length <= 1, 'CANARY_ARTIFACT_AMBIGUOUS');
      if (response.artifacts.length) {
        state.id = artifactOwner(response.artifacts[0], state);
        save(state);
      }
    }
    const result = await toolkit.client.DeleteArtifact({...toolkit.ids, name: state.name});
    check(result.ok && number(result.artifactId), 'CANARY_DELETE');
    if (state.id) check(result.artifactId === state.id, 'CANARY_DELETE_ID');
    state.id = result.artifactId;
    state.deleted = true;
    save(state);
  }
  check((await list(toolkit, state)).length === 0, 'CANARY_DELETE_LIST');
  await api(`/repos/${REPOSITORY}/actions/artifacts/${state.id}`, 404);
  state.cleanup_verified = true;
  save(state);
  emit({event: 'deleted', name: state.name, artifact_id: state.id, api_404: true});
}

async function downloader(state, toolkit, downloadHost) {
  fs.writeFileSync(path.join(state.root, 'download-output'), '', {mode: 0o600, flag: 'wx'});
  const env = {...process.env, CANARY_INSTALL_GUARD: '1',
    CANARY_HTTPS_HOSTS: `${new URL(process.env.ACTIONS_RESULTS_URL).hostname},${downloadHost}`,
    INPUT_NAME: state.name, INPUT_PATH: path.join(state.root, 'destination'),
    INPUT_PATTERN: '', 'INPUT_MERGE-MULTIPLE': 'false', 'INPUT_ARTIFACT-IDS': '',
    'INPUT_GITHUB-TOKEN': '', INPUT_REPOSITORY: '', 'INPUT_RUN-ID': '',
    GITHUB_OUTPUT: path.join(state.root, 'download-output'), NODE_OPTIONS: '',
  };
  const child = fork(path.join(__dirname, 'index.cjs'), ['--download', toolkit.filename], {
    env, execArgv: ['--require', path.join(__dirname, 'guard.cjs')],
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
  });
  let output = '', bytes = 0, failed = false;
  const closed = new Promise(resolve => child.once('close', resolve));
  const timer = setTimeout(abortOwnGroup, 90000);
  child.on('error', () => { failed = true; });
  for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
    bytes += chunk.length;
    if (bytes > 1024 * 1024) abortOwnGroup();
    else output += chunk.toString('utf8');
  });
  try {
    await new Promise((resolve, reject) => {
      child.once('spawn', resolve);
      child.once('error', reject);
    });
    await registerDownloader(child);
    await new Promise((resolve, reject) => child.send({type: 'start'}, error => error ? reject(error) : resolve()));
    const code = await closed;
    check(!failed && code === 0 && output.includes(`SHA256 digest of downloaded artifact is ${state.digest}`) &&
      !output.includes('digest mismatch'), 'CANARY_DOWNLOADER');
  } catch (error) {
    child.kill('SIGKILL');
    await closed;
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function run(mode) {
  if (mode === 'cleanup' && (!process.env.STATE_owned_root || !fs.existsSync(process.env.STATE_owned_root))) {
    return emit({event: 'post_cleanup', present: false});
  }
  const current = await identity(mode);
  let state, toolkit;
  if (mode === 'cleanup') {
    const root = process.env.STATE_owned_root;
    state = readState(root, current);
    toolkit = await loadToolkit(state);
    await cleanup(state, toolkit);
    return emit({event: 'cleanup_ready', root: state.root});
  }
  const nonce = crypto.randomBytes(16).toString('hex');
  const root = path.join(fs.realpathSync(process.env.RUNNER_TEMP), `gha-artifact-canary-${nonce}`);
  fs.mkdirSync(root, {mode: 0o700});
  const bytes = archive(nonce);
  state = {...current, nonce, root, name: `gha-infra-owned-${current.run_id}-${nonce}`,
    size: bytes.length, digest: sha256(bytes), create_attempted: false};
  save(state);
  fs.appendFileSync(process.env.GITHUB_STATE, `owned_root=${root}\n`);
  emit({event: 'prepared', ...current, name: state.name, digest: state.digest, size: state.size});
  try {
    toolkit = await loadToolkit(state);
    check((await list(toolkit, state)).length === 0, 'CANARY_NAME_EXISTS');
    state.creation_scope_verified = true;
    state.create_attempted = true;
    save(state);
    const created = await toolkit.client.CreateArtifact({...toolkit.ids, name: state.name, version: 4,
      expiresAt: toolkit.generated.Timestamp.fromDate(new Date(Date.now() + 86400000))});
    check(created.ok, 'CANARY_CREATE');
    state.create_confirmed = true;
    save(state);
    const uploadURL = storageURL(created.signedUploadUrl);
    const uploaded = await toolkit.upload(uploadURL.href, Readable.from([bytes]));
    check(uploaded.uploadSize === state.size && uploaded.sha256Hash === state.digest, 'CANARY_UPLOAD_DIGEST');
    const finalized = await toolkit.client.FinalizeArtifact({...toolkit.ids, name: state.name,
      size: String(state.size), hash: {value: `sha256:${state.digest}`}});
    check(finalized.ok && number(finalized.artifactId), 'CANARY_FINALIZE');
    state.id = finalized.artifactId;
    save(state);
    emit({event: 'finalized', name: state.name, artifact_id: state.id, digest: state.digest, size: state.size});
    const artifacts = await list(toolkit, state);
    check(artifacts.length === 1, 'CANARY_ARTIFACT_MISSING');
    matchesArtifact(artifacts[0], state);
    matchesPublicArtifact(await api(`/repos/${REPOSITORY}/actions/artifacts/${state.id}`), state);
    const signed = await toolkit.client.GetSignedArtifactURL({...toolkit.ids, name: state.name});
    const downloadURL = storageURL(signed.signedUrl);
    const raw = await get(downloadURL, LIMIT);
    check(raw.status === 200 && raw.body.equals(bytes), 'CANARY_SERVICE_BYTES');
    fs.mkdirSync(path.join(root, 'destination'), {mode: 0o700});
    await downloader(state, toolkit, downloadURL.hostname);
    verifyFiles(state);
    emit({event: 'proof', artifact_id: state.id, digest: state.digest,
      service_bytes_equal: true, internal_escape: true, leading_contained: true, ordinary_contained: true});
  } finally {
    if (toolkit) await cleanup(state, toolkit);
    if (!state.create_attempted || state.cleanup_verified) {
      emit({event: 'cleanup_ready', root: state.root});
    }
  }
}

module.exports = {PIN, BUNDLE_SHA, BRANCH, WORKFLOW, REPOSITORY, REPO_ID, FORK_ID, PR_NUMBER, LIMIT,
  archive, modules, storageURL, eventIdentity, runIdentity, priorAttemptIdentity, artifactOwner, matchesArtifact, publicArtifactOwner, matchesPublicArtifact,
  readState, removeOwnedRoot, verifyFiles, run, sha256};
