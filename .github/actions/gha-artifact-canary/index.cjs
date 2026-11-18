'use strict';

const {fork} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

function processIdentity(pid) {
  if (!Number.isSafeInteger(pid) || pid < 1) throw new Error('CANARY_PROCESS_IDENTITY');
  try {
    const stat = fs.readFileSync(`/proc/${pid}/stat`, 'utf8');
    const fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
    return {pid, parent: Number(fields[1]), group: Number(fields[2]), session: Number(fields[3]), start: fields[19]};
  } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ESRCH') return undefined;
    throw error;
  }
}

function parseProcess(value) {
  const identity = JSON.parse(value);
  if (!Number.isSafeInteger(identity.pid) || identity.pid <= 1 || !/^[0-9]+$/.test(identity.start)) {
    throw new Error('CANARY_PROCESS_IDENTITY');
  }
  return {pid: identity.pid, start: identity.start};
}

function sameProcess(identity) {
  identity = parseProcess(JSON.stringify(identity));
  const current = processIdentity(identity.pid);
  return current !== undefined && current.start === identity.start;
}

async function waitGone(identity) {
  const deadline = Date.now() + 5000;
  while (sameProcess(identity)) {
    if (Date.now() >= deadline) throw new Error('CANARY_PROCESS_NOT_REAPED');
    await new Promise(resolve => setTimeout(resolve, 25));
  }
}

function abortOwnGroup() {
  const current = processIdentity(process.pid);
  if (!current || current.group !== process.pid || current.session !== process.pid) {
    throw new Error('CANARY_PROCESS_IDENTITY');
  }
  // Signal the live caller's own group, never a saved numeric PGID.
  process.kill(0, 'SIGKILL');
}

function registerDownloader(child) {
  const current = processIdentity(child.pid);
  if (!current || current.parent !== process.pid) throw new Error('CANARY_PROCESS_IDENTITY');
  const identity = parseProcess(JSON.stringify(current));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error('CANARY_PROCESS_HANDSHAKE')), 15000);
    const message = value => {
      if (value?.type !== 'downloader_registered') return;
      if (JSON.stringify(value.identity) !== JSON.stringify(identity)) return finish(new Error('CANARY_PROCESS_IDENTITY'));
      finish();
    };
    function finish(error) {
      clearTimeout(timer);
      process.removeListener('message', message);
      if (error) reject(error);
      else resolve();
    }
    process.on('message', message);
    process.send({event: 'register_downloader', identity}, error => { if (error) finish(error); });
  });
}

async function supervise(entry, args, options) {
  const child = fork(entry, args, {
    env: options.env, detached: true,
    stdio: ['ignore', 'pipe', 'pipe', 'ipc'],
  });
  let bytes = 0, worker, descendant, stopPromise, failed = false;
  const closed = new Promise(resolve => child.once('close', (code, signal) => resolve({code, signal})));
  const stop = () => {
    failed = true;
    if (!stopPromise) stopPromise = (async () => {
      child.kill('SIGKILL');
      await closed;
      if (descendant) await waitGone(descendant);
    })();
    return stopPromise;
  };
  const onAbort = () => { stop().catch(() => {}); };
  const timer = setTimeout(onAbort, options.timeout);
  options.signal?.addEventListener('abort', onAbort, {once: true});
  child.on('error', onAbort);
  for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
    bytes += chunk.length;
    if (bytes > 4 * 1024 * 1024) onAbort();
  });
  child.on('message', message => {
    if (message?.event !== 'register_downloader') return options.onMessage(message);
    try {
      const identity = parseProcess(JSON.stringify(message.identity));
      const current = processIdentity(identity.pid);
      if (failed || descendant || !worker || !current || current.start !== identity.start ||
          current.parent !== worker.pid || current.group !== worker.pid || current.session !== worker.pid) {
        throw new Error('CANARY_PROCESS_IDENTITY');
      }
      descendant = identity;
      options.recordDescendant(identity);
      child.send({type: 'downloader_registered', identity}, error => { if (error) onAbort(); });
    } catch {
      onAbort();
    }
  });
  try {
    await new Promise((resolve, reject) => {
      child.once('spawn', resolve);
      child.once('error', reject);
    });
    const current = processIdentity(child.pid);
    if (!current || current.group !== child.pid || current.session !== child.pid) throw new Error('CANARY_PROCESS_IDENTITY');
    worker = parseProcess(JSON.stringify(current));
    options.recordProcess(worker);
    if (options.signal?.aborted || failed) await stop();
    else child.send({type: 'start', identity: worker}, error => { if (error) onAbort(); });
    const result = await closed;
    if (stopPromise) await stopPromise;
    else if (descendant) await waitGone(descendant);
    return {...result, failed, lifetime_drained: true};
  } catch (error) {
    await stop();
    throw error;
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener('abort', onAbort);
  }
}

async function main() {
  if (process.argv[2] === '--download') {
    // The real action bundle cannot execute until its parent registration is saved.
    process.once('disconnect', () => process.kill(process.pid, 'SIGKILL'));
    process.once('message', message => {
      if (message?.type !== 'start') throw new Error('CANARY_PROCESS_HANDSHAKE');
      process.channel.unref();
      require(path.resolve(process.argv[3]));
    });
    return;
  }
  if (process.argv[2] === '--worker') {
    const disconnected = () => abortOwnGroup();
    process.once('message', async message => {
      const current = parseProcess(JSON.stringify(processIdentity(process.pid)));
      if (message?.type !== 'start' || JSON.stringify(message.identity) !== JSON.stringify(current)) {
        throw new Error('CANARY_PROCESS_HANDSHAKE');
      }
      process.once('disconnect', disconnected);
      try {
        await require('./lib.cjs').run(process.argv[3]);
      } catch (error) {
        const code = /^CANARY_[A-Z_]+$/.test(error.message) ? error.message : 'CANARY_FAILED';
        process.send?.({event: 'failed', code});
        process.exitCode = 1;
      } finally {
        process.removeListener('disconnect', disconnected);
        if (process.connected) process.disconnect();
      }
    });
    return;
  }
  const mode = process.env.STATE_is_post === 'true' ? 'cleanup' : 'main';
  if (mode === 'main') fs.appendFileSync(process.env.GITHUB_STATE, 'is_post=true\n');
  if (mode === 'cleanup') {
    if ((process.env.STATE_owned_root || process.env.STATE_downloader_process) && !process.env.STATE_worker_process) throw new Error('CANARY_PROCESS_IDENTITY');
    for (const key of ['STATE_worker_process', 'STATE_downloader_process']) {
      if (process.env[key]) await waitGone(parseProcess(process.env[key]));
    }
  }
  const controller = new AbortController();
  const onSignal = () => controller.abort();
  process.on('SIGINT', onSignal);
  process.on('SIGTERM', onSignal);
  let cleanupRoot;
  try {
    const result = await supervise(path.resolve(__filename), ['--worker', mode], {
      env: {...process.env, ACTIONS_ARTIFACT_UPLOAD_CONCURRENCY: '1', ACTIONS_ARTIFACT_UPLOAD_TIMEOUT_MS: '15000'},
      timeout: mode === 'cleanup' ? 90000 : 300000,
      signal: controller.signal,
      recordProcess: identity => {
        if (mode === 'main') fs.appendFileSync(process.env.GITHUB_STATE, `worker_process=${JSON.stringify(identity)}\n`);
      },
      recordDescendant: identity => {
        if (mode !== 'main') throw new Error('CANARY_PROCESS_IDENTITY');
        fs.appendFileSync(process.env.GITHUB_STATE, `downloader_process=${JSON.stringify(identity)}\n`);
      },
      onMessage: message => {
        if (message?.event === 'cleanup_ready') cleanupRoot = message.root;
        else console.log(JSON.stringify(message));
      },
    });
    if (cleanupRoot) {
      require('./lib.cjs').removeOwnedRoot(cleanupRoot);
      console.log(JSON.stringify({event: 'local_cleanup', removed: true}));
    }
    if (result.code !== 0 || result.failed) process.exitCode = 1;
  } finally {
    process.removeListener('SIGINT', onSignal);
    process.removeListener('SIGTERM', onSignal);
  }
}

module.exports = {processIdentity, parseProcess, sameProcess, waitGone, abortOwnGroup, registerDownloader, supervise};
if (require.main === module) main().catch(() => {
  console.error('CANARY_PROCESS_FAILED');
  process.exitCode = 1;
});
