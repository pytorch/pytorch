'use strict';

const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const dns = require('node:dns/promises');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');

const SPECIAL_IPV4 = new net.BlockList();
for (const [address, prefix] of [
  ['0.0.0.0', 8], ['10.0.0.0', 8], ['100.64.0.0', 10], ['127.0.0.0', 8],
  ['169.254.0.0', 16], ['172.16.0.0', 12], ['192.0.0.0', 24], ['192.0.2.0', 24],
  ['192.31.196.0', 24], ['192.52.193.0', 24], ['192.88.99.0', 24], ['192.168.0.0', 16],
  ['192.175.48.0', 24], ['198.18.0.0', 15], ['198.51.100.0', 24], ['203.0.113.0', 24],
  ['224.0.0.0', 4], ['240.0.0.0', 4]
]) SPECIAL_IPV4.addSubnet(address, prefix, 'ipv4');

function eligibleIPv4Address(address) {
  // Only IPv4 outside the explicit special-use CIDR denylist is eligible.
  return net.isIP(address) === 4 && !SPECIAL_IPV4.check(address, 'ipv4');
}

function line(socket, limit, callback) {
  let buffer = Buffer.alloc(0);
  const receive = data => {
    buffer = Buffer.concat([buffer, data]);
    const end = buffer.indexOf(10);
    if (buffer.length > limit && end < 0) return socket.destroy();
    if (end < 0) return;
    socket.off('data', receive);
    if (end > limit) return socket.destroy();
    callback(buffer.subarray(0, end).toString(), buffer.subarray(end + 1));
  };
  socket.on('data', receive);
}

async function runIsolated(root, downloadHost, python, script, source, token) {
  assert.equal(fs.realpathSync(root), root);
  assert(/^[a-z0-9.-]+$/.test(downloadHost));
  const allowed = new Set(['api.github.com', downloadHost]);
  const observations = {admission_denied: 0, dns_lookups: 0, dials: 0};
  const socketPath = path.join(root, 'https-relay.sock');
  const sockets = new Set();
  const relay = net.createServer(client => {
    sockets.add(client);
    client.on('close', () => sockets.delete(client));
    client.on('error', () => {});
    client.setTimeout(30000, () => client.destroy());
    line(client, 300, async (host, rest) => {
      if (!allowed.has(host) || rest.length) {
        observations.admission_denied++;
        return client.destroy();
      }
      try {
        observations.dns_lookups++;
        const addresses = await dns.lookup(host, {all: true, family: 4, verbatim: true});
        assert(addresses.length && addresses.every(item => item.family === 4 && eligibleIPv4Address(item.address)));
        const selected = addresses[0];
        observations.dials++;
        const upstream = net.connect({host: selected.address, port: 443, family: selected.family});
        sockets.add(upstream);
        upstream.on('close', () => { sockets.delete(upstream); client.destroy(); });
        client.on('close', () => upstream.destroy());
        upstream.on('error', () => client.destroy());
        upstream.setTimeout(30000, () => upstream.destroy());
        upstream.once('connect', () => { client.write('OK\n'); client.pipe(upstream); upstream.pipe(client); });
      } catch { client.destroy(); }
    });
  });
  await new Promise((resolve, reject) => { relay.once('error', reject); relay.listen(socketPath, resolve); });
  fs.chmodSync(socketPath, 0o600);
  const home = path.join(root, 'private-home');
  fs.mkdirSync(home, {mode: 0o700});
  try {
    const env = {PATH: `${path.dirname(process.execPath)}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin`, HOME: home, GITHUB_TOKEN: token, AWS_EC2_METADATA_DISABLED: 'true', AWS_CONFIG_FILE: '/dev/null', AWS_SHARED_CREDENTIALS_FILE: '/dev/null'};
    const args = ['--user', '--map-root-user', '--net', '--', 'bash', '-ceu', 'ip link set lo up; exec "$@"', 'artifact-attempt-net', process.execPath, __filename, 'inner', socketPath, python, script, source, root];
    const child = spawn('unshare', args, {env, stdio: 'inherit', detached: true});
    const deadline = setTimeout(() => { if (child.pid) { try { process.kill(-child.pid, 'SIGKILL'); } catch {} } }, 120000);
    let status;
    try {
      status = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', (code, signal) => resolve({code, signal})); });
    } finally {
      clearTimeout(deadline);
      if (child.pid) { try { process.kill(-child.pid, 'SIGKILL'); } catch {} }
    }
    assert.deepEqual(status, {code: 0, signal: null}, 'Isolated original downloader failed');
  } finally {
    for (const socket of sockets) socket.destroy();
    await new Promise(resolve => relay.close(resolve));
    fs.rmSync(socketPath, {force: true});
  }
  return Object.freeze({...observations});
}

async function inner(socketPath, python, script, source, root) {
  const server = http.createServer((_request, response) => { response.writeHead(405); response.end(); });
  const sockets = new Set();
  server.on('connection', socket => { sockets.add(socket); socket.on('close', () => sockets.delete(socket)); socket.on('error', () => {}); });
  server.on('connect', (request, client, head) => {
    const match = /^([a-z0-9.-]+):443$/.exec(request.url);
    if (!match) return client.destroy();
    const upstream = net.connect(socketPath);
    upstream.on('error', () => client.destroy());
    upstream.on('close', () => client.destroy());
    client.on('close', () => upstream.destroy());
    upstream.once('connect', () => upstream.write(`${match[1]}\n`));
    line(upstream, 3, (status, rest) => {
      if (status !== 'OK') return client.destroy();
      client.write('HTTP/1.1 200 Connection Established\r\n\r\n');
      if (head.length) upstream.write(head);
      if (rest.length) client.write(rest);
      client.pipe(upstream); upstream.pipe(client);
    });
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const proxy = `http://127.0.0.1:${server.address().port}`;
  try {
    const env = {...process.env, HTTPS_PROXY: proxy, https_proxy: proxy, HTTP_PROXY: proxy, http_proxy: proxy, NO_PROXY: '', no_proxy: '', PYTHONDONTWRITEBYTECODE: '1', PYTHONNOUSERSITE: '1'};
    const child = spawn(python, ['-I', script, source, root], {env, cwd: root, stdio: 'inherit'});
    const status = await new Promise((resolve, reject) => { child.once('error', reject); child.once('exit', (code, signal) => resolve({code, signal})); });
    assert.deepEqual(status, {code: 0, signal: null});
  } finally {
    for (const socket of sockets) socket.destroy();
    await new Promise(resolve => server.close(resolve));
  }
}

module.exports = {eligibleIPv4Address, runIsolated};
if (require.main === module) {
  assert.equal(process.argv[2], 'inner');
  inner(...process.argv.slice(3)).catch(() => { console.error('Isolated HTTPS tunnel failed closed'); process.exitCode = 1; });
}
