'use strict';

const http = require('node:http');
const https = require('node:https');

const allowed = new Set((process.env.CANARY_HTTPS_HOSTS || '').split(','));
const originalRequest = https.request;

function fail() {
  throw new Error('CANARY_NETWORK_SCOPE');
}

function checkURL(value, hosts = allowed) {
  const url = new URL(value);
  if (url.protocol !== 'https:' || url.username || url.password ||
      (url.port && url.port !== '443') || !hosts.has(url.hostname)) fail();
  return url;
}

function guardedRequest(...args) {
  const first = args[0];
  let options, target;
  if (typeof first === 'string' || first instanceof URL) {
    const url = checkURL(first);
    target = url.hostname;
    options = args[1] && typeof args[1] === 'object' ? args[1] : {};
    if (options.hostname || options.host || options.socketPath || options.createConnection) fail();
    args = [url, {...options, rejectUnauthorized: true},
      typeof args[1] === 'function' ? args[1] : args[2]];
  } else {
    options = first || {};
    if (options.socketPath || options.createConnection || options.protocol === 'http:') fail();
    target = checkURL(`https://${options.hostname || options.host}${options.port ? `:${options.port}` : ''}/`).hostname;
    args[0] = {...options, rejectUnauthorized: true};
  }
  for (const [key] of Object.entries(options.headers || {})) {
    if (key.toLowerCase() === 'proxy-authorization') fail();
    if (key.toLowerCase() === 'authorization' && !['api.github.com', process.env.CANARY_RESULTS_HOST].includes(target)) fail();
  }
  const request = originalRequest.apply(https, args);
  request.setTimeout(15000, () => request.destroy(new Error('CANARY_NETWORK_TIMEOUT')));
  return request;
}

function install() {
  https.request = guardedRequest;
  https.get = (...args) => {
    const request = guardedRequest(...args);
    request.end();
    return request;
  };
  http.request = fail;
  http.get = fail;
}

if (process.env.CANARY_INSTALL_GUARD === '1') install();
module.exports = {allowed, checkURL, install};
