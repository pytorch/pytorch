#!/usr/bin/env bash
# Build a docker image on the remote BuildKit pool used by OSDC/ARC runners,
# which have no host docker daemon. Registers a remote buildx builder, then runs
# the passed command ("$@", e.g. a build.sh or `make ... -image`) and retries
# only connection-phase failures while the autoscaled pool is cold/at capacity.
# Genuine build errors (once BuildKit has started the build) are never retried.
set -euo pipefail

case "$(uname -m)" in
  aarch64|arm64) buildkit_addr="tcp://buildkitd-arm64.buildkit:1234" ;;
  *)             buildkit_addr="tcp://buildkitd-amd64.buildkit:1234" ;;
esac
docker buildx create --name remote-buildkit --driver remote --use "${buildkit_addr}" >/dev/null 2>&1 \
  || docker buildx use remote-buildkit

log="$(mktemp)"
trap 'rm -f "${log}"' EXIT

attempts="${REMOTE_BUILDKIT_CONNECT_ATTEMPTS:-40}"
delay="${REMOTE_BUILDKIT_CONNECT_DELAY:-15}"
for attempt in $(seq 1 "${attempts}"); do
  set +e
  "$@" 2>&1 | tee "${log}"
  rc="${PIPESTATUS[0]}"
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    exit 0
  fi
  # Retry only while buildx never reached a worker (cold pool). Once BuildKit has
  # started the build it emits progress ("load build definition", context
  # transfer, "[n/m]" steps); a failure after that is a real build error.
  if [[ "${attempt}" -lt "${attempts}" ]] \
     && ! grep -qE "load build definition|transferring context|\[[0-9 ]+/[0-9 ]+\]" "${log}" \
     && grep -qiE "waiting for connection|failed to (dial|list workers)|connection (refused|reset)|no such host|context deadline exceeded|server preface" "${log}"; then
    echo "Remote BuildKit not ready yet (attempt ${attempt}/${attempts}); retrying in ${delay}s..." >&2
    sleep "${delay}"
    continue
  fi
  exit "${rc}"
done
