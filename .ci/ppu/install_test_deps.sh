#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="$REPO_ROOT/.ci/docker/requirements-ci.txt"
WANTED=(pytest pytest-xdist pytest-flakefinder pytest-rerunfailures pytest-subtests expecttest hypothesis pyyaml ninja)
PINNED=(scipy==1.13.1 psutil)
SPECS_RAW="$(python - "$REQ_FILE" "${WANTED[@]}" <<'PY'
import re
import sys

req_file, wanted = sys.argv[1], sys.argv[2:]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

found: dict[str, str] = {}
with open(req_file) as f:
    for line in f:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9._-]+)\s*((?:==|>=|<=|~=|!=|<|>).+)$", line)
        if m and norm(m.group(1)) not in found:
            found[norm(m.group(1))] = m.group(1) + m.group(2).replace(" ", "")

missing = [p for p in wanted if norm(p) not in found]
if missing:
    sys.exit(f"[deps] No version constraint in requirements-ci.txt for: {' '.join(missing)}")
print(" ".join(found[norm(p)] for p in wanted))
PY
)"
read -r -a PKGS <<<"$SPECS_RAW"
PKGS+=("${PINNED[@]}")
echo "=== Target versions (from .ci/docker/requirements-ci.txt and pinned in script) ==="
printf '  %s\n' "${PKGS[@]}"

echo "=== Test deps check ==="
if python - "${PKGS[@]}" <<'PY'
import re
import sys
from importlib.metadata import PackageNotFoundError, version


def key(v: str) -> tuple[int, ...]:
    # Compare numeric parts only (7.3.2 -> (7,3,2)); enough for these == / >= constraints, no packaging dependency
    return tuple(int(x) for x in re.findall(r"\d+", v)[:4])


def pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


unsatisfied = []
for spec in sys.argv[1:]:
    if re.fullmatch(r"[A-Za-z0-9._-]+", spec):
        try:
            print(f"  {spec} {version(spec)} (any version) OK")
        except PackageNotFoundError:
            print(f"  {spec} not installed (any version)")
            unsatisfied.append(spec)
        continue
    m = re.match(r"^([A-Za-z0-9._-]+)\s*(==|>=|<=|~=|<|>)\s*(.+)$", spec)
    if not m:
        unsatisfied.append(spec)
        continue
    name, op, want = m.groups()
    try:
        got = version(name)
    except PackageNotFoundError:
        print(f"  {name} not installed (need {op}{want})")
        unsatisfied.append(spec)
        continue
    a, b = pad(key(got), key(want))
    ok = {
        "==": a == b,
        ">=": a >= b,
        "<=": a <= b,
        ">": a > b,
        "<": a < b,
        "~=": a >= b and a[: len(b) - 1] == b[: len(b) - 1],
    }[op]
    print(f"  {name} {got} (need {op}{want}) {'OK' if ok else 'MISMATCH'}")
    if not ok:
        unsatisfied.append(spec)
raise SystemExit(1 if unsatisfied else 0)
PY
then
    echo "[deps] All versions satisfied, skipping install"
    exit 0
fi

source "$REPO_ROOT/.ci/ppu/pip_sources.sh"
ppu_build_pip_candidates
CANDIDATES=("${PIP_CANDIDATES[@]}")

echo "=== pip source connectivity check ==="
echo "[probe] Proxy env vars set (names only, values hidden to avoid leaking credentials): $(
    _set_proxy_names=""
    for _pv in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
        [[ -n "${!_pv:-}" ]] && _set_proxy_names="${_set_proxy_names}${_pv} "
    done
    printf '%s' "${_set_proxy_names:-<none>}"
)"
echo "[probe] pip config:"
python -m pip config list 2>&1 | sed 's/^/        /' || true
python - "${CANDIDATES[@]}" <<'PY' || true
import socket
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

dns_cache: dict[str, str] = {}
for index in sys.argv[1:]:
    host = urlsplit(index).hostname or "?"
    if host not in dns_cache:
        try:
            dns_cache[host] = ", ".join(socket.gethostbyname_ex(host)[2])
        except OSError as exc:
            dns_cache[host] = f"resolve failed ({exc})"
    print(f"[probe] {index}")
    print(f"        dns({host}) = {dns_cache[host]}")
    url = index.rstrip("/") + "/pytest-rerunfailures/"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            print(f"        GET .../pytest-rerunfailures/ = {resp.status}")
    except urllib.error.HTTPError as exc:
        # 404 = repo/package not in this Artifactory view; 401/403 = auth required
        print(f"        GET .../pytest-rerunfailures/ = {exc.code} {exc.reason}")
    except Exception as exc:
        print(f"        GET .../pytest-rerunfailures/ = unreachable ({type(exc).__name__}: {exc})")
PY

echo "=== Installing test deps ==="
for index in "${CANDIDATES[@]}" __pip_default__; do
    pip_args=(--disable-pip-version-check --retries 1 --timeout 20)
    if [[ "${index}" == "__pip_default__" ]]; then
        label="pip default index"
    else
        label="${index}"
        pip_args+=(-i "${index}")
    fi
    echo "[deps] Trying index: ${label}"
    if python -m pip install "${pip_args[@]}" "${PKGS[@]}"; then
        echo "[deps] Installed (index: ${label})"
        exit 0
    fi
    echo "[deps][warn] Index ${label} failed, trying next"
done

cat >&2 <<'EOF'
[deps][error] No candidate pip index could install the test deps listed above.
Check the "pip source connectivity check" output above:
  - all 404          -> the pod's Artifactory view lacks these pypi repos; ask ops for the right repo name,
                        then update PIP_INDEX in the workflow (or add candidates via PIP_INDEX_FALLBACKS).
  - 401/403          -> the index needs auth; provide credentials in the pod (pip.conf / .netrc).
  - DNS resolve fail -> the pod's DNS cannot resolve the host.
  - public unreachable -> the pod has no internet egress; use an internal index.
Fallback: preinstall these packages in the image per .ci/docker/requirements-ci.txt; this script skips install when satisfied.
EOF
exit 1