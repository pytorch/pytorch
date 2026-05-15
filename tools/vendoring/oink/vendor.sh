#!/usr/bin/env bash
# Vendor a subset of the kernelagent_oink library into torch/_vendor/oink.
#
# Usage:
#   tools/vendoring/oink/vendor.sh <sha>                    # clone upstream
#   tools/vendoring/oink/vendor.sh <sha> <local-checkout>   # use existing clone
#
# Pipeline:
#   1. fetch upstream at <sha>
#   2. copy whitelisted modules + LICENSE into torch/_vendor/oink/
#   3. apply tools/vendoring/oink/patches/*.patch (if any)
#   4. rewrite `kernelagent_oink.blackwell.*` imports to package-relative
#   5. verify copyright/license notices still match upstream
#   6. write a fresh __init__.py recording the SHA and upstream version
#
# If a patch fails, upstream has drifted -- inspect the .rej and re-roll.
# If notice verification fails, a patch moved or removed an attribution
# line -- fix the patch rather than the check.

set -euo pipefail

UPSTREAM_URL="https://github.com/meta-pytorch/KernelAgent.git"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DEST="$REPO_ROOT/torch/_vendor/oink"
PATCHES_DIR="$SCRIPT_DIR/patches"

# Path inside the upstream checkout where the blackwell modules live.
UPSTREAM_SRC_SUBDIR="oink/src/kernelagent_oink/blackwell"

# Modules that rmsnorm depends on (transitively). Everything else upstream
# ships -- cross_entropy, softmax, layernorm, oink_custom_ops, etc. -- is
# deliberately excluded.
FILES=(
    _cutedsl_cache.py
    _rmsnorm_impl.py
    _rmsnorm_simple_weightonly.py
    _rmsnorm_smallm_cuda.py
    fast_launch.py
    lite_quack.py
    rmsnorm.py
)

die()   { echo "vendor_oink: $*" >&2; exit 1; }
usage() { echo "usage: $0 <sha> [local-kernelagent-checkout]" >&2; exit 2; }

# Echo the path to a kernelagent checkout at $sha. If $local is given,
# validate it's at the requested SHA; otherwise clone into a tmpdir and
# register a cleanup trap on the caller's shell.
fetch_upstream() {
    local sha=$1 local_checkout=${2:-}

    if [[ -n "$local_checkout" ]]; then
        local head
        head=$(git -C "$local_checkout" rev-parse HEAD)
        [[ "$head" == "$sha"* || "$sha" == "$head"* ]] \
            || die "$local_checkout is at $head, not $sha"
        echo "$local_checkout"
        return
    fi

    local tmp
    tmp=$(mktemp -d -t oink-vendor-XXXXXX)
    # shellcheck disable=SC2064  # expand $tmp now, not at trap time
    trap "rm -rf '$tmp'" EXIT
    git clone --quiet "$UPSTREAM_URL" "$tmp"
    git -C "$tmp" checkout --quiet "$sha"
    echo "$tmp"
}

extract_version() {
    local pyproject=$1 version
    version=$(sed -n 's/^version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$pyproject" | head -1)
    [[ -n "$version" ]] || die "could not parse version from $pyproject"
    echo "$version"
}

copy_pristine() {
    local upstream=$1
    for f in "${FILES[@]}"; do
        cp "$upstream/$UPSTREAM_SRC_SUBDIR/$f" "$DEST/$f"
    done
    # Apache-2.0 attribution: kernelagent_oink is redistributed under its
    # upstream license, which must accompany the vendored source.
    cp "$upstream/LICENSE" "$DEST/LICENSE"
}

apply_patches() {
    shopt -s nullglob
    for p in "$PATCHES_DIR"/*.patch; do
        patch -p1 -d "$DEST" --no-backup-if-mismatch --forward < "$p"
    done
    shopt -u nullglob
}

# Rewrite the three `kernelagent_oink.blackwell.*` import forms actually
# used in the vendored subset. Using [ \t] (not \s) keeps each match on a
# single line so blank lines aren't eaten by the substitution.
rewrite_imports() {
    for f in "${FILES[@]}"; do
        sed -i -E '
            # from kernelagent_oink.blackwell.X import Y -> from .X import Y
            s|^([ \t]*)from kernelagent_oink\.blackwell\.([[:alnum:]_.]+) import |\1from .\2 import |

            # from kernelagent_oink.blackwell import X as Y -> from . import X as Y
            s|^([ \t]*)from kernelagent_oink\.blackwell import ([[:alnum:]_]+) as ([[:alnum:]_]+)[ \t]*$|\1from . import \2 as \3|

            # from kernelagent_oink.blackwell import X -> from . import X
            s|^([ \t]*)from kernelagent_oink\.blackwell import |\1from . import |
        ' "$DEST/$f"
    done
}

# Guard against patches or import rewrites accidentally dropping or
# relocating a copyright/license/SPDX line. Each vendored .py must carry
# the same notice lines on the same line numbers as its upstream source.
# Bails on the first mismatch so the operator can inspect before the
# commit lands.
verify_notices() {
    local upstream=$1
    local pattern='[Cc]opyright|[Ll]icense|SPDX|[Aa]ll [Rr]ights [Rr]eserved'
    for f in "${FILES[@]}"; do
        if ! diff -u \
                <(grep -nE "$pattern" "$upstream/$UPSTREAM_SRC_SUBDIR/$f" || true) \
                <(grep -nE "$pattern" "$DEST/$f" || true) \
                > /dev/null; then
            echo "vendor_oink: notice drift in $f:" >&2
            diff -u \
                <(grep -nE "$pattern" "$upstream/$UPSTREAM_SRC_SUBDIR/$f" || true) \
                <(grep -nE "$pattern" "$DEST/$f" || true) >&2 || true
            die "attribution must match upstream byte-for-byte; fix the patch"
        fi
    done
    cmp -s "$upstream/LICENSE" "$DEST/LICENSE" \
        || die "LICENSE differs from upstream"
}

write_init() {
    local sha=$1 version=$2
    # Heredoc is unquoted so $sha and $version interpolate. The \`\` escapes
    # keep reStructuredText-style ``double backticks`` literal in the output.
    cat > "$DEST/__init__.py" <<EOF
"""Vendored subset of the kernelagent_oink library
(https://github.com/meta-pytorch/KernelAgent).

Upstream SHA: $sha (kernelagent-oink $version)

Only the modules required by torch._native.ops.norm.oink_rmsnorm_impl are
vendored. Imports are rewritten to be package-relative so this copy is
independent of any \`\`kernelagent_oink\`\` top-level package that may be
installed via pip.
"""
__version__ = "$version"

from .rmsnorm import rmsnorm_forward, rmsnorm_backward  # noqa: E402


__all__ = [
    "rmsnorm_forward",
    "rmsnorm_backward",
]
EOF
}

main() {
    [[ $# -eq 1 || $# -eq 2 ]] || usage
    local sha=$1 local_checkout=${2:-} upstream version

    upstream=$(fetch_upstream "$sha" "$local_checkout")
    version=$(extract_version "$upstream/oink/pyproject.toml")

    mkdir -p "$DEST"
    rm -f "$DEST"/*.py "$DEST/LICENSE"

    copy_pristine "$upstream"
    apply_patches
    rewrite_imports
    verify_notices "$upstream"
    write_init "$sha" "$version"

    echo "Vendored kernelagent-oink @ $sha (kernelagent-oink $version) into torch/_vendor/oink"
}

main "$@"
