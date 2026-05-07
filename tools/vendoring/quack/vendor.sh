#!/usr/bin/env bash
# Vendor a subset of the quack library into torch/_vendor/quack.
#
# Usage:
#   tools/vendoring/quack/vendor.sh <sha>                    # clone upstream
#   tools/vendoring/quack/vendor.sh <sha> <local-checkout>   # use existing clone
#
# Pipeline:
#   1. fetch upstream at <sha>
#   2. copy whitelisted modules + LICENSE into torch/_vendor/quack/
#   3. apply tools/vendoring/quack/patches/*.patch
#          (strip torch.library decorators, rename branded strings)
#   4. rewrite `quack.*` imports to package-relative
#   5. verify copyright/license notices still match upstream
#   6. write a fresh __init__.py recording the SHA and upstream version
#
# If a patch fails, upstream has drifted — inspect the .rej and re-roll.
# If notice verification fails, a patch moved or removed an attribution
# line — fix the patch rather than the check.

set -euo pipefail

UPSTREAM_URL="https://github.com/Dao-AILab/quack.git"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DEST="$REPO_ROOT/torch/_vendor/quack"
PATCHES_DIR="$SCRIPT_DIR/patches"

# Transitive closure of the modules required by rmsnorm and the gemm
# family. Everything else upstream ships — softmax, cross-entropy, rotary,
# topk, linear, etc. — is deliberately excluded.
FILES=(
    # rmsnorm + shared support
    cache_utils.py
    compile_utils.py
    copy_utils.py
    cute_dsl_elf_fix.py
    cute_dsl_mlir_threading.py
    cute_dsl_utils.py
    layout_utils.py
    reduce.py
    reduction_base.py
    rmsnorm.py
    rounding.py
    utils.py

    # gemm family + transitive deps
    activation.py
    autotuner.py
    blockscaled_gemm_utils.py
    epi_composable.py
    epi_ops.py
    epi_utils.py
    fast_math.py
    gemm.py
    gemm_act.py
    gemm_base.py
    gemm_blockscaled_interface.py
    gemm_config.py
    gemm_dact.py
    gemm_default_epi.py
    gemm_interface.py
    gemm_norm_act.py
    gemm_sm80.py
    gemm_sm90.py
    gemm_sm100.py
    gemm_sm120.py
    gemm_sq_reduce.py
    gemm_symmetric.py
    gemm_tvm_ffi_utils.py
    mx_utils.py
    nvmmh_heuristic.py
    pipeline.py
    rms_final_reduce.py
    sm90_utils.py
    sm100_utils.py
    tile_scheduler.py
    trace.py
    varlen_utils.py
)

die()   { echo "vendor_quack: $*" >&2; exit 1; }
usage() { echo "usage: $0 <sha> [local-quack-checkout]" >&2; exit 2; }

# Echo the path to a quack checkout at $sha. If $local is given, validate
# it's at the requested SHA; otherwise clone into a tmpdir and register a
# cleanup trap on the caller's shell.
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
    tmp=$(mktemp -d -t quack-vendor-XXXXXX)
    # shellcheck disable=SC2064  # expand $tmp now, not at trap time
    trap "rm -rf '$tmp'" EXIT
    git clone --quiet "$UPSTREAM_URL" "$tmp"
    git -C "$tmp" checkout --quiet "$sha"
    echo "$tmp"
}

extract_version() {
    local init=$1 version
    version=$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$init")
    [[ -n "$version" ]] || die "could not parse __version__ from $init"
    echo "$version"
}

copy_pristine() {
    local upstream=$1
    for f in "${FILES[@]}"; do
        cp "$upstream/quack/$f" "$DEST/$f"
    done
    # Apache-2.0 attribution: quack is redistributed under its upstream
    # license, which must accompany the vendored source.
    cp "$upstream/LICENSE" "$DEST/LICENSE"
}

apply_patches() {
    for p in "$PATCHES_DIR"/*.patch; do
        patch -p1 -d "$DEST" --no-backup-if-mismatch --forward < "$p"
    done
}

# Rewrite the four `quack.*` import forms used in the vendored subset.
# Using [ \t] (not \s) keeps each match on a single line so blank lines
# aren't eaten by the substitution. The "alias == module" rule comes
# before the general "alias != module" rule so the redundant alias is
# collapsed rather than preserved.
rewrite_imports() {
    for f in "${FILES[@]}"; do
        sed -i -E '
            # from quack.X import Y       -> from .X import Y
            s|^([ \t]*)from quack\.([[:alnum:]_.]+) import |\1from .\2 import |

            # from quack import X         -> from . import X
            s|^([ \t]*)from quack import |\1from . import |

            # import quack.X as X         -> from . import X   (drop redundant alias)
            s|^([ \t]*)import quack\.([[:alnum:]_]+) as \2[ \t]*$|\1from . import \2|

            # import quack.X as Y         -> from . import X as Y
            s|^([ \t]*)import quack\.([[:alnum:]_]+) as ([[:alnum:]_]+)[ \t]*$|\1from . import \2 as \3|
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
                <(grep -nE "$pattern" "$upstream/quack/$f" || true) \
                <(grep -nE "$pattern" "$DEST/$f" || true) \
                > /dev/null; then
            echo "vendor_quack: notice drift in $f:" >&2
            diff -u \
                <(grep -nE "$pattern" "$upstream/quack/$f" || true) \
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
"""Vendored subset of the quack library (https://github.com/Dao-AILab/quack).

Upstream SHA: $sha (quack $version)

Only the modules required by torch._native (rmsnorm and the gemm family)
are vendored. softmax, cross-entropy, rotary, topk, linear, etc. are
deliberately excluded. Imports within the vendored tree are rewritten to
be package-relative so this copy is independent of any \`\`quack\`\`
top-level package that may be installed via pip, and \`\`torch.library\`\`
op registrations are stripped so the vendored copy does not claim the
\`\`quack::\`\` namespace at import time.

Gemm entry points live in \`\`torch._vendor.quack.gemm_interface\`\` — they
are not re-exported here to keep this module's import footprint small.
"""
__version__ = "$version"

# Two CuTeDSL workarounds, both must run before the first cute.compile call:
#   - cutlass#3161: duplicate .text section flags break MCJIT in multi-process
#     loads (see cute_dsl_elf_fix).
#   - cutlass#3062: ir.Context spawns LLVM thread pools that leak across
#     compiles, eventually exhausting pthreads (see cute_dsl_mlir_threading).
from . import cute_dsl_elf_fix
from . import cute_dsl_mlir_threading

cute_dsl_elf_fix.patch()
cute_dsl_mlir_threading.patch()

from .rmsnorm import rmsnorm  # noqa: E402


__all__ = [
    "rmsnorm",
]
EOF
}

main() {
    [[ $# -eq 1 || $# -eq 2 ]] || usage
    local sha=$1 local_checkout=${2:-} upstream version

    upstream=$(fetch_upstream "$sha" "$local_checkout")
    version=$(extract_version "$upstream/quack/__init__.py")

    mkdir -p "$DEST"
    rm -f "$DEST"/*.py "$DEST/LICENSE"

    copy_pristine "$upstream"
    apply_patches
    rewrite_imports
    verify_notices "$upstream"
    write_init "$sha" "$version"

    echo "Vendored quack @ $sha (quack $version) into torch/_vendor/quack"
}

main "$@"
