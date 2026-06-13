#!/usr/bin/env bash
# Vendor a subset of the quack library into torch/_vendor/quack.
#
# The pinned upstream commit lives in PINNED_SHA below. To bump the vendored
# version, edit that one line and re-run this script; the SHA is never passed
# on the command line.
#
# Usage:
#   tools/vendoring/quack/vendor.sh                        # re-vendor the pinned SHA
#   tools/vendoring/quack/vendor.sh --src <local-checkout> # reuse an existing clone
#   tools/vendoring/quack/vendor.sh --check [--src <dir>]  # re-render + diff, no writes
#
# Pipeline:
#   1. verify the pinned SHA is reachable from upstream main
#   2. fetch upstream main and check out the pinned SHA
#   3. copy whitelisted modules + LICENSE into torch/_vendor/quack/
#   4. apply tools/vendoring/quack/patches/*.patch
#          (PyTorch vendoring/runtime patches only)
#   5. apply tools/vendoring/quack/flex_gemm_patches/series
#          (FlexGEMM QuACK feature deltas and runtime support patches)
#   6. rewrite `quack.*` imports to package-relative
#   7. verify copyright/license notices still match upstream
#   8. write a fresh __init__.py recording the SHA and upstream version
#
# With --check the subset is rendered into a tempdir and diffed against the
# committed tree instead of overwriting it; a nonzero exit means a vendored file
# drifted from what the patches produce (e.g. a hand-edit that bypassed them).
#
# If a FlexGEMM patch fails, upstream has drifted — inspect the .rej and
# rebase the patchset. If a vendoring patch fails, update only the mechanical
# PyTorch delta. If notice verification fails, a patch moved or removed an
# attribution line — fix the patch rather than the check.

set -euo pipefail

UPSTREAM_URL="https://github.com/Dao-AILab/quack.git"
PINNED_SHA="99bd7973bf3dc6db40961e413d4bdfea6c6fee3e"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DEST="$REPO_ROOT/torch/_vendor/quack"
FLEX_GEMM_PATCHES_DIR="$SCRIPT_DIR/flex_gemm_patches"
PATCHES_DIR="$SCRIPT_DIR/patches"
GITATTRIBUTES="$REPO_ROOT/.gitattributes"
GENERATED_ATTRIBUTE='torch/_vendor/quack/** linguist-generated=true'

# Temp dirs (cloned upstream, --check render) are removed together on exit via a
# single trap; helpers append here rather than each installing their own trap.
CLEANUP_DIRS=()
UPSTREAM_DIR=""
cleanup() {
    local d
    for d in ${CLEANUP_DIRS[@]+"${CLEANUP_DIRS[@]}"}; do
        rm -rf "$d"
    done
}
trap cleanup EXIT

# Modules that rmsnorm and the selected GEMM epilogue implementation paths depend
# on transitively. Everything else upstream ships — softmax, cross-entropy, topk,
# etc. — is deliberately excluded.
PYTORCH_ONLY_FILES=(
    cute_dsl_elf_fix.py
    cute_dsl_mlir_threading.py
)

FLEX_GEMM_CREATED_FILES=()

FILES=(
    _compile_worker.py
    activation.py
    autotuner.py
    bench/__init__.py
    bench/bench_utils.py
    blockscaled_gemm_utils.py
    cache/__init__.py
    cache/compile_only.py
    cache/jit.py
    compile_utils.py
    copy_utils.py
    cute_dsl_utils.py
    epi_composable.py
    epi_ops.py
    epi_utils.py
    fast_math.py
    gemm_act.py
    gemm_base.py
    gemm_blockscaled_interface.py
    gemm_config.py
    gemm_default_epi.py
    gemm_sm100.py
    gemm_sm120.py
    gemm_sm80.py
    gemm_sm90.py
    gemm_tvm_ffi_utils.py
    layout_utils.py
    mx_utils.py
    pipeline.py
    reduce.py
    reduction_base.py
    rmsnorm.py
    rmsnorm_config.py
    rounding.py
    sm100_utils.py
    sm80_utils.py
    sm90_utils.py
    tile_scheduler.py
    trace.py
    utils.py
    varlen_utils.py
)

die()   { echo "vendor_quack: $*" >&2; exit 1; }
usage() { echo "usage: $0 [--check] [--src <local-quack-checkout>]" >&2; exit 2; }

assert_pinned_sha_on_upstream_main() {
    local sha=$1 mainline_dir
    mainline_dir=$(mktemp -d -t quack-mainline-check-XXXXXX)
    CLEANUP_DIRS+=("$mainline_dir")
    git -C "$mainline_dir" init --quiet
    git -C "$mainline_dir" remote add origin "$UPSTREAM_URL"
    git -C "$mainline_dir" fetch --quiet --filter=blob:none \
        origin refs/heads/main:refs/remotes/origin/main
    git -C "$mainline_dir" merge-base --is-ancestor "$sha" refs/remotes/origin/main \
        || die "PINNED_SHA $sha must be reachable from $UPSTREAM_URL main"
}

# Set UPSTREAM_DIR to a quack checkout at $sha. With a local checkout, validate
# it is at the requested SHA. Otherwise fetch upstream main into a tempdir and
# check out the pinned mainline commit.
fetch_upstream() {
    local sha=$1 local_checkout=${2:-}

    assert_pinned_sha_on_upstream_main "$sha"

    if [[ -n "$local_checkout" ]]; then
        local head
        head=$(git -C "$local_checkout" rev-parse HEAD)
        [[ "$head" == "$sha"* || "$sha" == "$head"* ]] \
            || die "$local_checkout is at $head, not $sha"
        UPSTREAM_DIR="$local_checkout"
        return
    fi

    UPSTREAM_DIR=$(mktemp -d -t quack-vendor-XXXXXX)
    CLEANUP_DIRS+=("$UPSTREAM_DIR")
    git -C "$UPSTREAM_DIR" init --quiet
    git -C "$UPSTREAM_DIR" remote add origin "$UPSTREAM_URL"
    git -C "$UPSTREAM_DIR" fetch --quiet --filter=blob:none \
        origin refs/heads/main:refs/remotes/origin/main
    git -C "$UPSTREAM_DIR" checkout --quiet "$sha"
}

extract_version() {
    local init=$1 version
    version=$(sed -n 's/^__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$init")
    [[ -n "$version" ]] || die "could not parse __version__ from $init"
    echo "$version"
}

# Return the pinned upstream commit. This constant is the single, human-edited
# source of truth: the script consumes it and never invents or accepts a SHA on
# the command line.
pinned_sha() {
    [[ "$PINNED_SHA" =~ ^[0-9a-f]{7,40}$ ]] || die "invalid PINNED_SHA: $PINNED_SHA"
    echo "$PINNED_SHA"
}

is_flex_gemm_created_file() {
    local f=$1 created
    for created in "${FLEX_GEMM_CREATED_FILES[@]}"; do
        [[ "$f" == "$created" ]] && return 0
    done
    return 1
}

copy_pristine() {
    local upstream=$1
    for f in "${FILES[@]}"; do
        mkdir -p "$DEST/$(dirname "$f")"
        if [[ ! -f "$upstream/quack/$f" ]] && is_flex_gemm_created_file "$f"; then
            continue
        fi
        cp "$upstream/quack/$f" "$DEST/$f"
    done
    # Apache-2.0 attribution: quack is redistributed under its upstream
    # license, which must accompany the vendored source.
    cp "$upstream/LICENSE" "$DEST/LICENSE"
}

copy_pytorch_only() {
    local f
    for f in "${PYTORCH_ONLY_FILES[@]}"; do
        git -C "$REPO_ROOT" show "HEAD:torch/_vendor/quack/$f" > "$DEST/$f"
    done
}

apply_patch_file() {
    local patch_file=$1
    patch -p1 -d "$DEST" --no-backup-if-mismatch --forward < "$patch_file"
}

apply_patch_dir() {
    local dir=$1 p
    for p in "$dir"/*.patch; do
        [[ -e "$p" ]] || continue
        apply_patch_file "$p"
    done
}

apply_patch_series() {
    local dir=$1 line patch_name p seen_patches=""
    local series="$dir/series"
    [[ -f "$series" ]] || die "missing patch series: $series"
    while IFS= read -r line || [[ -n "$line" ]]; do
        line=${line%%#*}
        patch_name=$(printf "%s" "$line" | sed -E "s/^[[:space:]]+//;s/[[:space:]]+$//")
        [[ -z "$patch_name" ]] && continue
        [[ "$patch_name" == */* ]] && die "patch series entries must be filenames: $patch_name"
        [[ "$patch_name" == *.patch ]] || die "patch series entry must end in .patch: $patch_name"
        [[ -f "$dir/$patch_name" ]] || die "patch listed in $series not found: $patch_name"
        if printf "%s" "$seen_patches" | grep -Fxq "$patch_name"; then
            die "duplicate patch in $series: $patch_name"
        fi
        seen_patches="${seen_patches}${patch_name}"$'\n'
        apply_patch_file "$dir/$patch_name"
    done < "$series"
    for p in "$dir"/*.patch; do
        [[ -e "$p" ]] || continue
        patch_name=$(basename "$p")
        printf "%s" "$seen_patches" | grep -Fxq "$patch_name" \
            || die "patch missing from $series: $patch_name"
    done
}

apply_flex_gemm_patches() {
    apply_patch_series "$FLEX_GEMM_PATCHES_DIR"
}

apply_patches() {
    apply_patch_dir "$PATCHES_DIR"
}

# Rewrite the three `quack.*` import forms actually used in the vendored
# subset. Using [ \t] (not \s) keeps each match on a single line so blank
# lines aren't eaten by the substitution.
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

            # import quack.X.Y as Z       -> from .X import Y as Z
            s|^([ \t]*)import quack\.([[:alnum:]_]+)\.([[:alnum:]_]+) as ([[:alnum:]_]+)[ \t]*$|\1from .\2 import \3 as \4|
        ' "$DEST/$f"
    done

    # The generic rewrite runs relative to torch._vendor.quack, but files inside
    # nested packages need imports relative to their own package.
    sed -i -E '
        s|from \.cache\.jit import |from .jit import |
        s|from \.cache\.compile_only import |from .compile_only import |
    ' "$DEST/cache/__init__.py"
    sed -i -E '
        s|from \. import cache as _state|import torch._vendor.quack.cache as _state|
    ' "$DEST/cache/compile_only.py"
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
        if [[ ! -f "$upstream/quack/$f" ]] && is_flex_gemm_created_file "$f"; then
            continue
        fi
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

has_gitattributes_marker() {
    [[ -f "$GITATTRIBUTES" ]] && grep -Fxq "$GENERATED_ATTRIBUTE" "$GITATTRIBUTES"
}

ensure_gitattributes() {
    if has_gitattributes_marker; then
        return
    fi
    printf '%s\n' "$GENERATED_ATTRIBUTE" >> "$GITATTRIBUTES"
}

assert_gitattributes() {
    has_gitattributes_marker || die "$GITATTRIBUTES must mark torch/_vendor/quack as generated"
}

write_init() {
    local sha=$1 version=$2
    # Heredoc is unquoted so $sha and $version interpolate. The \`\` escapes
    # keep reStructuredText-style ``double backticks`` literal in the output.
    cat > "$DEST/__init__.py" <<EOF
"""Vendored subset of the quack library (https://github.com/Dao-AILab/quack).

The pinned upstream commit is recorded in \`\`__upstream_sha__\`\` below and is
sourced from \`\`PINNED_SHA\`\` in tools/vendoring/quack/vendor.sh. The
vendoring script verifies that commit is reachable from Dao-AILab/quack main
before applying the local FlexGEMM patchset. Only the modules required by
torch._native.ops.norm.rmsnorm_impl and selected GEMM epilogue implementation
paths are vendored. Imports are rewritten to be package-relative
so this copy is independent of any \`\`quack\`\` top-level package that may be
installed via pip. Custom op namespaces are renamed from \`\`quack::\`\` to
\`\`torch_vendor_quack::\`\` for the same reason.
"""
__version__ = "$version"
__upstream_sha__ = "$sha"

# Two CuTeDSL workarounds, both must run before the first cute.compile call:
#   - cutlass#3161: duplicate .text section flags break MCJIT in multi-process
#     loads (see cute_dsl_elf_fix).
#   - cutlass#3062: ir.Context spawns LLVM thread pools that leak across
#     compiles, eventually exhausting pthreads (see cute_dsl_mlir_threading).
from . import cute_dsl_elf_fix
from . import cute_dsl_mlir_threading

cute_dsl_elf_fix.patch()
cute_dsl_mlir_threading.patch()

def __getattr__(name):
    if name == "rmsnorm":
        from .rmsnorm import rmsnorm

        return rmsnorm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "rmsnorm",
]
EOF
}

# Render the vendored subset into $DEST, wiping any previous contents first.
render() {
    local upstream=$1 sha=$2 version=$3
    mkdir -p "$DEST"
    find "$DEST" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    copy_pristine "$upstream"
    copy_pytorch_only
    apply_patches
    apply_flex_gemm_patches
    rewrite_imports
    verify_notices "$upstream"
    write_init "$sha" "$version"
}

# Diff a freshly rendered $DEST against the committed tree; nonzero on drift.
assert_matches() {
    local committed=$1 drift
    if drift=$(diff -r --exclude=__pycache__ "$committed" "$DEST"); then
        echo "OK: re-vendoring reproduces $committed"
        return
    fi
    echo "vendor_quack: re-vendoring does not match $committed:" >&2
    echo "$drift" >&2
    die "edit tools/vendoring/quack patches, not the vendored files"
}

main() {
    local check_only=0 local_checkout=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check) check_only=1; shift ;;
            --src)   [[ $# -ge 2 ]] || usage; local_checkout=$2; shift 2 ;;
            *)       usage ;;
        esac
    done

    local sha version
    sha=$(pinned_sha)
    fetch_upstream "$sha" "$local_checkout"
    version=$(extract_version "$UPSTREAM_DIR/quack/__init__.py")

    if [[ $check_only -eq 0 ]]; then
        render "$UPSTREAM_DIR" "$sha" "$version"
        ensure_gitattributes
        echo "Vendored quack @ $sha (quack $version) into $DEST"
        return
    fi

    local committed=$DEST
    DEST=$(mktemp -d -t quack-vendor-check-XXXXXX)
    CLEANUP_DIRS+=("$DEST")
    render "$UPSTREAM_DIR" "$sha" "$version"
    assert_matches "$committed"
    assert_gitattributes
}

main "$@"
