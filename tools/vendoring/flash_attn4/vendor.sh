#!/usr/bin/env bash
# Vendor the FlashAttention-4 CuTe implementation and its QuACK utility subset.

set -euo pipefail

FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention.git"
FLASH_ATTN_SHA="5835c733e7e9c07606b045255768e8a7e9e851bd"
FLASH_ATTN_TAG="fa4-v4.0.0.beta21"
FLASH_ATTN_VERSION="4.0.0b21"
QUACK_URL="https://github.com/Dao-AILab/quack.git"
QUACK_SHA="06eea2eda29e36c7a861a17237e06380b46f03b8"
QUACK_TAG="v0.5.3"
QUACK_VERSION="0.5.3"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
DEST="$REPO_ROOT/torch/_vendor/flash_attn"
GITATTRIBUTES="$REPO_ROOT/.gitattributes"
GENERATED_ATTRIBUTE='torch/_vendor/flash_attn/** linguist-generated=true'

FLASH_ATTN_FILES=(
    __init__.py
    ampere_helpers.py
    barrier.py
    blackwell_helpers.py
    block_info.py
    block_sparse_utils.py
    block_sparsity.py
    cache_utils.py
    copy_utils.py
    cute_dsl_ptxas.py
    cute_dsl_utils.py
    fa_logging.py
    fast_math.py
    flash_bwd.py
    flash_bwd_mla_dk_sm100.py
    flash_bwd_mla_dq_dqv_sm100.py
    flash_bwd_mla_sm100.py
    flash_bwd_postprocess.py
    flash_bwd_preprocess.py
    flash_bwd_sm100.py
    flash_bwd_sm120.py
    flash_bwd_sm90.py
    flash_fwd.py
    flash_fwd_combine.py
    flash_fwd_mla_sm100.py
    flash_fwd_sm100.py
    flash_fwd_sm120.py
    flash_fwd_sm90.py
    interface.py
    mask.py
    mma_sm100_desc.py
    named_barrier.py
    pack_gqa.py
    paged_kv.py
    pipeline.py
    seqlen_info.py
    sm100_hd256_2cta_fmha_backward.py
    sm100_hd256_2cta_fmha_backward_dkdvkernel.py
    sm100_hd256_2cta_fmha_backward_dqkernel.py
    sm100_hd256_2cta_fmha_forward.py
    softmax.py
    tile_scheduler.py
    topk_gather_kv.py
    utils.py
)

QUACK_FILES=(
    activation.py
    compile_utils.py
    copy_utils.py
    cute_dsl_utils.py
    layout_utils.py
    rounding.py
    sm90_utils.py
    utils.py
)

CLEANUP_DIRS=()
FLASH_ATTN_DIR=""
QUACK_DIR=""
cleanup() {
    local dir
    for dir in ${CLEANUP_DIRS[@]+"${CLEANUP_DIRS[@]}"}; do
        rm -rf "$dir"
    done
}
trap cleanup EXIT

die() { echo "vendor_flash_attn4: $*" >&2; exit 1; }
usage() {
    echo "usage: $0 [--check] [--src-fa <flash-attention-checkout>] [--src-quack <quack-checkout>]" >&2
    exit 2
}

fetch_repo() {
    local url=$1 tag=$2 destination_var=$3 local_checkout=${4:-} checkout
    if [[ -n "$local_checkout" ]]; then
        checkout=$local_checkout
    else
        checkout=$(mktemp -d -t flash-attn4-vendor-XXXXXX)
        CLEANUP_DIRS+=("$checkout")
        git -C "$checkout" init --quiet
        git -C "$checkout" remote add origin "$url"
        git -C "$checkout" fetch --quiet --filter=blob:none origin "refs/tags/$tag:refs/tags/$tag"
    fi
    printf -v "$destination_var" '%s' "$checkout"
}

verify_pin() {
    local checkout=$1 tag=$2 expected_sha=$3 actual_sha
    actual_sha=$(git -C "$checkout" rev-parse "$tag^{commit}") \
        || die "$tag is unavailable in $checkout"
    [[ "$actual_sha" == "$expected_sha" ]] \
        || die "$tag resolves to $actual_sha, expected $expected_sha"
}

copy_sources() {
    local file
    mkdir -p "$DEST/cute" "$DEST/quack"
    for file in "${FLASH_ATTN_FILES[@]}"; do
        git -C "$FLASH_ATTN_DIR" show "$FLASH_ATTN_SHA:flash_attn/cute/$file" > "$DEST/cute/$file"
    done
    for file in LICENSE AUTHORS; do
        git -C "$FLASH_ATTN_DIR" show "$FLASH_ATTN_SHA:flash_attn/cute/$file" > "$DEST/$file"
    done
    for file in "${QUACK_FILES[@]}"; do
        git -C "$QUACK_DIR" show "$QUACK_SHA:quack/$file" > "$DEST/quack/$file"
    done
    git -C "$QUACK_DIR" show "$QUACK_SHA:LICENSE" > "$DEST/quack/LICENSE"
}

rewrite_imports() {
    local file original
    for file in "${FLASH_ATTN_FILES[@]}"; do
        sed -i -E '
            s|^([[:space:]]*)from flash_attn\.cute\.([[:alnum:]_]+) import |\1from .\2 import |
            s|^([[:space:]]*)from flash_attn\.cute import |\1from . import |
            s|^([[:space:]]*)import flash_attn\.cute\.([[:alnum:]_]+) as ([[:alnum:]_]+)|\1from . import \2 as \3|
            s|^([[:space:]]*)from quack\.([[:alnum:]_]+) import |\1from ..quack.\2 import |
            s|^([[:space:]]*)from quack import |\1from ..quack import |
            s|^([[:space:]]*)import quack\.activation|\1from ..quack import activation as quack_activation|
            s|quack\.activation\.|quack_activation.|g
            s|# type: ignore \[|# type: ignore[|g
            s|[[:space:]]+$||
        ' "$DEST/cute/$file"
    done
    sed -i 's|^from \.testing import is_fake_mode$|from .runtime_utils import is_fake_mode|' \
        "$DEST/cute/interface.py"
    for file in "${QUACK_FILES[@]}"; do
        original=$(mktemp)
        CLEANUP_DIRS+=("$original")
        cp "$DEST/quack/$file" "$original"
        sed -i -E '
            s|^([[:space:]]*)from quack\.([[:alnum:]_]+) import |\1from .\2 import |
            s|^([[:space:]]*)from quack import |\1from . import |
        ' "$DEST/quack/$file"
        if ! cmp -s "$original" "$DEST/quack/$file"; then
            sed -i '1i# Modified by PyTorch to use package-relative imports for vendoring.' \
                "$DEST/quack/$file"
        fi
        rm "$original"
    done
}

write_init_files() {
    cat > "$DEST/__init__.py" <<'EOF'
"""Vendored FlashAttention-4 dependency universe."""
EOF
    cat > "$DEST/cute/__init__.py" <<EOF
"""Vendored FlashAttention-4 CuTe implementation."""

__version__ = "$FLASH_ATTN_VERSION"
__upstream_sha__ = "$FLASH_ATTN_SHA"
EOF
    cat > "$DEST/cute/runtime_utils.py" <<'EOF'
"""Runtime-only utilities extracted from upstream FA4 test helpers."""

from torch._guards import active_fake_mode


def is_fake_mode() -> bool:
    return active_fake_mode() is not None
EOF
    cat > "$DEST/quack/__init__.py" <<EOF
"""QuACK utilities pinned for the vendored FlashAttention-4 implementation."""

__version__ = "$QUACK_VERSION"
__upstream_sha__ = "$QUACK_SHA"
EOF
}

assert_imports_rewritten() {
    if grep -REn '^[[:space:]]*(from|import)[[:space:]]+(flash_attn|quack)([.[:space:]]|$)' "$DEST/cute" "$DEST/quack"; then
        die "vendored sources contain external flash_attn or quack imports"
    fi
}

assert_relative_import_closure() {
    python3 - "$DEST" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
allowed_import_roots = {
    "ctypes", "cuda", "cutlass", "dataclasses", "enum", "fcntl", "functools",
    "getpass", "hashlib", "inspect", "logging", "math", "operator", "os",
    "pathlib", "pickle", "re", "subprocess", "sys", "tempfile", "time",
    "torch", "triton", "tvm_ffi", "types", "typing", "typing_extensions",
}
missing = []
absolute_import_roots = set()
for source in root.rglob("*.py"):
    package = source.parent
    for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
        if isinstance(node, ast.Import):
            absolute_import_roots.update(alias.name.split(".")[0] for alias in node.names)
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module:
                absolute_import_roots.add(node.module.split(".")[0])
            continue
        target = package
        for _ in range(node.level - 1):
            target = target.parent
        modules = [node.module] if node.module else [alias.name for alias in node.names]
        for module in modules:
            path = target.joinpath(*module.split("."))
            if not path.with_suffix(".py").is_file() and not (path / "__init__.py").is_file():
                missing.append(f"{source.relative_to(root)}: {'.' * node.level}{module}")
if missing:
    raise SystemExit("Missing vendored relative imports:\n" + "\n".join(missing))
unexpected = absolute_import_roots - allowed_import_roots
if unexpected:
    raise SystemExit("Unexpected vendored import roots: " + ", ".join(sorted(unexpected)))
PY
}

ensure_gitattributes() {
    grep -Fxq "$GENERATED_ATTRIBUTE" "$GITATTRIBUTES" \
        || printf '%s\n' "$GENERATED_ATTRIBUTE" >> "$GITATTRIBUTES"
}

render() {
    rm -rf "$DEST"
    mkdir -p "$DEST"
    copy_sources
    rewrite_imports
    write_init_files
    assert_imports_rewritten
    assert_relative_import_closure
}

assert_matches() {
    local committed=$1
    diff -r --exclude=__pycache__ "$committed" "$DEST" \
        || die "re-vendoring does not match $committed"
    grep -Fxq "$GENERATED_ATTRIBUTE" "$GITATTRIBUTES" \
        || die "$GITATTRIBUTES must mark the vendored package as generated"
}

main() {
    local check_only=0 fa_checkout="" quack_checkout=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check) check_only=1; shift ;;
            --src-fa) [[ $# -ge 2 ]] || usage; fa_checkout=$2; shift 2 ;;
            --src-quack) [[ $# -ge 2 ]] || usage; quack_checkout=$2; shift 2 ;;
            *) usage ;;
        esac
    done

    fetch_repo "$FLASH_ATTN_URL" "$FLASH_ATTN_TAG" FLASH_ATTN_DIR "$fa_checkout"
    fetch_repo "$QUACK_URL" "$QUACK_TAG" QUACK_DIR "$quack_checkout"
    verify_pin "$FLASH_ATTN_DIR" "$FLASH_ATTN_TAG" "$FLASH_ATTN_SHA"
    verify_pin "$QUACK_DIR" "$QUACK_TAG" "$QUACK_SHA"

    if [[ $check_only -eq 0 ]]; then
        render
        ensure_gitattributes
        echo "Vendored FA4 $FLASH_ATTN_VERSION and QuACK $QUACK_VERSION into $DEST"
        return
    fi

    local committed=$DEST
    DEST=$(mktemp -d -t flash-attn4-vendor-check-XXXXXX)
    CLEANUP_DIRS+=("$DEST")
    render
    assert_matches "$committed"
}

main "$@"
