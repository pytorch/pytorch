#!/usr/bin/env bash

set -e

EXTERNAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

refDef=refs/heads/main
ref=$refDef
upstreamDef=https://github.com/KhronosGroup/glslang
upstream=$upstreamDef

while [[ "$#" -gt 0 ]]; do
  case $1 in
  -h | --help) help=1 ;;
  --release)
    ref="refs/tags/$2"
    shift
    ;;
  --ref)
    ref="$2"
    shift
    ;;
  --upstream)
    upstream="$2"
    shift
    ;;
  --do-commit)
    do_commit=1
    ;;
  --do-fetch)
    do_fetch=1
    ;;
  *)
    echo "Unknown parameter passed: $1" >&2
    exit 1
    ;;
  esac
  shift
done

if [ "$help" ]; then
  me=$(basename "$0")
  cat <<EOF
$me: Update external/glslang and dependencies

- Merge the latest upstream glslang (or a specified commit)
- Checkout the 'known good' revision of and spirv-headers
- Merge the 'known_good' revision of spirv-tools with our changes
- Regenerate the contents of external/glslang-generated and external/spirv-tools-generated
- Optionally commit the changes

Options:
  --do-fetch    : Fetch new changes to glslang spirv-tools spirv-headers, if
                  this isn't specified then the ref/release/upstream options do
                  nothing

  --ref 2b2523f : merge this specific commit into our branch
                  defaults to $refDef

  --release foo : short for --ref refs/tags/foo

  --upstream https://example.com/glslang
                : Specify the url of the repo from which to merge
                  defaults to $upstreamDef

  --do-commit   : Commit the changes to git

EOF
  exit
fi

big_msg() {
  echo
  echo "################################################################"
  echo "$1"
  echo "################################################################"
}

require_bin() {
  if ! command -v "$1" &>/dev/null; then
    echo "This script needs $1, but it isn't in \$PATH"
    missing_bin=1
  fi
}
require_bin "jq"
require_bin "git"
require_bin "python"
require_bin "cmake"
if [ "$missing_bin" ]; then
  exit 1
fi

glslang=$EXTERNAL_DIR/glslang
glslang_generated=$EXTERNAL_DIR/glslang-generated
spirv_headers=$EXTERNAL_DIR/spirv-headers
spirv_tools=$EXTERNAL_DIR/spirv-tools
spirv_tools_generated=$EXTERNAL_DIR/spirv-tools-generated
effcee=$spirv_tools/external/effcee
absl=$spirv_tools/external/effcee/third_party/abseil_cpp
re2=$spirv_tools/external/re2

if ! test -f "$glslang/.git"; then
  echo "$glslang doesn't appear to be a git repo"
  echo "Perhaps you need to 'git submodule update --init'"
  exit 1
fi

known_good_commit() {
  jq <"$glslang/known_good.json" \
    ".commits | .[] | select(.name == \"$1\") | .commit" \
    --raw-output
}

bump_dep() {
  commit=$(known_good_commit "$2")
  big_msg "Fetching $commit from origin in $1"
  git -C "$1" fetch origin "$commit"
  big_msg "Checking out $commit in $1"
  git -C "$1" checkout "$commit"
}

declare -A old_ref
declare -A new_ref
merge_dep() {
  name=$1
  dir=$2
  up=$3
  r=$4
  old_ref["$1"]=$(git -C "$dir" describe --exclude master-tot --tags HEAD)
  big_msg "Fetching $name $r from $up"
  git -C "$dir" fetch --tags --force "$up"
  git -C "$dir" fetch "$up" "$r"
  big_msg "Merging $r into our branch"
  git -C "$dir" merge --no-edit FETCH_HEAD
  new_ref["$1"]=$(git -C "$dir" describe --exclude master-tot --tags HEAD)
}

if [ "$do_fetch" ]; then
  merge_dep glslang "$glslang" "$upstream" "$ref"

  spirv_tools_upstream=https://github.com/$(
    jq <"$glslang/known_good.json" \
      ".commits | .[] | select(.name == \"spirv-tools\") | .subrepo" \
      --raw-output
  )

  merge_dep spirv-tools "$spirv_tools" "$spirv_tools_upstream" "$(known_good_commit "spirv-tools")"

  bump_dep "$spirv_headers" "spirv-tools/external/spirv-headers"
fi

# Make sure we have the dependencies of spirv-tools up to date

test -d "$effcee" || git clone https://github.com/google/effcee.git "$effcee"
git -C "$effcee" pull

test -d "$absl" || git clone https://github.com/abseil/abseil-cpp "$absl"
git -C "$absl" pull

test -d "$re2" || git clone https://github.com/google/re2.git "$re2"
git -C "$re2" pull

rm -rf "$spirv_tools/external/spirv-headers"
ln -s "$spirv_headers" "$spirv_tools/external/spirv-headers"

#
# We have now checked out everything, time to generate the files for
# glslang-generated and friends.
#

big_msg "Generating files for glslang-generated"
set -x
(cd "$glslang" && python build_info.py . -i build_info.h.tmpl -o "$glslang_generated"/glslang/build_info.h)
set +x

big_msg "Generating files for spirv-tools"
build="$spirv_tools/build"
mkdir -p "$build"
echo "Building spirv-tools"
cmake -Wno-dev -B "$build" "$spirv_tools"
# These are the targets which generate .inc and .h files (just what we need,
# specify them here to avoid a lengthy compile)
cmake --build "$build" \
  --target spirv-tools-build-version \
  --target core_tables \
  --target enum_string_mapping \
  --target extinst_tables
echo "Replacing existing .inc and .h files in $spirv_tools_generated"
set -x
rm -f "$spirv_tools_generated/*.{inc,h}"
cp --target-directory "$spirv_tools_generated" "$build"/*.{inc,h}
set +x

if [ "$do_commit" ]; then
  big_msg "Committing changes"
  msg=$(
    cat <<EOF
external/glslang: ${old_ref["glslang"]} -> ${new_ref["glslang"]} 

external/spirv-tools: ${old_ref["spirv-tools"]} -> ${new_ref["spirv-tools"]}"
EOF
  )

  git commit \
    --message "$msg" \
    "$glslang" \
    "$glslang_generated" \
    "$spirv_headers" \
    "$spirv_tools" \
    "$spirv_tools_generated"

else

  cat <<EOFF
Commit these changes with:
  msg=\$(cat <<EOF
external/glslang: ${old_ref["glslang"]} -> ${new_ref["glslang"]} 

external/spirv-tools: ${old_ref["spirv-tools"]} -> ${new_ref["spirv-tools"]}" 
EOF
)
  git commit \\
    --message "\$msg"
    "$glslang" \\
    "$glslang_generated" \\
    "$spirv_headers" \\
    "$spirv_tools" \\
    "$spirv_tools_generated"

EOFF

fi

echo "Please also don't forget to push the new commits in external/glslang and"
echo "external/spirv-tools to our forks!"
