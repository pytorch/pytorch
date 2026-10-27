#!/bin/bash
# Collect LLVM BOLT profiles for the libraries of the installed torch package.
#
# The installed wheel must be BOLT-compatible but not BOLT-optimized: build it
# with USE_LLVM_BOLT=1 and LLVM_BOLT_PROFILES_DIR pointing to an empty
# directory (see cmake/bolt_profiles/README.md). This script then
#
#   1. downloads llvm-bolt and merge-fdata from an LLVM release,
#   2. instruments every torch library that was linked with --emit-relocs,
#   3. runs tools/bolt/synthetic_workload.py on the instrumented libraries,
#   4. merges the per-process profiles of each library with merge-fdata,
#   5. converts the merged profiles to YAML and validates them with a trial
#      optimization that uses the options of the build-time BOLT step,
#   6. packages the YAML profiles as torch.tar.zst, in the layout of
#      cmake/bolt_profiles/torch.tar.zst.
#
# The installed libraries are restored when the script exits.
#
# Usage: tools/bolt/collect_profiles.sh <output-dir>
#
# Environment:
#   LLVM_BOLT_VERSION          LLVM release to take llvm-bolt from (default
#                              21.1.8, the version the manywheel images build
#                              in .ci/docker/common/install_llvm_tools.sh, so
#                              the profiles match the BOLT that consumes them)
#   LLVM_BOLT_HOME             use the llvm-bolt and merge-fdata of this LLVM
#                              install instead of downloading a release
#   BOLT_WORKLOAD_VARIANT      synthetic_workload.py --variant (default all)
#   BOLT_WORKLOAD_STEPS        synthetic_workload.py --steps (default 20000)
#   BOLT_WORKLOAD_WARMUP       synthetic_workload.py --warmup (default 50)
#   BOLT_WORKLOAD_MAX_SECONDS  synthetic_workload.py --max-seconds (default 0,
#                              no limit)
#   BOLT_VALIDATE              set to 0 to skip the trial optimization

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-dir>" >&2
  exit 1
fi

LLVM_BOLT_VERSION="${LLVM_BOLT_VERSION:-21.1.8}"
BOLT_WORKLOAD_VARIANT="${BOLT_WORKLOAD_VARIANT:-all}"
BOLT_WORKLOAD_STEPS="${BOLT_WORKLOAD_STEPS:-20000}"
BOLT_WORKLOAD_WARMUP="${BOLT_WORKLOAD_WARMUP:-50}"
BOLT_WORKLOAD_MAX_SECONDS="${BOLT_WORKLOAD_MAX_SECONDS:-0}"
BOLT_VALIDATE="${BOLT_VALIDATE:-1}"

# Options of the build-time optimization in torch_optimize_layout_if_enabled
# (cmake/public/utils.cmake). Keep in sync.
BOLT_OPTIMIZE_OPTIONS=(
  -lite -infer-stale-profile
  -reorder-blocks=ext-tsp -reorder-functions=cdsort
  -split-functions -split-all-cold -split-eh -dyno-stats
  --update-debug-sections
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "$1"
OUTPUT_DIR="$(cd "$1" && pwd)"
if [[ -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
  echo "Output directory ${OUTPUT_DIR} is not empty" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"/{fdata,yaml,logs}

WORK_DIR="$(mktemp -d)"
mkdir -p "${WORK_DIR}"/{original,instrumented,raw,bolted}

log() {
  echo "[collect_profiles] $*"
}

die() {
  echo "[collect_profiles] ERROR: $*" >&2
  exit 1
}

# Run llvm-bolt / merge-fdata with output in a log file; show its tail on
# failure.
run_logged() {
  local logfile="$1"
  shift
  if ! "$@" > "${logfile}" 2>&1; then
    tail -n 50 "${logfile}" >&2
    die "command failed, see ${logfile}: $*"
  fi
}

# Libraries to profile, file names relative to TORCH_LIB_DIR.
libs=()

restore_libraries() {
  local lib
  for lib in "${libs[@]}"; do
    if [[ -f "${WORK_DIR}/original/${lib}" ]]; then
      cp "${WORK_DIR}/original/${lib}" "${TORCH_LIB_DIR}/${lib}"
    fi
  done
}

cleanup() {
  restore_libraries
  cd /
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

# Import torch from site-packages, never from a source checkout in the cwd.
cd "${WORK_DIR}"

TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
log "torch libraries: ${TORCH_LIB_DIR}"

# BOLT needs the static relocations that --emit-relocs keeps in .rela.text.
# The build links only the libraries it would BOLT-optimize with it.
for path in "${TORCH_LIB_DIR}"/lib*.so; do
  sections="$(readelf -S --wide "${path}")"
  if [[ "${sections}" != *" .rela.text "* ]]; then
    log "Skipping $(basename "${path}"): no .rela.text"
    continue
  fi
  if [[ "${sections}" == *" .note.bolt_info "* ]]; then
    die "$(basename "${path}") is already BOLT-optimized; build with an empty LLVM_BOLT_PROFILES_DIR"
  fi
  libs+=("$(basename "${path}")")
done
if [[ ${#libs[@]} -eq 0 ]]; then
  die "no library in ${TORCH_LIB_DIR} has .rela.text; build the wheel with USE_LLVM_BOLT=1"
fi
log "libraries to profile: ${libs[*]}"

# ---------------------------------------------------------------------------
# llvm-bolt
# ---------------------------------------------------------------------------

if [[ -n "${LLVM_BOLT_HOME:-}" ]]; then
  LLVM_HOME="${LLVM_BOLT_HOME}"
else
  case "$(uname -m)" in
    aarch64) llvm_arch=ARM64 ;;
    x86_64) llvm_arch=X64 ;;
    *) die "unsupported architecture: $(uname -m)" ;;
  esac
  # Every release ships .tar.xz; newer ones also ship .tar.zst.
  archive="LLVM-${LLVM_BOLT_VERSION}-Linux-${llvm_arch}.tar.xz"
  url="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_BOLT_VERSION}/${archive}"
  log "Downloading ${url}"
  curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 -o "${WORK_DIR}/${archive}" "${url}"
  LLVM_HOME="${WORK_DIR}/llvm"
  mkdir -p "${LLVM_HOME}"
  # Only the BOLT tools and the instrumentation runtime are needed. llvm-bolt
  # looks the runtime up in ../lib relative to its own location.
  tar -x --xz -f "${WORK_DIR}/${archive}" -C "${LLVM_HOME}" \
    --strip-components=1 --wildcards \
    '*/bin/llvm-bolt' '*/bin/merge-fdata' '*/lib/libbolt_rt_instr.a'
  rm -f "${WORK_DIR:?}/${archive}"
fi
LLVM_BOLT="${LLVM_HOME}/bin/llvm-bolt"
MERGE_FDATA="${LLVM_HOME}/bin/merge-fdata"
"${LLVM_BOLT}" --version | tee "${OUTPUT_DIR}/logs/llvm-bolt-version.txt"
"${MERGE_FDATA}" --version

# ---------------------------------------------------------------------------
# Instrument and run the workload
# ---------------------------------------------------------------------------

for lib in "${libs[@]}"; do
  stem="${lib%.so}"
  cp "${TORCH_LIB_DIR}/${lib}" "${WORK_DIR}/original/${lib}"
  log "Instrumenting ${lib}"
  # Every process that loads the library writes <stem>.<pid>.fdata at exit.
  run_logged "${OUTPUT_DIR}/logs/instrument-${stem}.txt" \
    "${LLVM_BOLT}" "${WORK_DIR}/original/${lib}" -instrument \
    --instrumentation-file-append-pid \
    "--instrumentation-file=${WORK_DIR}/raw/${stem}" \
    -o "${WORK_DIR}/instrumented/${lib}"
done
for lib in "${libs[@]}"; do
  cp "${WORK_DIR}/instrumented/${lib}" "${TORCH_LIB_DIR}/${lib}"
done

log "Running the synthetic workload on the instrumented libraries"
python3 "${SCRIPT_DIR}/synthetic_workload.py" \
  --variant "${BOLT_WORKLOAD_VARIANT}" \
  --steps "${BOLT_WORKLOAD_STEPS}" \
  --warmup "${BOLT_WORKLOAD_WARMUP}" \
  --max-seconds "${BOLT_WORKLOAD_MAX_SECONDS}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/workload.txt"

restore_libraries

# ---------------------------------------------------------------------------
# Merge, convert and validate the profiles
# ---------------------------------------------------------------------------

declare -A raw_count
for lib in "${libs[@]}"; do
  stem="${lib%.so}"
  raw=("${WORK_DIR}/raw/${stem}".*.fdata)
  if [[ ! -e "${raw[0]}" ]]; then
    die "no profile was written for ${lib}; did the workload exit normally?"
  fi
  raw_count["${lib}"]="${#raw[@]}"
  log "Merging ${#raw[@]} profile(s) of ${lib}"
  run_logged "${OUTPUT_DIR}/logs/merge-fdata-${stem}.txt" \
    "${MERGE_FDATA}" "${raw[@]}" -o "${OUTPUT_DIR}/fdata/${stem}.fdata"
  if [[ ! -s "${OUTPUT_DIR}/fdata/${stem}.fdata" ]]; then
    die "merge-fdata produced an empty profile for ${lib}"
  fi
done

# The build-time step consumes YAML profiles named lib<target>.yaml. Unlike
# fdata, they carry block hashes that let -infer-stale-profile match profiles
# to binaries built from slightly different sources.
for lib in "${libs[@]}"; do
  stem="${lib%.so}"
  log "Converting the ${lib} profile to YAML"
  run_logged "${OUTPUT_DIR}/logs/convert-${stem}.txt" \
    "${LLVM_BOLT}" "${WORK_DIR}/original/${lib}" -o /dev/null \
    "-data=${OUTPUT_DIR}/fdata/${stem}.fdata" \
    -w "${OUTPUT_DIR}/yaml/${stem}.yaml"
done

if [[ "${BOLT_VALIDATE}" == "1" ]]; then
  quality_args=()
  for lib in "${libs[@]}"; do
    stem="${lib%.so}"
    log "Trial optimization of ${lib}"
    run_logged "${OUTPUT_DIR}/logs/optimize-${stem}.txt" \
      "${LLVM_BOLT}" "${WORK_DIR}/original/${lib}" \
      -o "${WORK_DIR}/bolted/${lib}" \
      "-data=${OUTPUT_DIR}/yaml/${stem}.yaml" \
      "-log-file=${OUTPUT_DIR}/logs/llvm-bolt-${stem}.txt" \
      "${BOLT_OPTIMIZE_OPTIONS[@]}"
    rm -f "${WORK_DIR}/bolted/${lib}"
    quality_args+=(--profile "${OUTPUT_DIR}/yaml/${stem}.yaml" "${OUTPUT_DIR}/logs/llvm-bolt-${stem}.txt")
  done
  python3 "${REPO_ROOT}/tools/bolt_profile_quality.py" "${quality_args[@]}" \
    | tee "${OUTPUT_DIR}/quality_report.txt"
fi

# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------

(
  cd "${OUTPUT_DIR}/yaml"
  if command -v cmake > /dev/null; then
    cmake -E tar cf ../torch.tar.zst --zstd -- *.yaml
  else
    tar --zstd -cf ../torch.tar.zst -- *.yaml
  fi
)

{
  python3 - << 'EOF'
import sysconfig

import torch

print(f"torch: {torch.__version__} (git {torch.version.git_version})")
print(f"cuda: {torch.version.cuda}")
print(f"python: {sysconfig.get_config_var('SOABI')}")
print(f"device: {torch.cuda.get_device_name(0)}")
EOF
  echo "llvm-bolt: $(grep -m1 'LLVM version' "${OUTPUT_DIR}/logs/llvm-bolt-version.txt")"
  echo "workload: --variant ${BOLT_WORKLOAD_VARIANT} --steps ${BOLT_WORKLOAD_STEPS} --warmup ${BOLT_WORKLOAD_WARMUP} --max-seconds ${BOLT_WORKLOAD_MAX_SECONDS}"
  grep -E '^\[bolt-synth\] (variants=|done |stopping |WARNING: dropping)' \
    "${OUTPUT_DIR}/logs/workload.txt" || true
  echo
  for lib in "${libs[@]}"; do
    stem="${lib%.so}"
    echo "${lib}:"
    echo "  sha256: $(sha256sum "${WORK_DIR}/original/${lib}" | cut -d' ' -f1)"
    echo "  build-id: $(readelf -n "${WORK_DIR}/original/${lib}" | awk '/Build ID/ {print $3}')"
    echo "  merged profiles: ${raw_count[${lib}]}"
    echo "  $(grep -m1 'have non-empty execution profile' "${OUTPUT_DIR}/logs/convert-${stem}.txt" || echo 'profiled functions: n/a')"
  done
} > "${OUTPUT_DIR}/manifest.txt"

cat "${OUTPUT_DIR}/manifest.txt"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" && -f "${OUTPUT_DIR}/quality_report.txt" ]]; then
  {
    echo '### BOLT profiles'
    echo '```'
    cat "${OUTPUT_DIR}/manifest.txt"
    echo
    cat "${OUTPUT_DIR}/quality_report.txt"
    echo '```'
  } >> "${GITHUB_STEP_SUMMARY}"
fi
log "Profiles written to ${OUTPUT_DIR}"
