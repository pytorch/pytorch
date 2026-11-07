#!/bin/bash
# Collect LLVM BOLT profiles for the torch libraries of an aarch64 CUDA
# nightly build from its unoptimized libraries, which it uploads as the
# prebolt-cuda-aarch64-<cuda> artifact: libc10.so, libc10_cuda.so,
# libtorch_cpu.so, libtorch_cuda.so and a <SOABI>/libtorch_python.so for each
# Python version. Only the run step needs a GPU machine:
#
#   instrument <prebolt-dir> <out-dir>
#       Instrument the libraries of <prebolt-dir>.
#   run <instrumented-dir> <runs-dir>
#       Copy the instrumented libraries over those of the torch installed for
#       python3, a wheel of the same build, run tools/bolt/synthetic_workload.py
#       and collect the per-process profiles in <runs-dir>/<SOABI>. Run it with
#       each Python version. The installed libraries are restored on exit.
#   finalize <prebolt-dir> <runs-dir> <out-dir>
#       Merge the profiles of each library with merge-fdata, convert them to
#       YAML, validate them with a trial optimization that uses the options of
#       the build-time BOLT step, and package them as torch.tar.zst in the
#       layout of cmake/bolt_profiles/torch.tar.zst. libtorch_python-<SOABI>.yaml
#       comes from the run of that Python version, the other profiles and
#       libtorch_python.yaml from the run of BOLT_SHARED_PROFILE_SOABI.
#
# Environment:
#   LLVM_BOLT_HOME             LLVM install with llvm-bolt, merge-fdata and the
#                              instrumentation runtime (default /usr/local, where
#                              .ci/docker/common/install_llvm_tools.sh puts the
#                              BOLT that optimizes the nightly)
#   BOLT_FDATA_DIR            directory the instrumented libraries write their
#                              profiles to (instrument, default /tmp/bolt-fdata)
#   BOLT_WORKLOAD_VARIANT      synthetic_workload.py --variant (run, default all)
#   BOLT_WORKLOAD_STEPS        synthetic_workload.py --steps (run, default 45000)
#   BOLT_WORKLOAD_WARMUP       synthetic_workload.py --warmup (run, default 50)
#   BOLT_WORKLOAD_MAX_SECONDS  synthetic_workload.py --max-seconds (run,
#                              default 0, no limit)
#   BOLT_SHARED_PROFILE_SOABI  Python version whose run provides the profiles
#                              of the libraries that do not depend on Python
#                              (finalize, default cpython-312-<arch>-linux-gnu)
#   BOLT_VALIDATE              set to 0 to skip the trial optimization (finalize)

set -euo pipefail

LLVM_BOLT_HOME="${LLVM_BOLT_HOME:-/usr/local}"
BOLT_FDATA_DIR="${BOLT_FDATA_DIR:-/tmp/bolt-fdata}"
BOLT_WORKLOAD_VARIANT="${BOLT_WORKLOAD_VARIANT:-all}"
BOLT_WORKLOAD_STEPS="${BOLT_WORKLOAD_STEPS:-45000}"
BOLT_WORKLOAD_WARMUP="${BOLT_WORKLOAD_WARMUP:-50}"
BOLT_WORKLOAD_MAX_SECONDS="${BOLT_WORKLOAD_MAX_SECONDS:-0}"
BOLT_SHARED_PROFILE_SOABI="${BOLT_SHARED_PROFILE_SOABI:-cpython-312-$(uname -m)-linux-gnu}"
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

log() {
  echo "[collect_profiles] $*"
}

die() {
  echo "[collect_profiles] ERROR: $*" >&2
  exit 1
}

usage() {
  echo "Usage: $0 instrument <prebolt-dir> <out-dir>" >&2
  echo "       $0 run <instrumented-dir> <runs-dir>" >&2
  echo "       $0 finalize <prebolt-dir> <runs-dir> <out-dir>" >&2
  exit 1
}

# Run a command with its output in a log file; show the tail on failure.
run_logged() {
  local logfile="$1"
  shift
  if ! "$@" > "${logfile}" 2>&1; then
    tail -n 50 "${logfile}" >&2
    die "command failed, see ${logfile}: $*"
  fi
}

# Create an empty output directory and set OUTPUT_DIR to its absolute path.
make_output_dir() {
  mkdir -p "$1"
  OUTPUT_DIR="$(cd "$1" && pwd)"
  if [[ -n "$(ls -A "${OUTPUT_DIR}")" ]]; then
    die "output directory ${OUTPUT_DIR} is not empty"
  fi
  mkdir -p "${OUTPUT_DIR}/logs"
}

# The profile name of library $1, a path relative to the prebolt directory:
# the name the build looks the profile up by (cmake/bolt_profiles/README.md)
# without .yaml, e.g. libtorch_python-cpython-312-aarch64-linux-gnu for
# cpython-312-aarch64-linux-gnu/libtorch_python.so.
profile_stem() {
  if [[ "$1" == */* ]]; then
    echo "$(basename "$1" .so)-$(dirname "$1")"
  else
    echo "${1%.so}"
  fi
}

build_id() {
  readelf -n "$1" 2> /dev/null | awk '/Build ID/ {print $3}'
}

setup_llvm() {
  LLVM_BOLT="${LLVM_BOLT_HOME}/bin/llvm-bolt"
  MERGE_FDATA="${LLVM_BOLT_HOME}/bin/merge-fdata"
  "${LLVM_BOLT}" --version | tee "${OUTPUT_DIR}/logs/llvm-bolt-version.txt"
}

# Check that the libraries in directory $1 are the ones listed in sha256sums
# file $2, i.e. the libraries that were instrumented.
verify_libraries() {
  (cd "$1" && sha256sum --check --quiet "$2") \
    || die "the libraries in $1 differ from the instrumented ones"
}

cmd_instrument() {
  [[ $# -eq 2 ]] || usage
  local prebolt_dir rel sections stem
  prebolt_dir="$(cd "$1" && pwd)"
  make_output_dir "$2"

  local -a libs
  mapfile -t libs < <(cd "${prebolt_dir}" && find . -name 'lib*.so' -printf '%P\n' | sort)
  [[ ${#libs[@]} -gt 0 ]] || die "no libraries in ${prebolt_dir}"
  for rel in "${libs[@]}"; do
    # BOLT needs the static relocations that --emit-relocs keeps in .rela.text.
    sections="$(readelf -S --wide "${prebolt_dir}/${rel}")"
    [[ "${sections}" == *" .rela.text "* ]] || die "${rel} has no .rela.text; was it linked with --emit-relocs?"
    [[ "${sections}" != *" .note.bolt_info "* ]] || die "${rel} is already BOLT-optimized"
  done
  log "Libraries to profile: ${libs[*]}"

  setup_llvm
  for rel in "${libs[@]}"; do
    log "Instrumenting ${rel}"
    stem="$(profile_stem "${rel}")"
    mkdir -p "$(dirname "${OUTPUT_DIR}/lib/${rel}")"
    # Every process that loads the library writes <stem>.<pid>.fdata to
    # BOLT_FDATA_DIR at exit. The path is compiled into the library.
    run_logged "${OUTPUT_DIR}/logs/instrument-${stem}.txt" \
      "${LLVM_BOLT}" "${prebolt_dir}/${rel}" -instrument \
      --instrumentation-file-append-pid \
      "--instrumentation-file=${BOLT_FDATA_DIR}/${stem}" \
      -o "${OUTPUT_DIR}/lib/${rel}"
  done
  (cd "${prebolt_dir}" && sha256sum "${libs[@]}") > "${OUTPUT_DIR}/sha256sums"
  echo "${BOLT_FDATA_DIR}" > "${OUTPUT_DIR}/fdata-dir"
  log "Instrumented libraries written to ${OUTPUT_DIR}/lib"
}

cmd_run() {
  [[ $# -eq 2 ]] || usage
  local instrumented_dir runs_dir fdata_dir soabi rel name stem
  instrumented_dir="$(cd "$1" && pwd)"
  mkdir -p "$2"
  runs_dir="$(cd "$2" && pwd)"

  # Import torch from site-packages, never from a source checkout in the cwd.
  cd "${WORK_DIR}"
  soabi="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"
  TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  log "Python ${soabi}, torch libraries: ${TORCH_LIB_DIR}"
  [[ -f "${instrumented_dir}/lib/${soabi}/libtorch_python.so" ]] \
    || die "no libtorch_python was instrumented for ${soabi}"
  make_output_dir "${runs_dir}/${soabi}"
  mkdir -p "${OUTPUT_DIR}/raw"
  fdata_dir="$(cat "${instrumented_dir}/fdata-dir")"
  mkdir -p "${fdata_dir}"
  # Drop what an earlier run that did not finish left behind.
  rm -f "${fdata_dir}"/*.fdata

  # The libraries that do not depend on Python and this Python's libtorch_python.
  local -a libs rpath_args
  mapfile -t libs < <(cd "${instrumented_dir}/lib" && find . -maxdepth 1 -name 'lib*.so' -printf '%P\n' | sort)
  libs+=("${soabi}/libtorch_python.so")
  for rel in "${libs[@]}"; do
    name="$(basename "${rel}")"
    if [[ "$(build_id "${TORCH_LIB_DIR}/${name}")" != "$(build_id "${instrumented_dir}/lib/${rel}")" ]]; then
      log "WARNING: the installed ${name} and the instrumented ${rel} have different build IDs"
    fi
    cp "${TORCH_LIB_DIR}/${name}" "${WORK_DIR}/original/${name}"
    RESTORE_LIBS+=("${name}")
    cp "${instrumented_dir}/lib/${rel}" "${TORCH_LIB_DIR}/${name}"
    # The build tree's libraries keep the build's RPATH; the wheel build sets
    # its own (.ci/wheel/linux/repair_wheel.py), as DT_RPATH for CUDA wheels.
    rpath_args=(--set-rpath "$(patchelf --print-rpath "${WORK_DIR}/original/${name}")")
    [[ "$(readelf -d "${WORK_DIR}/original/${name}")" != *"(RPATH)"* ]] || rpath_args+=(--force-rpath)
    patchelf "${rpath_args[@]}" "${TORCH_LIB_DIR}/${name}"
  done

  log "Running the synthetic workload on the instrumented libraries"
  python3 "${SCRIPT_DIR}/synthetic_workload.py" \
    --variant "${BOLT_WORKLOAD_VARIANT}" \
    --steps "${BOLT_WORKLOAD_STEPS}" \
    --warmup "${BOLT_WORKLOAD_WARMUP}" \
    --max-seconds "${BOLT_WORKLOAD_MAX_SECONDS}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/workload.txt"
  restore_libraries

  local -a raw
  for rel in "${libs[@]}"; do
    stem="$(profile_stem "${rel}")"
    raw=("${fdata_dir}/${stem}".*.fdata)
    [[ -e "${raw[0]}" ]] || die "no profile was written for ${rel}; did the workload exit normally?"
    log "${rel}: ${#raw[@]} profile(s)"
    mv "${raw[@]}" "${OUTPUT_DIR}/raw/"
  done
  # The same for the runs of every Python version.
  mkdir -p "${runs_dir}/logs"
  cp "${instrumented_dir}/sha256sums" "${runs_dir}/"
  cp "${instrumented_dir}/logs/"* "${runs_dir}/logs/"
  {
    python3 - << 'EOF'
import torch

print(f"torch: {torch.__version__} (git {torch.version.git_version})")
print(f"cuda: {torch.version.cuda}")
print(f"device: {torch.cuda.get_device_name(0)}")
EOF
    echo "workload: --variant ${BOLT_WORKLOAD_VARIANT} --steps ${BOLT_WORKLOAD_STEPS} --warmup ${BOLT_WORKLOAD_WARMUP} --max-seconds ${BOLT_WORKLOAD_MAX_SECONDS}"
    grep -E '^\[bolt-synth\] (variants=|done |stopping |WARNING: dropping)' \
      "${OUTPUT_DIR}/logs/workload.txt" || true
  } > "${OUTPUT_DIR}/run-info.txt"
  log "Profiles written to ${OUTPUT_DIR}/raw"
}

# Merge the profiles of library $1 (relative to $prebolt_dir) from the run of
# Python $2 into fdata/<stem>.fdata, convert them to yaml/<stem>.yaml and
# validate them, and append its section of the manifest to manifest.txt.
make_profile() {
  local rel="$1" run="$2" stem lib opt_log
  local -a raw
  stem="$(profile_stem "${rel}")"
  lib="${prebolt_dir}/${rel}"
  raw=("${runs_dir}/${run}/raw/${stem}".*.fdata)
  log "Merging ${#raw[@]} profile(s) of ${rel} from the ${run} run"
  run_logged "${OUTPUT_DIR}/logs/merge-fdata-${stem}.txt" \
    "${MERGE_FDATA}" "${raw[@]}" -o "${OUTPUT_DIR}/fdata/${stem}.fdata"
  [[ -s "${OUTPUT_DIR}/fdata/${stem}.fdata" ]] || die "merge-fdata produced an empty profile for ${rel}"

  # The build-time step consumes YAML profiles. Unlike fdata, they carry block
  # hashes that let -infer-stale-profile match profiles to binaries built from
  # slightly different sources.
  log "Converting the ${rel} profile to YAML"
  run_logged "${OUTPUT_DIR}/logs/convert-${stem}.txt" \
    "${LLVM_BOLT}" "${lib}" -o /dev/null \
    "-data=${OUTPUT_DIR}/fdata/${stem}.fdata" -w "${OUTPUT_DIR}/yaml/${stem}.yaml"

  opt_log="${OUTPUT_DIR}/logs/optimize-${stem}.txt"
  if [[ "${BOLT_VALIDATE}" == "1" ]]; then
    log "Trial optimization of ${rel}"
    run_logged "${opt_log}" \
      "${LLVM_BOLT}" "${lib}" -o "${WORK_DIR}/bolted.so" \
      "-data=${OUTPUT_DIR}/yaml/${stem}.yaml" "${BOLT_OPTIMIZE_OPTIONS[@]}"
    rm -f "${WORK_DIR}/bolted.so"
  fi

  {
    echo
    echo "${stem}.yaml (${rel}, ${run} run):"
    echo "  sha256: $(grep " ${rel}\$" "${runs_dir}/sha256sums" | cut -d' ' -f1)"
    echo "  build-id: $(build_id "${lib}")"
    echo "  merged profiles: ${#raw[@]}"
    grep -m1 'have non-empty execution profile' "${OUTPUT_DIR}/logs/convert-${stem}.txt" \
      | sed 's/^BOLT-INFO: /  /' || true
    if [[ -f "${opt_log}" ]]; then
      grep -m1 'could not be optimized' "${opt_log}" | sed 's/^BOLT-INFO: /  /' || true
      grep -m1 -o 'function CFG discontinuity.*' "${opt_log}" | sed 's/^/  /' || true
      awk '/: taken branches \(/ {print "  taken branches after BOLT: " $1 " " $5; exit}' "${opt_log}"
    fi
  } >> "${OUTPUT_DIR}/manifest.txt"
}

cmd_finalize() {
  [[ $# -eq 3 ]] || usage
  local prebolt_dir runs_dir rel run
  prebolt_dir="$(cd "$1" && pwd)"
  runs_dir="$(cd "$2" && pwd)"
  make_output_dir "$3"
  mkdir -p "${OUTPUT_DIR}"/{fdata,yaml}
  [[ -f "${runs_dir}/${BOLT_SHARED_PROFILE_SOABI}/run-info.txt" ]] \
    || die "no run for ${BOLT_SHARED_PROFILE_SOABI}, which provides the profiles of the libraries that do not depend on Python"
  verify_libraries "${prebolt_dir}" "${runs_dir}/sha256sums"
  cp "${runs_dir}/logs/"* "${OUTPUT_DIR}/logs/"
  setup_llvm

  {
    echo "llvm-bolt: $(grep -m1 'LLVM version' "${OUTPUT_DIR}/logs/llvm-bolt-version.txt")"
    for run in "${runs_dir}"/*/run-info.txt; do
      run="$(basename "$(dirname "${run}")")"
      cp "${runs_dir}/${run}/logs/workload.txt" "${OUTPUT_DIR}/logs/workload-${run}.txt"
      echo
      echo "${run} run:"
      sed 's/^/  /' "${runs_dir}/${run}/run-info.txt"
    done
  } > "${OUTPUT_DIR}/manifest.txt"

  local -a rels missing=()
  mapfile -t rels < <(awk '{print $2}' "${runs_dir}/sha256sums")
  for rel in "${rels[@]}"; do
    run="${BOLT_SHARED_PROFILE_SOABI}"
    [[ "${rel}" != */* ]] || run="$(dirname "${rel}")"
    if [[ -f "${runs_dir}/${run}/run-info.txt" ]]; then
      make_profile "${rel}" "${run}"
    else
      missing+=("$(profile_stem "${rel}").yaml")
    fi
  done
  # The fallback for Python versions without a profile of their own.
  cp "${OUTPUT_DIR}/yaml/libtorch_python-${BOLT_SHARED_PROFILE_SOABI}.yaml" "${OUTPUT_DIR}/yaml/libtorch_python.yaml"
  {
    echo
    echo "libtorch_python.yaml: libtorch_python-${BOLT_SHARED_PROFILE_SOABI}.yaml"
    [[ ${#missing[@]} -eq 0 ]] || echo "missing (no run): ${missing[*]}"
  } >> "${OUTPUT_DIR}/manifest.txt"

  (cd "${OUTPUT_DIR}/yaml" && tar -I zstd -cf ../torch.tar.zst -- *.yaml)

  cat "${OUTPUT_DIR}/manifest.txt"
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    { echo '### BOLT profiles'; echo '```'; cat "${OUTPUT_DIR}/manifest.txt"; echo '```'; } >> "${GITHUB_STEP_SUMMARY}"
  fi
  log "Profiles written to ${OUTPUT_DIR}"
}

# Libraries of the installed torch that cmd_run replaced, restored on exit.
RESTORE_LIBS=()
restore_libraries() {
  local lib
  for lib in "${RESTORE_LIBS[@]}"; do
    cp "${WORK_DIR}/original/${lib}" "${TORCH_LIB_DIR}/${lib}"
  done
  RESTORE_LIBS=()
}

cleanup() {
  restore_libraries
  cd /
  rm -rf "${WORK_DIR}"
}

[[ $# -ge 1 ]] || usage
command="$1"
shift
WORK_DIR="$(mktemp -d)"
mkdir -p "${WORK_DIR}/original"
trap cleanup EXIT
case "${command}" in
  instrument) cmd_instrument "$@" ;;
  run) cmd_run "$@" ;;
  finalize) cmd_finalize "$@" ;;
  *) usage ;;
esac
