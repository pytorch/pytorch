#!/usr/bin/env bash
set -e

show_help() {
  me=$(basename "$0")
  cat <<EOF
$me: Run all of the examples in test mode

Usage: $me --os <os> --config <config> --bin-dir <path> --skip-file <path> [--dry-run]

Options:
  --help                 Show this help message
  --dry-run              Skip running the examples.
  --bin-dir   <path>     Path to binary directory.
                         Must contain all of the example binaries.
  --skip-file <path>     Path to file containing skip patterns.
                         See the 'Skip file' section, below.
  --os        <os>       Operating system.
                         Valid <os> values: 'macos', 'windows', 'linux'
  --config    <config>   Build configuration.
                         Valid <config> values: 'debug', 'release'.

  --platform  <platform> Target platform.
                         Valid <platform> values: 'x86_64', 'aarch64'.

Skip file:

  The skip patterns are regexp patterns on the following format:
  <os>:<platform>:<config>:<sample> # Some comment describing why test is disabled

  For example:
  # Bug 123: foo-example fails (both debug and release)
  windows:x86_64(debug|release):foo-example
  # Bug 456: bar-example fails in release mode on Linux
  linux:aarch64:release:bar-example

EOF
}

function user_error() {
  echo "error: $1" >&2
  echo "" >&2
  show_help >&2
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
  -h | --help)
    show_help
    exit 0
    ;;
  --dry-run)
    dry_run=true
    ;;
  --bin-dir)
    bin_dir="$2"
    shift
    ;;
  --skip-file)
    skip_file="$2"
    shift
    ;;
  --os)
    case $2 in
    windows | linux | macos) ;;
    *)
      user_error "Unrecognized os: '$2'"
      ;;
    esac
    os="$2"
    shift
    ;;
  --config)
    case $2 in
    debug | release) ;;
    *)
      user_error "Unrecognized config: '$2'"
      ;;
    esac
    config="$2"
    shift
    ;;
  --platform)
    case $2 in
    x86_64 | aarch64) ;;
    *)
      user_error "Unrecognized platform: '$2'"
      ;;
    esac
    platform="$2"
    shift
    ;;
  *)
    user_error "Unrecognized argument: '$1'"
    ;;
  esac
  shift
done

if [[ "$os" == "" ]]; then
  user_error "No OS specified."
fi

if [[ "$config" == "" ]]; then
  user_error "No build configuration specified."
fi

if [[ "$bin_dir" == "" ]]; then
  user_error "No binary directory specified."
fi

if [[ "$skip_file" == "" ]]; then
  user_error "No skip file specified."
fi

if [[ "$platform" == "" ]]; then
  user_error "No platform specified."
fi

if [[ ! -f "$skip_file" ]]; then
  user_error "Skip file '$skip_file' does not exist."
fi

if [[ ! -d "$bin_dir" ]]; then
  user_error "Binary directory '$bin_dir' does not exist."
fi

summary=()
failure_count=0
skip_count=0
sample_count=0

function skip {
  local p
  local line_index
  p="$1"
  line_index=1
  while read -r pattern; do
    pat=$pattern
    if [[ ! $pat =~ .*# ]]; then
      user_error "Skip pattern on line $line_index is missing a comment!"
    fi
    pat="${pattern%% *#*}"
    if [[ $p =~ ^$pat$ ]]; then
      return 0
    fi
    line_index=$((line_index + 1))
  done <"$skip_file"

  return 1
}

function run_sample {
  local sample
  local args
  sample="$1"
  shift
  args=("$@")
  sample_count=$((sample_count + 1))
  summary=("${summary[@]}" "$sample: ")
  if skip "$os:$platform:$config:$sample"; then
    echo "Skipping $sample..."
    summary=("${summary[@]}" "  skipped")
    skip_count=$((skip_count + 1))
    return
  fi
  echo "Running '$sample ${args[*]}'..."
  result=0
  pushd "$bin_dir" 1>/dev/null 2>&1
  if [[ ! "$dry_run" = true ]]; then
    ./"$sample" "${args[@]}" || result=$?
    if [[ -f ./"log-$sample.txt" ]]; then
      cat ./"log-$sample.txt"
    fi
  fi
  if [[ $result -eq 0 ]]; then
    summary=("${summary[@]}" "  success")
  else
    summary=("${summary[@]}" "  failure (exit code: $result)")
    failure_count=$((failure_count + 1))
  fi
  popd 1>/dev/null 2>&1
}

sample_commands=(
  'platform-test --test-mode'
  'ray-tracing-pipeline --test-mode'
  'ray-tracing --test-mode'
  'shader-toy --test-mode'
  'triangle --test-mode'
  'model-viewer --test-mode'
  'shader-object'
  'reflection-api'
  'hello-world'
  'gpu-printing'
  'cpu-hello-world'
  'cpu-com-example'
)

for sample_command in "${sample_commands[@]}"; do
  run_sample ${sample_command}
  echo ""
done

echo ""
echo "Summary: "
echo
for line in "${summary[@]}"; do
  printf '  %s\n' "$line"
done
echo ""
echo "$failure_count failed, and $skip_count skipped, out of $sample_count tests"
if [[ $failure_count -ne 0 ]]; then
  exit 1
fi
