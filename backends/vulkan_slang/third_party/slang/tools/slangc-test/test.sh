#!/usr/bin/env bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

summary=()
failure_count=0
test_count=0

test() {
  local name
  local exit_code
  name=$1
  shift
  pushd "$name" 1>/dev/null 2>&1
  echo "Running $name..."
  "$@" || exit_code=$?
  summary=("${summary[@]}" "$name: ")
  if [[ $exit_code -eq 0 ]]; then
    summary=("${summary[@]}" "  success")
  else
    summary=("${summary[@]}" "  failure (exit code: $exit_code)")
  fi
  popd 1>/dev/null 2>&1
  echo
  test_count=$((test_count + 1))
}

cd "${script_dir}"

test multiple-source-files slangc source1.slang source2.slang

echo ""
echo "Summary: "
echo
for line in "${summary[@]}"; do
  printf '  %s\n' "$line"
done
echo ""
echo "$failure_count failed, out of $test_count tests"
if [[ $failure_count -ne 0 ]]; then
  exit 1
fi
