#!/bin/bash

setup_torch_trace() {
  if [[ "${ENABLE_TORCH_TRACE:-0}" != "1" ]]; then
    return
  fi
  local trace_dir="${RUNNER_TEMP:-/tmp}/torch_traces"
  mkdir -p "$trace_dir"
  export TORCH_TRACE="$trace_dir"
  echo "TORCH_TRACE enabled: writing structured trace logs to $trace_dir"
}

collect_tlparse_output() {
  if [[ "${ENABLE_TORCH_TRACE:-0}" != "1" ]]; then
    return
  fi
  local trace_dir="${RUNNER_TEMP:-/tmp}/torch_traces"
  local test_reports_dir
  test_reports_dir=$(pwd)/test/test-reports

  local trace_files=()
  for f in "$trace_dir"/*.log; do
    [ -f "$f" ] || continue
    trace_files+=("$f")
  done

  if [[ ! -d "$trace_dir" ]] || [[ "${#trace_files[@]}" -eq 0 ]]; then
    echo "No torch trace files found in $trace_dir, skipping tlparse"
    return
  fi

  echo "Collecting torch trace output from $trace_dir"

  local tlparse_output_dir="$test_reports_dir/tlparse_output"
  local raw_output_dir="$tlparse_output_dir/raw"
  local parsed_output_dir="$tlparse_output_dir/parsed"

  # Always preserve raw trace logs (gzipped) for downstream analysis.
  # Keep them out of the tlparse output directory because tlparse --overwrite
  # removes its output directory before writing the parsed report.
  mkdir -p "$raw_output_dir"
  for f in "${trace_files[@]}"; do
    gzip -c "$f" > "$raw_output_dir/$(basename "$f").gz"
  done
  echo "Raw trace logs saved to $raw_output_dir/"

  # Try to generate HTML report via tlparse (best-effort).
  if ! command -v tlparse &>/dev/null; then
    pip install tlparse 2>/dev/null || {
      echo "Warning: failed to install tlparse, skipping HTML generation"
      return
    }
  fi

  tlparse -o "$parsed_output_dir" --no-browser --overwrite "$trace_dir" 2>&1 || {
    echo "Warning: tlparse failed for $trace_dir"
  }
}
