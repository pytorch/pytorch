#!/bin/bash
# Checks internal links and anchors in the built HTML documentation.
#
# This script uses linkchecker (https://github.com/linkchecker/linkchecker)
# to verify that all internal href targets and #anchor fragments resolve
# to existing files/IDs.
#
# Usage:
#   ./scripts/lint_doc_anchors.sh BASE_SHA HEAD_SHA   # check only changed files (for PRs)
#   ./scripts/lint_doc_anchors.sh --all               # check entire site (for CI scheduled jobs)
#
# Prerequisites:
#   - Build the docs first:  cd docs && make html
#   - Install linkchecker:   pip install linkchecker

set -eo pipefail

DOCS_DIR="docs/build/html"
DOCS_SOURCE="docs/source"
CONFIG=".linkcheckerrc"
OUTPUT_FILE="linkchecker-report.txt"

if ! command -v linkchecker &>/dev/null; then
  echo "ERROR: linkchecker is not installed."
  echo "Install via:  pip install linkchecker"
  echo "See https://github.com/linkchecker/linkchecker"
  exit 1
fi

# Map a source file (e.g., docs/source/notes/cuda.rst) to its built HTML path
source_to_html() {
  local src="$1"
  # Remove docs/source/ prefix and change extension to .html
  local rel="${src#$DOCS_SOURCE/}"
  local html="${rel%.rst}.html"
  html="${html%.md}.html"
  echo "$DOCS_DIR/$html"
}

# Full site check (for scheduled CI)
if [[ "$1" == "--all" ]]; then
  ENTRY="$DOCS_DIR/index.html"
  if [ ! -f "$ENTRY" ]; then
    echo "ERROR: Built docs not found at $DOCS_DIR"
    echo "Build them first:  cd docs && make html"
    exit 1
  fi
  # Dynamically set threads based on available CPU cores
  # CLI arg (-t) overrides the config file setting
  THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 10)

  echo "Checking ALL internal links and anchors in $DOCS_DIR ..."
  echo "Using $THREADS threads (based on available CPU cores)"
  echo "This may take a while..."
  echo "Output will be saved to $OUTPUT_FILE"

  start_time=$(date +%s)
  # Capture linkchecker output to both terminal and file
  # Use tee to show progress while also saving for parsing
  # Disable exit-on-error temporarily so we can capture the exit code
  # and still print the summary even when linkchecker finds errors
  set +e
  linkchecker \
    --config="$CONFIG" \
    --threads="$THREADS" \
    "$ENTRY" 2>&1 | tee "$OUTPUT_FILE"
  exit_code=${PIPESTATUS[0]}
  set -e
  end_time=$(date +%s)
  elapsed=$((end_time - start_time))

  echo ""
  echo ""

  # Print summary of errors/broken links at the end
  if [[ -f "$OUTPUT_FILE" ]]; then
    # Extract stats from the report - linkchecker outputs stats at the end
    # Format: "That's it. X links in Y URLs checked. Z warnings found. A errors found."
    stats_line=$(grep -E "^That's it\." "$OUTPUT_FILE" 2>/dev/null || echo "")

    if [[ -n "$stats_line" ]]; then
      links_checked=$(echo "$stats_line" | grep -oE '[0-9]+ links' | grep -oE '[0-9]+' || echo "unknown")
      urls_checked=$(echo "$stats_line" | grep -oE '[0-9]+ URLs' | grep -oE '[0-9]+' || echo "unknown")
      warnings=$(echo "$stats_line" | grep -oE '[0-9]+ warnings' | grep -oE '[0-9]+' || echo "0")
      error_count=$(echo "$stats_line" | grep -oE '[0-9]+ errors' | grep -oE '[0-9]+' || echo "0")
    else
      # Fallback: count errors by looking for "Error:" lines
      error_count=$(grep -c "^Real URL" "$OUTPUT_FILE" 2>/dev/null || echo "0")
      urls_checked="unknown"
      warnings="0"
      links_checked="unknown"
    fi

    echo "=============================================="
    echo "SUMMARY"
    echo "=============================================="
    echo "Links checked:   ${links_checked:-unknown}"
    echo "URLs checked:    ${urls_checked:-unknown}"
    echo "Warnings:        ${warnings:-0}"
    echo "Errors:          ${error_count:-0}"
    echo "Time elapsed:    ${elapsed}s ($(printf '%d:%02d' $((elapsed/60)) $((elapsed%60))) min)"
    echo "=============================================="
    echo "Full report:     $OUTPUT_FILE"
    echo "=============================================="

    if [[ "${error_count:-0}" -gt 0 ]]; then
      echo ""
      echo "ERRORS FOUND:"
      echo "-------------"
      # Extract error blocks from the report using awk
      # Error blocks start with "URL" and end with "Result.*Error"
      # Progress lines (e.g., "10 threads active...") may appear between blocks
      awk '
        /^URL / {
          block = $0
          in_block = 1
          next
        }
        in_block && /^Result.*Error/ {
          print block
          print $0
          print ""
          in_block = 0
          block = ""
          next
        }
        in_block && /^Result/ {
          in_block = 0
          block = ""
          next
        }
        in_block && /threads active/ {
          next
        }
        in_block {
          block = block "\n" $0
        }
      ' "$OUTPUT_FILE" || true
      echo "-------------"
      echo ""
      echo "To fix these errors:"
      echo "  1. Fix the broken links in your source files:"
      echo "     - Documentation: docs/source/*.rst, docs/source/*.md"
      echo "     - Docstrings: Python files under torch/, torchgen/, etc."
      echo "  2. Rebuild docs:  cd docs && make html"
      echo "  3. Re-run:        ./scripts/lint_doc_anchors.sh --all"
    else
      echo ""
      echo "SUCCESS: No broken links found!"
    fi
  else
    # No output file - still show timing
    echo "=============================================="
    echo "SUMMARY"
    echo "=============================================="
    echo "Time elapsed:    ${elapsed}s ($(printf '%d:%02d' $((elapsed/60)) $((elapsed%60))) min)"
    echo "=============================================="
    if [[ $exit_code -eq 0 ]]; then
      echo "SUCCESS: No broken links found!"
    else
      echo "Errors were found. Exit code: $exit_code"
    fi
  fi

  exit $exit_code
fi

# Changed files check (for PRs / local development)
if [[ $# -eq 2 ]]; then
  BASE_SHA="$1"
  HEAD_SHA="$2"
  echo "Checking links/anchors in doc files changed between $BASE_SHA and $HEAD_SHA ..."

  # Find changed doc source files
  changed_files=$(git diff --name-only "$BASE_SHA...$HEAD_SHA" -- "$DOCS_SOURCE" | grep -E '\.(rst|md)$' || true)

  if [[ -z "$changed_files" ]]; then
    echo "No documentation files changed. Nothing to check."
    exit 0
  fi

  # Check each changed file's built HTML
  status=0
  for src in $changed_files; do
    html=$(source_to_html "$src")
    if [[ -f "$html" ]]; then
      echo "Checking $html ..."
      linkchecker \
        --config="$CONFIG" \
        -r 0 \
        "$html" || status=1
    else
      echo "WARN: Built HTML not found for $src (expected $html)"
    fi
  done
  exit $status
fi

# No arguments - show usage
echo "Usage:"
echo "  $0 BASE_SHA HEAD_SHA   # check only changed doc files (for PRs)"
echo "  $0 --all               # check entire site (for scheduled CI)"
echo ""
echo "Examples:"
echo "  $0 HEAD~1 HEAD         # check docs changed in last commit"
echo "  $0 main HEAD           # check docs changed vs main branch"
echo "  $0 --all               # full site check"
exit 1
