#!/usr/bin/env bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$script_dir")"
check_only=0

show_help() {
  me=$(basename "$0")
  cat <<EOF
$me: Build table of contents for documentation directories

Usage: $me [--help] [--source <path>] [--check-only]

Options:
  --help           Show this help message
  --source         Path to project root directory (defaults to parent of the script directory)
  --check-only     Check if TOC needs updating, exit 1 if changes needed
EOF
}

while [[ "$#" -gt 0 ]]; do
  case $1 in
  -h | --help)
    show_help
    exit 0
    ;;
  --source)
    project_root="$2"
    shift
    ;;
  --check-only)
    check_only=1
    ;;
  *)
    echo "unrecognized argument: $1" >&2
    show_help >&2
    exit 1
    ;;
  esac
  shift
done

missing_bin=0

require_bin() {
  local name="$1"
  if ! command -v "$name" &>/dev/null; then
    echo "This script needs $name, but it isn't in \$PATH" >&2
    missing_bin=1
    return
  fi
}

require_bin "mcs"
require_bin "mono"

if [ "$missing_bin" -eq 1 ]; then
  exit 1
fi

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT

docs_dir="$project_root/docs"

cat >"$temp_dir/temp_program.cs" <<EOL
$(cat "$script_dir/scripts/Program.cs")

namespace toc
{
    class Program
    {
        static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("Please provide a directory path");
                return 1;
            }

            try
            {
                Builder.Run(args[0]);
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine(\$"Error: {ex.Message}");
                return 1;
            }
        }
    }
}
EOL

if ! mcs -r:System.Core "$temp_dir/temp_program.cs" -out:"$temp_dir/toc-builder.exe"; then
  echo "Compilation of $script_dir/scripts/Program.cs failed" >&2
  exit 1
fi

for dir in "user-guide" "gfx-user-guide"; do
  if [ -d "$docs_dir/$dir" ]; then
    if [ "$check_only" -eq 1 ]; then
      # Ensure working directory is clean
      if ! git -C "$project_root" diff --quiet "docs/$dir/toc.html" 2>/dev/null; then
        echo "Working directory not clean, cannot check TOC" >&2
        exit 1
      fi
    fi

    if ! mono "$temp_dir/toc-builder.exe" "$docs_dir/$dir"; then
      echo "TOC generation failed for $dir" >&2
      exit 1
    fi

    if [ "$check_only" -eq 1 ]; then
      if ! git -C "$project_root" diff --quiet "docs/$dir/toc.html" 2>/dev/null; then
        git -C "$project_root" diff --color "docs/$dir/toc.html"
        git -C "$project_root" checkout -- "docs/$dir/toc.html" 2>/dev/null
        exit 1
      fi
    fi
  else
    echo "Directory $dir not found" >&2
  fi
done
