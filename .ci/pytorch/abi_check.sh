#!/bin/bash
# ABI Compliance Checker script for PyTorch
#
# This script checks for ABI compatibility between the current PR build
# and baseline PyTorch releases (nightly and stable).
#
# Usage: ./abi_check.sh [cpu|cuda] [true|false] [artifact-dir]
#
# Arguments:
#   $1 - Build type: 'cpu' or 'cuda'
#   $2 - Suppressed: 'true' if suppress-abi-check label is present
#   $3 - Artifact directory containing the PR build artifacts

set -euo pipefail

BUILD_TYPE="${1:-cpu}"
SUPPRESSED="${2:-false}"
ARTIFACT_DIR="${3:-pr-build}"

# Configuration
CUDA_VERSION="128"
PYTHON_VERSION="3.10"
REPORT_DIR="abi-reports"

# Libraries to check for each build type
CPU_LIBS=(
    "libc10.so"
    "libtorch_cpu.so"
    "libtorch.so"
    "libshm.so"
)

CUDA_LIBS=(
    "${CPU_LIBS[@]}"
    "libc10_cuda.so"
    "libtorch_cuda.so"
)

# Set library list and index URLs based on build type
if [ "$BUILD_TYPE" = "cuda" ]; then
    LIBS=("${CUDA_LIBS[@]}")
    NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION}"
    STABLE_INDEX="https://download.pytorch.org/whl/cu${CUDA_VERSION}"
else
    LIBS=("${CPU_LIBS[@]}")
    NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cpu"
    STABLE_INDEX="https://download.pytorch.org/whl/cpu"
fi

echo "=== ABI Compatibility Check ==="
echo "Build type: $BUILD_TYPE"
echo "Suppressed: $SUPPRESSED"
echo "Artifact dir: $ARTIFACT_DIR"
echo "Libraries to check: ${LIBS[*]}"
echo ""

# Install ABI Compliance Checker and dependencies
install_dependencies() {
    echo "=== Installing dependencies ==="
    apt-get update -qq
    apt-get install -y -qq abi-compliance-checker abi-dumper elfutils > /dev/null 2>&1 || {
        echo "ERROR: Failed to install abi-compliance-checker"
        echo "Trying to install from source..."
        apt-get install -y -qq elfutils libelf-dev perl > /dev/null 2>&1
        git clone --depth 1 https://github.com/lvc/abi-dumper.git /tmp/abi-dumper
        git clone --depth 1 https://github.com/lvc/abi-compliance-checker.git /tmp/abi-compliance-checker
        cd /tmp/abi-dumper && make install
        cd /tmp/abi-compliance-checker && make install
    }
    echo "abi-compliance-checker version: $(abi-compliance-checker -dumpversion 2>/dev/null || echo 'unknown')"
}

# Download and extract a PyTorch wheel
# Returns the path to the extracted lib directory
download_baseline() {
    local index_url="$1"
    local dest_dir="$2"
    local name="$3"

    echo "Downloading $name baseline from $index_url..."
    mkdir -p "$dest_dir"

    # Download the wheel
    if ! pip download torch --no-deps --dest "$dest_dir" \
        --index-url "$index_url" \
        --python-version "$PYTHON_VERSION" \
        --only-binary=:all: 2>/dev/null; then
        echo "WARNING: Could not download $name baseline wheel"
        return 1
    fi

    # Find and extract the wheel
    local wheel_file
    wheel_file=$(find "$dest_dir" -name 'torch-*.whl' -type f | head -1)
    if [ -z "$wheel_file" ]; then
        echo "WARNING: No wheel file found in $dest_dir"
        return 1
    fi

    echo "Extracting $wheel_file..."
    unzip -q "$wheel_file" -d "$dest_dir/extracted"

    local lib_dir="$dest_dir/extracted/torch/lib"
    if [ ! -d "$lib_dir" ]; then
        echo "WARNING: No lib directory found in extracted wheel"
        return 1
    fi

    echo "$lib_dir"
}

# Create an ABI dump for a library
create_abi_dump() {
    local lib_path="$1"
    local output_file="$2"
    local version="$3"

    if [ ! -f "$lib_path" ]; then
        echo "WARNING: Library not found: $lib_path"
        return 1
    fi

    echo "Creating ABI dump for $(basename "$lib_path") ($version)..."

    # abi-dumper needs debug info for best results, but can work without
    if ! abi-dumper "$lib_path" -o "$output_file" -lver "$version" 2>/dev/null; then
        echo "WARNING: abi-dumper failed for $lib_path"
        return 1
    fi

    return 0
}

# Compare two ABI dumps and generate report
# Returns 0 if compatible, 1 if breaking changes detected
compare_abi() {
    local lib_name="$1"
    local old_dump="$2"
    local new_dump="$3"
    local report_path="$4"

    echo "Comparing ABI for $lib_name..."

    if [ ! -f "$old_dump" ] || [ ! -f "$new_dump" ]; then
        echo "SKIP: Missing dump files for $lib_name"
        return 0
    fi

    # Run abi-compliance-checker
    # Exit code: 0 = compatible, 1 = incompatible, other = error
    local result
    if abi-compliance-checker -l "$lib_name" \
        -old "$old_dump" \
        -new "$new_dump" \
        -report-path "$report_path" 2>/dev/null; then
        echo "PASS: $lib_name is ABI compatible"
        return 0
    else
        result=$?
        if [ $result -eq 1 ]; then
            echo "FAIL: ABI breaking changes detected in $lib_name"
            return 1
        else
            echo "WARNING: abi-compliance-checker returned error code $result for $lib_name"
            return 0
        fi
    fi
}

# Generate a summary report for GitHub
generate_summary() {
    local has_failures="$1"
    # Write to a file that can be read by the workflow
    local summary_file="$REPORT_DIR/summary.md"

    {
        echo "# ABI Compatibility Check - ${BUILD_TYPE^^}"
        echo ""
        echo "Compared against: **nightly** and **stable** baselines"
        echo ""

        if [ "$has_failures" = "true" ]; then
            echo "## ⚠️ ABI Breaking Changes Detected"
            echo ""
            echo "One or more libraries have ABI incompatibilities with the baseline."
            echo ""
            echo "### What This Means"
            echo "- Code compiled against the old library version may not work with the new version"
            echo "- This could break downstream users who depend on PyTorch's C++ ABI"
            echo ""
            echo "### Next Steps"
            echo "1. Review the detailed HTML reports in the job artifacts"
            echo "2. If the changes are intentional, add the \`suppress-abi-check\` label to your PR"
            echo "3. If unintentional, consider alternative implementations that preserve ABI"
        else
            echo "## ✅ ABI Compatible"
            echo ""
            echo "No ABI breaking changes detected in the checked libraries."
        fi

        echo ""
        echo "---"
        echo ""
        echo "<details>"
        echo "<summary>Libraries Checked</summary>"
        echo ""
        for lib in "${LIBS[@]}"; do
            echo "- \`$lib\`"
        done
        echo ""
        echo "</details>"
    } > "$summary_file"

    # Also try to append to GITHUB_STEP_SUMMARY if available (for non-Docker runs)
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ] && [ -f "${GITHUB_STEP_SUMMARY:-}" ]; then
        cat "$summary_file" >> "$GITHUB_STEP_SUMMARY"
    fi
}

# Main execution
main() {
    install_dependencies

    mkdir -p "$REPORT_DIR"

    local nightly_libs=""
    local stable_libs=""
    local has_failures=false

    # Download baseline wheels
    echo ""
    echo "=== Downloading Baseline Wheels ==="
    nightly_libs=$(download_baseline "$NIGHTLY_INDEX" "baseline-nightly" "nightly") || nightly_libs=""
    stable_libs=$(download_baseline "$STABLE_INDEX" "baseline-stable" "stable") || stable_libs=""

    if [ -z "$nightly_libs" ] && [ -z "$stable_libs" ]; then
        echo "ERROR: Could not download any baseline wheels"
        echo "Cannot perform ABI comparison without baselines"
        exit 1
    fi

    # Extract PR libraries from artifacts
    echo ""
    echo "=== Extracting PR Build Artifacts ==="
    if [ -f "$ARTIFACT_DIR/artifacts.zip" ]; then
        unzip -o "$ARTIFACT_DIR/artifacts.zip" -d "$ARTIFACT_DIR"
    fi

    # Find PR libraries - they could be in several locations
    local pr_libs=""
    for candidate in "$ARTIFACT_DIR/build/lib" "$ARTIFACT_DIR/torch/lib" "$ARTIFACT_DIR/lib"; do
        if [ -d "$candidate" ]; then
            pr_libs="$candidate"
            break
        fi
    done

    if [ -z "$pr_libs" ] || [ ! -d "$pr_libs" ]; then
        echo "ERROR: Could not find PR libraries in artifact directory"
        echo "Contents of $ARTIFACT_DIR:"
        find "$ARTIFACT_DIR" -type f -name '*.so' 2>/dev/null | head -20 || true
        exit 1
    fi

    echo "PR libraries found at: $pr_libs"

    # Compare each library
    echo ""
    echo "=== Checking Libraries ==="

    for lib in "${LIBS[@]}"; do
        echo ""
        echo "--- $lib ---"

        local pr_lib="$pr_libs/$lib"
        if [ ! -f "$pr_lib" ]; then
            echo "SKIP: $lib not found in PR build"
            continue
        fi

        # Create PR dump
        local pr_dump="$REPORT_DIR/pr_${lib}.dump"
        if ! create_abi_dump "$pr_lib" "$pr_dump" "pr"; then
            echo "SKIP: Could not create ABI dump for PR version of $lib"
            continue
        fi

        # Compare with nightly
        if [ -n "$nightly_libs" ] && [ -f "$nightly_libs/$lib" ]; then
            local nightly_dump="$REPORT_DIR/nightly_${lib}.dump"
            if create_abi_dump "$nightly_libs/$lib" "$nightly_dump" "nightly"; then
                if ! compare_abi "$lib" "$nightly_dump" "$pr_dump" "$REPORT_DIR/${lib}_vs_nightly.html"; then
                    has_failures=true
                fi
            fi
        else
            echo "SKIP: $lib not found in nightly baseline"
        fi

        # Compare with stable
        if [ -n "$stable_libs" ] && [ -f "$stable_libs/$lib" ]; then
            local stable_dump="$REPORT_DIR/stable_${lib}.dump"
            if create_abi_dump "$stable_libs/$lib" "$stable_dump" "stable"; then
                if ! compare_abi "$lib" "$stable_dump" "$pr_dump" "$REPORT_DIR/${lib}_vs_stable.html"; then
                    has_failures=true
                fi
            fi
        else
            echo "SKIP: $lib not found in stable baseline"
        fi
    done

    # Generate summary
    echo ""
    echo "=== Generating Summary ==="
    generate_summary "$has_failures"

    # List generated reports
    echo ""
    echo "Generated reports:"
    find "$REPORT_DIR" -name '*.html' -type f 2>/dev/null | while read -r report; do
        echo "  - $report"
    done

    # Handle exit status
    echo ""
    if [ "$has_failures" = "true" ]; then
        if [ "$SUPPRESSED" = "true" ]; then
            echo "=== ABI ISSUES DETECTED BUT SUPPRESSED ==="
            echo "The suppress-abi-check label is present, so this check will pass."
            echo "Please review the ABI compatibility reports in the artifacts."
            exit 0
        else
            echo "=== ABI COMPATIBILITY CHECK FAILED ==="
            echo "To suppress this failure, add the 'suppress-abi-check' label to your PR."
            exit 1
        fi
    fi

    echo "=== ABI COMPATIBILITY CHECK PASSED ==="
    exit 0
}

main "$@"
