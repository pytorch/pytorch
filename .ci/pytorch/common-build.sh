#!/bin/bash
# Required environment variables:
#   $BUILD_ENVIRONMENT (should be set by your Docker image)

if [[ "$BUILD_ENVIRONMENT" != *win-* ]]; then
    # Save the absolute path in case later we chdir (as occurs in the gpu perf test)
    script_dir="$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )"

    if which sccache > /dev/null; then
        # Save sccache logs to file
        sccache --stop-server > /dev/null  2>&1 || true
        rm -f ~/sccache_error.log || true

        function sccache_epilogue() {
            echo "::group::Sccache Compilation Log"
            echo '=================== sccache compilation log ==================='
            python "$script_dir/print_sccache_log.py" ~/sccache_error.log 2>/dev/null || true
            echo '=========== If your build fails, please take a look at the log above for possible reasons ==========='
            sccache --show-stats
            sccache --stop-server || true
            echo "::endgroup::"
        }

        # Register the function here so that the error log can be printed even when
        # sccache fails to start, i.e. timeout error
        trap_add sccache_epilogue EXIT

        if [[ -n "${SKIP_SCCACHE_INITIALIZATION:-}" ]]; then
            # sccache --start-server seems to hang forever on self hosted runners for GHA
            # so let's just go ahead and skip the --start-server altogether since it seems
            # as though sccache still gets used even when the sscache server isn't started
            # explicitly
            echo "Skipping sccache server initialization, setting environment variables"
            export SCCACHE_IDLE_TIMEOUT=0
            export SCCACHE_ERROR_LOG=~/sccache_error.log
            export RUST_LOG=sccache::server=error
        elif [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
            SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 sccache --start-server
        else
            # increasing SCCACHE_IDLE_TIMEOUT so that extension_backend_test.cpp can build after this PR:
            # https://github.com/pytorch/pytorch/pull/16645
            SCCACHE_ERROR_LOG=~/sccache_error.log SCCACHE_IDLE_TIMEOUT=0 RUST_LOG=sccache::server=error sccache --start-server
        fi

        # Report sccache stats for easier debugging. It's ok if this commands
        # timeouts and fails on MacOS
        sccache --zero-stats || true
    fi

    if which ccache > /dev/null; then
        # Report ccache stats for easier debugging
        ccache --zero-stats
        ccache --show-stats
        function ccache_epilogue() {
            ccache --show-stats
        }
        trap_add ccache_epilogue EXIT
    fi
fi
