# --- begin runfiles.bash initialization ---
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation rules_cc/tests/system_library/unittest.bash)" \
  || { echo "Could not rules_cc/source tests/system_library/unittest.bash" >&2; exit 1; }


function setup_system_library() {
  mkdir -p systemlib

  cat << EOF > systemlib/foo.cc
int bar() {
  return 42;
}
EOF

  cat << EOF > systemlib/foo.h
int bar();
EOF

  cd systemlib

  g++ -c -fpic foo.cc || fail "Expected foo.o to build successfully"
  g++ -shared -o libfoo.so foo.o || fail "Expected foo.so to build successfully"
  g++ -c foo.cc || fail "Expected foo.o to build successfully"
  ar rvs foo.a foo.o || fail "Expected foo.a to build successfully"

  cd ..

  cat << EOF > WORKSPACE
load("//:cc/system_library.bzl", "system_library")
system_library(
    name = "foo",
    hdrs = [
        "foo.h",
    ],
    static_lib_names = ["libfoo.a"],
    shared_lib_names = ["libfoo.so"]
)

system_library(
    name = "foo_hardcoded_path",
    hdrs = [
        "foo.h",
    ],
    static_lib_names = ["libfoo.a"],
    shared_lib_names = ["libfoo.so"],
    lib_path_hints = ["${PWD}/systemlib"],
    includes = ["${PWD}/systemlib"]
)
EOF

  cat << EOF > BUILD
cc_binary(
    name = "test",
    srcs = ["test.cc"],
    deps = ["@foo"]
)

cc_binary(
    name = "test_static",
    srcs = ["test.cc"],
    deps = ["@foo"],
    linkstatic = True
)

cc_binary(
    name = "test_hardcoded_path",
    srcs = ["test.cc"],
    deps = ["@foo_hardcoded_path"]
)

cc_binary(
    name = "test_static_hardcoded_path",
    srcs = ["test.cc"],
    deps = ["@foo_hardcoded_path"],
    linkstatic = True
)

cc_binary(
    name = "fake_rbe",
    srcs = ["test.cc"],
    deps = ["@foo_hardcoded_path"]
)
EOF

  cat << EOF > test.cc
#include "foo.h"

int main() {
  return 42 - bar();
}
EOF
}
#### TESTS #############################################################

# Make sure it fails with a correct message when no library is found
function test_system_library_not_found() {
  setup_system_library

  bazel run //:test \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  &> $TEST_log \
  || true
  expect_log "Library foo could not be found"

    bazel run //:test_static \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  &> $TEST_log \
  || true
  expect_log "Library foo could not be found"
  }

function test_override_paths() {
  setup_system_library

  bazel run //:test \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  --action_env=BAZEL_LIB_OVERRIDE_PATHS=foo="${PWD}"/systemlib \
  --action_env=BAZEL_INCLUDE_OVERRIDE_PATHS=foo="${PWD}"/systemlib \
  || fail "Expected test to run successfully"

  bazel run //:test_static \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  --action_env=BAZEL_LIB_OVERRIDE_PATHS=foo="${PWD}"/systemlib \
  --action_env=BAZEL_INCLUDE_OVERRIDE_PATHS=foo="${PWD}"/systemlib \
  || fail "Expected test_static to run successfully"
}

function test_additional_paths() {
  setup_system_library

  bazel run //:test \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  --action_env=BAZEL_LIB_ADDITIONAL_PATHS=foo="${PWD}"/systemlib \
  --action_env=BAZEL_INCLUDE_ADDITIONAL_PATHS=foo="${PWD}"/systemlib \
  || fail "Expected test to run successfully"

  bazel run //:test_static \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  --action_env=BAZEL_LIB_ADDITIONAL_PATHS=foo="${PWD}"/systemlib \
  --action_env=BAZEL_INCLUDE_ADDITIONAL_PATHS=foo="${PWD}"/systemlib \
  || fail "Expected test_static to run successfully"
}

function test_hardcoded_paths() {
  setup_system_library

  bazel run //:test_hardcoded_path \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  || fail "Expected test_hardcoded_path to run successfully"

  bazel run //:test_static_hardcoded_path \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  || fail "Expected test_static_hardcoded_path to run successfully"
}

function test_system_library_no_lib_names() {
  cat << EOF > WORKSPACE
load("//:cc/system_library.bzl", "system_library")
system_library(
    name = "foo",
    hdrs = [
        "foo.h",
    ]
)
EOF

  cat << EOF > BUILD
cc_binary(
    name = "test",
    srcs = ["test.cc"],
    deps = ["@foo"]
)
EOF

  # It should fail when no static_lib_names and static_lib_names are given
  bazel run //:test \
  --experimental_starlark_cc_import \
  --experimental_repo_remote_exec \
  &> $TEST_log \
  || true
  expect_log "Library foo could not be found"
}

run_suite "Integration tests for system_library."