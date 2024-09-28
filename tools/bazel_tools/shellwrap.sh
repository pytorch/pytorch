#!/bin/bash

# This script is helpful in entering an interactive shell from a bazel build
# before running a given bazel executable.
# This can provide a quick way to explore the sandbox directory and filesystem.
# Typical use is with
#
#     bazel run --run_under=//tools/bazel:shell_wrapper //:target
#     OR
#     bazel run --config=shell //:target

shell='/bin/bash'
rcfile='/tmp/pytorch_bazel_tools_shellwrap'
while [[ $# -gt 0 ]] ; do
    case "$1" in
        --shell_bin_path)
            # path for the shell executable
            shell="$2"
            shift 2
            ;;
        --rcfile)
            # path for the file used to write the environment
            rcfile="$2"
            shift 2
            ;;
        *)
            # remaining arguments are part of the command for execution
            break
            ;;
    esac
done

if ! tty -s; then
    echo 'A tty is not available.'
    echo "Use \`bazel run\`, not \`bazel test\`."
    exit 1
fi

NOCOLOR='\033[0m'
YELLOW='\033[1;33m'

# store the environment in a file
export PYTORCH_SHELL_COMMAND=$*
echo "alias run=\"$*\"" > "$rcfile"
echo "PS1='\s-\v\$ '" >> "$rcfile"

echo =====
# print the execution command (command is yellow)
echo -e "alias run=${YELLOW}$PYTORCH_SHELL_COMMAND${NOCOLOR}"
echo =====

echo "Entering interactive shell at the execution root:"

# quote escape all the arguments to use as a single input string
cmd="'$shell' --noprofile --rcfile '$rcfile'"

# run the command in a script psuedo terminal and dump to null
/usr/bin/script -c "$cmd" -q /dev/null
