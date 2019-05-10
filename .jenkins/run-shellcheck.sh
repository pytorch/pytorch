#!/bin/bash -xe

# One may want to invoke this script locally as follows:
#
#   .jenkins/run-shellcheck.sh --color=always | less -R


EXCLUSIONS=SC2086,SC1091,SC2155,SC1090,SC2164,SC1003

find .jenkins/pytorch -name *.sh | xargs shellcheck --exclude=$EXCLUSIONS --external-sources "$@" || true
