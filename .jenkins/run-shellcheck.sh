#!/bin/bash -xe

# One may want to invoke this script locally as follows:
#
#   .jenkins/run-shellcheck.sh --color=always | less -R


find .jenkins/pytorch -name *.sh | xargs shellcheck --external-sources -P SCRIPTDIR "$@"
