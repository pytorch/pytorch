#!/bin/bash -xe

# One may want to invoke this script locally as follows:
#
#   .jenkins/run-shellcheck.sh --color=always | less -R


find .jenkins/pytorch -name '*.sh' -print0 | xargs -0 -n1 shellcheck --external-sources
