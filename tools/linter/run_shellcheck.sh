#!/usr/bin/env bash
find "$@" -name '*.sh' -print0 | xargs -0 -n1 shellcheck --external-sources
