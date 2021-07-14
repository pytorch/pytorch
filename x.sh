#!/usr/bin/bash

set -eux
python3 -m tools.linter.clang_tidy --verbose
    
