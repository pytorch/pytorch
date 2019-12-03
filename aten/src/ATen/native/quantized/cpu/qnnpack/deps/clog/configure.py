#!/usr/bin/env python
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import confu
parser = confu.standard_parser("clog configuration script")


def main(args):
    options = parser.parse_args(args)
    build = confu.Build.from_options(options)

    build.export_cpath("include", ["clog.h"])

    with build.options(source_dir="src", extra_include_dirs="src"):
        build.static_library("clog", build.cc("clog.c"))

    with build.options(source_dir="test", deps={
            (build, build.deps.googletest): all,
            "log": build.target.is_android}):
        build.unittest("clog-test", build.cxx("clog.cc"))

    return build

if __name__ == "__main__":
    import sys
    main(sys.argv[1:]).generate()
