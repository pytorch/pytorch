"""
Generates a single header file that may be included in cling/jupyter, which
has all the necessary pragmas for cling to see all further headers and libraries
of PyTorch.
"""

import argparse
import os.path

FILE_TEMPLATE = """#pragma once

// Include directories
{}

// Library directories
{}

// Libraries
{}

#include <torch/torch.h>
"""

# Mainly taken from https://github.com/QuantStack/xeus-cling/issues/87#issuecomment-349053121
INCLUDE_DIR_TEMPLATE = '#pragma cling add_include_path("{}")'
LIBRARY_DIR_TEMPLATE = '#pragma cling add_library_path("{}")'
LIBRARIES_TEMPLATE = '#pragma cling load("{}")'


parser = argparse.ArgumentParser(description="Generate cling_pragmas.h")
parser.add_argument("--include-dirs", nargs="+", required=True)
parser.add_argument("--library-dirs", nargs="+", required=True)
parser.add_argument("--libraries", nargs="+", required=True)
parser.add_argument("-o", "--output")
options = parser.parse_args()


def format_paths(template, paths):
    return [template.format(path) for path in paths]


include_dirs = format_paths(INCLUDE_DIR_TEMPLATE, options.include_dirs)
library_dirs = format_paths(LIBRARY_DIR_TEMPLATE, options.library_dirs)
libraries = format_paths(LIBRARIES_TEMPLATE, options.libraries)

source = FILE_TEMPLATE.format(
    "\n".join(include_dirs), "\n".join(library_dirs), "\n".join(libraries)
)

if options.output:
    with open(options.output, "w") as output_file:
        output_file.write(source)
else:
    print(source)
