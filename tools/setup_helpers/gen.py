# Little stub file to get BUILD.bazel to play along

import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3 - 1]
sys.path.insert(0, str(REPO_ROOT))

import torchgen.gen


torchgen.gen.main()
