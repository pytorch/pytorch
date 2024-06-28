# Little stub file to get BUILD.bazel to play along

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torchgen.gen


torchgen.gen.main()
