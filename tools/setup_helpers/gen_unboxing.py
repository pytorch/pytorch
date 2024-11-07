# Little stub file to get BUILD.bazel to play along

import os.path
import sys


root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import tools.jit.gen_unboxing


tools.jit.gen_unboxing.main(sys.argv[1:])
