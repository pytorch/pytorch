# Little stub file to get BUILD.bazel to play along

import os.path
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root)

import tools.jit.gen_unboxing

<<<<<<< HEAD
tools.jit.gen_unboxing.main()
=======
tools.jit.gen_unboxing.main()
>>>>>>> 298fdd1d82 ([PyTorch] Enable lightweight dispatch as an option in cmake build)
