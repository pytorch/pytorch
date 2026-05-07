import os
import sys

import torch._export.db.examples as examples

TEMPLATE = '''import torch

def {case_name}(x):
    """
    """

    return
'''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise AssertionError(f"expected 2 arguments, got {len(sys.argv)}")
    root_dir = examples.__name__.replace(".", "/")
    if not os.path.exists(root_dir):
        raise AssertionError(f"root_dir does not exist: {root_dir}")
    with open(os.path.join(root_dir, sys.argv[1] + ".py"), "w") as f:
        print("Writing to", f.name, "...")
        f.write(TEMPLATE.format(case_name=sys.argv[1]))
