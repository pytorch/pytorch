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
    assert len(sys.argv) == 2
    root_dir = examples.__name__.replace(".", "/")
    assert os.path.exists(root_dir)
    with open(os.path.join(root_dir, sys.argv[1] + ".py"), "w") as f:
        print("Writing to", f.name, "...")
        f.write(TEMPLATE.format(case_name=sys.argv[1]))
