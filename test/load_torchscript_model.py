import sys

import torch

if __name__ == "__main__":
    script_mod = torch.jit.load(sys.argv[1])
    mod = torch.load(sys.argv[1] + ".orig")
    print(script_mod)
    inp = torch.rand(2, 28 * 28)
    _ = mod(inp)
    sys.exit(0)
