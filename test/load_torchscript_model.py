import sys
import torch

if __name__ == '__main__':
    print(torch.jit.load(sys.argv[1]))
    sys.exit(0)
