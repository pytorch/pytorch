import torch
import sys
from torch.jit.mobile import _load_for_lite_interpreter
import time

def main():
    x = sys.argv[1]
    total = 0
    for i in range(1):
        start = time.time()
        m = _load_for_lite_interpreter(sys.argv[1])
        end = time.time()
        total += (end - start)

    print('Total time measured in Python: ', total / 1, 'seconds')


if __name__ == '__main__':
    main()
