import torch
from test_indexing import *


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        run_tests()
    else:
        print("Skipping test_indexing_cuda.py")
