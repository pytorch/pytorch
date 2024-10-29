import unittest
import torch
import torch._functorch
import torch._functorch.vmap

## Commented tests are ones that result in a segmentation fault

class test_bugfix(unittest.TestCase):
    # def test_original_bug(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
    #     level = 0
    #     batch_size = 0
    #     out_dim = 1250999896764
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

    # def test_positive_level(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
    #     level = 1
    #     batch_size = 0
    #     out_dim = 1250999896764
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

    # def test_negative_level(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
    #     level = -1
    #     batch_size = 0
    #     out_dim = 1250999896764
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

    # def test_nonzero_batch(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
    #     level = 0
    #     batch_size = 1
    #     out_dim = 1250999896764
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

    # def test_negative_batch(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
    #     level = 0
    #     batch_size = -1
    #     out_dim = 1250999896764
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

    def test_small_outdim(self):
        self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
        level = 0
        batch_size = 0
        out_dim = 3
        torch._remove_batch_dim(self, level, batch_size, out_dim)

    def test_large_level(self):
        self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
        level = 1200
        batch_size = 0
        out_dim = 3
        torch._remove_batch_dim(self, level, batch_size, out_dim)

    def test_neg_level(self):
        self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
        level = -12
        batch_size = 0
        out_dim = 3
        torch._remove_batch_dim(self, level, batch_size, out_dim)

    # def test_med_neg_outdim(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 1, dtype=torch.float64, requires_grad=False)
    #     level = 0
    #     batch_size = 0
    #     out_dim = -10
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

    def test_small_neg_batch(self):
        self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 1, dtype=torch.float64, requires_grad=False)
        level = 0
        batch_size = -1 ## Works at -1 but not -100 (integer overflow)
        out_dim = 1
        torch._remove_batch_dim(self, level, batch_size, out_dim)

    # def test_med_neg_batch(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 1, dtype=torch.float64, requires_grad=False)
    #     level = 0
    #     batch_size = -2 ## Works at -1 but not -100 (integer overflow)
    #     out_dim = 1
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

    # def test_negative_outdim(self):
    #     self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 3.5e+35, dtype=torch.float64, requires_grad=False)
    #     level = 0
    #     batch_size = 0
    #     out_dim = -1
    #     torch._remove_batch_dim(self, level, batch_size, out_dim)

if __name__ == '__main__':
    self = torch.full((1, 1, 1, 1, 1, 1, 1, 1, 1, 1,), 5., dtype=torch.float64, requires_grad=False)
    level = 0
    batch_size = -15
    out_dim = 1
    print(self)
    self = torch._remove_batch_dim(self, level, batch_size, out_dim)
    print(self)
    # unittest.main()


# If outdim not feasible, get iterator out of bounds (segfault in release build)