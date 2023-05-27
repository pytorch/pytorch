# Owner(s): ["module: inductor"]
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch._dynamo.test_case import run_tests, TestCase
from torch import nn
import torch
import os
import copy
from torch._dynamo.utils import same

USE_DDP_WRAPPER = os.environ.get("USE_DDP_WRAPPER") == "1"

class TestLayoutOptim(TestCase):

    def verify_accuracy_for_model(self, model_class, use_ddp_wrapper=USE_DDP_WRAPPER):
        # there are 2 potential ways to introduce graph breaks
        # 1. manually
        # 2. using DDP
        # if we are not using DDP to introduce graph breaks, do that manually
        manual_graph_break = not use_ddp_wrapper
        mod = model_class(manual_graph_break=manual_graph_break).cuda()
        inp = [t.cuda() for t in mod.get_example_inputs()]
        expected_out = mod(*inp)

        fp64_mod = copy.deepcopy(mod).to(torch.float64)
        fp64_inp = [t.to(torch.float64) for t in copy.deepcopy(inp)]
        fp64_out = fp64_mod(*fp64_inp)

        if use_ddp_wrapper:
            import torch.distributed as dist
            port = 10001
            dist.init_process_group(backend='nccl',
                init_method=f'tcp://localhost:{port}',
                world_size=1, rank=0)
            from torch.nn.parallel import DistributedDataParallel as DDP

            wrapped_mod = DDP(mod)
            opt_mod = torch.compile(wrapped_mod)
        else:
            opt_mod = torch.compile(mod)
        actual_out = opt_mod(*inp)

        expected_sum = expected_out.sum()
        actual_sum = actual_out.sum()
        print(f"Expected sum {expected_sum}, actual sum {actual_sum}")
        self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))

    def test_2conv_with_graph_break(self):
        """
        Make sure graph break does not cause any accuracy issue.
        """
        class Model(nn.Module):
            def __init__(self, dim=512, manual_graph_break=False):
                super().__init__()
                self.conv1 = nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
                self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)
                self.manual_graph_break = manual_graph_break

            def forward(self, x):
                x = self.conv1(x)
                if self.manual_graph_break:
                    torch._dynamo.graph_break()
                x = self.conv2(x)
                return x

            def get_example_inputs(self):
                return torch.rand(2, 3, 16, 16), 

        self.verify_accuracy_for_model(Model)

    def test_3conv_with_graph_break(self):
        class Model(nn.Module):
            def __init__(self, dim=512, patch_size=7, kernel_size=7, manual_graph_break=False):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False),
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same", bias=False),
                )
                self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
                self.manual_graph_break = manual_graph_break

            def forward(self, x):
                x = self.seq(x)
                if self.manual_graph_break:
                    torch._dynamo.graph_break()
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                return torch.randn(2, 3, 16, 16),

        self.verify_accuracy_for_model(Model)

    def test_keep_output_layout_infer(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1, bias=False)

            def forward(self, x):
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                return torch.randn(2, 3, 5, 5),

        mod = Model().cuda()
        inp = [t.cuda() for t in mod.get_example_inputs()]
        out = mod(*inp)

        opt_mod = torch.compile(mod)
        opt_out = opt_mod(*inp)

        # We should be able to do view on eager output
        out.view(5, -1)

        # We should be able to do view on the output of the optimized module
        # Note that if the output is channels last, the view op will fail.
        opt_out.view(5, -1)

    def test_keep_output_layout_train(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 128, kernel_size=3, padding=1, stride=1, bias=False)
                self.linear = nn.Linear(128 * 5 * 5, 10)

            def forward(self, x):
                x = self.conv(x)
                x = torch.flatten(x, 1)
                x = self.linear(x)
                return x

            def get_example_inputs(self):
                return torch.randn(2, 3, 5, 5),

        mod = Model().cuda()
        inp = [t.cuda() for t in mod.get_example_inputs()]
        def f(inp):
            x = mod(*inp)
            x.sum().backward()
           
        opt_f = torch.compile(f)
        f(inp)
        opt_f(inp)
        assert False, "ni"  # TODO


if __name__ == "__main__":
    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests()
