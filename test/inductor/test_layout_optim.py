# Owner(s): ["module: inductor"]
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch._dynamo.test_case import run_tests, TestCase
from torch import nn
import torch

class TestLayoutOptim(TestCase):
    def test_graph_break(self): # not fully repro
        """
        Make sure graph break does not cause any accuracy issue.
        The numerical may not match if the output of upstream graph's stride does
        not match downstream graph's assumption for its inputs.
        """
        class Model(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.conv1 = nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
                self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                x = self.conv1(x)
                torch._dynamo.graph_break()
                x = self.conv2(x)
                return x

        # TODO use verify_accuracy_for_model
        mod = Model().to("cuda")
        inp = torch.rand(2, 3, 16, 16).to("cuda")
        expected_out = mod(inp)
        opt_mod = torch.compile(mod)
        actual_out = opt_mod(inp)
        self.assertTrue(torch.allclose(expected_out, actual_out), f"expected:\n{expected_out}\nactual:\n{actual_out}") 

    def verify_accuracy_for_model(self, model_class, use_ddp_wrapper=False):
        # there are 2 potential ways to introduce graph breaks
        # 1. manually
        # 2. using DDP
        # if we are not using DDP to introduce graph breaks, do that manually
        manual_graph_break = not use_ddp_wrapper
        mod = model_class(manual_graph_break=manual_graph_break).cuda()
        inp = [t.cuda() for t in mod.get_example_inputs()]
        expected_out = mod(*inp)

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
        self.assertTrue(torch.allclose(expected_out, actual_out), f"expected:\n{expected_out}\nactual:\n{actual_out}") 


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

        self.verify_accuracy_for_model(Model, use_ddp_wrapper=False)


if __name__ == "__main__":
    if HAS_CUDA and not TEST_WITH_ROCM:
        run_tests()
