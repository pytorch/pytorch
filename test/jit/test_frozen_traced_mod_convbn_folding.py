import torch
import torch.nn as nn
from torch.testing._internal.jit_utils import JitTestCase

from torch.testing import FileCheck

import itertools

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestFrozenTracedModuleConv2dBNFolding(JitTestCase):
    def test_folding(self):
        # Test that we find Conv-BN patterns in submodules
        class SubModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(SubModule, self).__init__()
                layers = []
                for i in range(num_blocks):
                    layers.append(torch.nn.Conv2d(20, 20, 5, 1, bias=enable_bias))
                    bn_obj = torch.nn.BatchNorm2d(num_features=20, affine=enable_affine)
                    if enable_affine:
                        bn_obj.weight = torch.nn.Parameter(torch.rand_like(bn_obj.weight))
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    layers.append(bn_obj)
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                return self.layers(x)

        class TestModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(TestModule, self).__init__()
                self.sub = SubModule(num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                x = self.sub(x)
                return x

        bias_affine_options = itertools.product([True, False], [True, False], [1, 2])
        for (enable_bias, enable_bn_affine, num_layers) in bias_affine_options:
            eager = TestModule(num_layers, enable_bias, enable_bn_affine)
            eager.eval()

            x = torch.rand(1, 20, 10, 10)
            traced = torch.jit.trace(eager, x)
            traced.eval()

            traced_copy = traced.copy()
            torch._C._jit_pass_inline(traced_copy.graph)

            FileCheck().check_count("aten::batch_norm", num_layers, exactly=True) \
                .run(traced_copy.graph)

            traced._c = torch._C._freeze_module(traced._c)
            torch._C._jit_pass_fold_convbn_in_frozen_traced_module_graph(traced.graph)

            FileCheck().check_not("aten::batch_norm").run(traced.graph)

            x = torch.rand(1, 20, 10, 10)
            self.assertEqual(eager(x), traced(x))

    def test_no_folding(self):
        # Test that we find Conv-BN patterns in submodules
        class SubModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(SubModule, self).__init__()
                layers = []
                for i in range(num_blocks):
                    layers.append(torch.nn.Conv2d(20, 20, 5, 1, bias=enable_bias))
                    layers.append(torch.nn.ReLU())
                    bn_obj = torch.nn.BatchNorm2d(num_features=20, affine=enable_affine)
                    if enable_affine:
                        bn_obj.weight = torch.nn.Parameter(torch.rand_like(bn_obj.weight))
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    layers.append(bn_obj)
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                return self.layers(x)

        class TestModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(TestModule, self).__init__()
                self.sub = SubModule(num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                x = self.sub(x)
                return x

        bias_affine_options = itertools.product([True, False], [True, False], [1, 2])
        for (enable_bias, enable_bn_affine, num_layers) in bias_affine_options:
            eager = TestModule(num_layers, enable_bias, enable_bn_affine)
            eager.eval()

            x = torch.rand(1, 20, 10, 10)
            traced = torch.jit.trace(eager, x)
            traced.eval()

            traced_copy = traced.copy()
            torch._C._jit_pass_inline(traced_copy.graph)

            FileCheck().check_count("aten::batch_norm", num_layers, exactly=True) \
                .run(traced_copy.graph)

            traced._c = torch._C._freeze_module(traced._c)
            torch._C._jit_pass_fold_convbn_in_frozen_traced_module_graph(traced.graph)

            FileCheck().check_count("aten::batch_norm", num_layers, exactly=True) \
                .run(traced_copy.graph)

            x = torch.rand(1, 20, 10, 10)
            self.assertEqual(eager(x), traced(x))

    def test_mixing_scripted_and_frozen_traced_module_folding(self):
        # Test that we find Conv-BN patterns in submodules
        class SubModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(SubModule, self).__init__()
                layers = []
                for i in range(num_blocks):
                    layers.append(torch.nn.Conv2d(20, 20, 5, 1, bias=enable_bias))
                    bn_obj = torch.nn.BatchNorm2d(num_features=20, affine=enable_affine)
                    if enable_affine:
                        bn_obj.weight = torch.nn.Parameter(torch.rand_like(bn_obj.weight))
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    layers.append(bn_obj)
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                return self.layers(x)

        class TestModule(torch.nn.Module):
            def __init__(self, num_blocks, enable_bias, enable_affine):
                super(TestModule, self).__init__()
                self.sub = SubModule(num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                x = self.sub(x)
                return x

        bias_affine_options = itertools.product([True, False], [True, False], [True, False], [1, 2])
        for (enable_scripting, enable_bias, enable_bn_affine, num_layers) in bias_affine_options:
            eager = TestModule(num_layers, enable_bias, enable_bn_affine)
            eager.eval()

            if enable_scripting:
                traced_or_scripted = torch.jit.script(eager)
            else:
                x = torch.rand(1, 20, 10, 10)
                traced_or_scripted = torch.jit.trace(eager, x)
                traced_or_scripted.eval()

            traced_or_scripted_copy = traced_or_scripted.copy()
            torch._C._jit_pass_inline(traced_or_scripted_copy.graph)

            FileCheck().check_count("aten::batch_norm", num_layers, exactly=True) \
                .run(traced_or_scripted_copy.graph)

            # The use case to test is, we dont know if the model is traced or script.
            # So we apply both forms of folding.
            # 1. Apply batch norm folding that works for scripted modules.
            # 2. Freeze module.
            # 3. Apply batch norm folding that works for frozen traced modules.
            # Regardless of the input model being scripted or traced, we should have
            # eliminated batch_norm.

            traced_or_scripted._c = torch._C._jit_pass_fold_convbn(traced_or_scripted._c)
            traced_or_scripted._c = torch._C._freeze_module(traced_or_scripted._c)
            torch._C._jit_pass_fold_convbn_in_frozen_traced_module_graph(traced_or_scripted.graph)

            FileCheck().check_not("aten::batch_norm").run(traced_or_scripted.graph)

            x = torch.rand(1, 20, 10, 10)
            self.assertEqual(eager(x), traced_or_scripted(x))

