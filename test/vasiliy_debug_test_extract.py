# not for land
# test harness for vasiliy-debug

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch._dynamo.vasiliy_debug_extract_subgraphs import debug_linears_for_float8
from torch._dynamo.vasiliy_debug_analyze_subgraphs import analyze_subgraphs

target_folder = '/home/vasiliy/local/tmp/20240802_dynamo_test'

def _test_impl(
    m: torch.nn.Module, 
    args, 
    num_subgraphs=1, 
    validate_subgraph_logs=True, 
    validate_skip_logs=False,
    linear_mod_filter_fn=None,
):
    # disable nn module inlining, our subgraph extraction logic depends on this
    torch._dynamo.config.inline_inbuilt_nn_modules = False

    # add a custom pass to inspect the pre-dispatch subgraphs and
    # extract linear microbenchmark info from them
    # context: https://github.com/pytorch/pytorch/pull/113823
    torch._inductor.config.pre_grad_custom_pass = \
        lambda g: debug_linears_for_float8(g, target_folder, linear_mod_filter_fn)

    # run the extraction
    m = torch.compile(m)
    out = m(*args)

    if validate_skip_logs:
        assert os.path.isfile(os.path.join(target_folder, 'skip_logs.txt'))

    if not validate_subgraph_logs:
        return

    # verify debug logs and summary got saved
    assert os.path.isfile(os.path.join(target_folder, 'debug_logs.txt'))
    assert os.path.isfile(os.path.join(target_folder, 'summary.csv'))

    # verify all expected subgraphs got saved
    for subgraph_idx in range(num_subgraphs):
        subgraph_filename = os.path.join(target_folder, f'subgraph_with_inputs_{subgraph_idx}.pt')
        assert os.path.isfile(subgraph_filename)

        # open subgraph and verify runnable
        gm, inputs = torch.load(subgraph_filename, weights_only=False)
        output = gm(*inputs)
        output = torch.cat([*output], dim=0)
        output.sum().backward()


def test_prev_mod():
    x = torch.randn(4, 4, device='cuda')
    m = nn.Sequential(nn.ReLU(), nn.Linear(4, 4)).cuda()
    _test_impl(m, (x,))

def test_prev_fun_mul():

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x, y):
            z = x * y
            z = self.fc(z)
            return z

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y))

def test_prev_fun_addcmul():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x, y, z):
            a = torch.addcmul(x, y, z)
            a = self.fc(a)
            return a

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    z = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y, z))

def test_prev_fun_addcmul_shared_input():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x, y):
            a = torch.addcmul(x, y, y)
            a = self.fc(a)
            return a

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y))

def test_prev_fun_layernorm():
    
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(4)
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            x = F.layer_norm(x, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)
            x = self.fc(x)
            return x

    x = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x,))
    
def test_next_fun_add_scalar():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            # Note: relu is here because linear as first op is not supported yet
            x = self.relu(x)
            x = self.fc(x)
            x = x + 1
            return x

    x = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x,))

def test_next_fun_add_tensor():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(4, 4)

        def forward(self, x, y):
            # Note: relu is here because linear as first op is not supported yet
            x = self.relu(x)
            x = self.fc(x)
            x = x + y
            return x

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y))

def test_next_mod_sigmoid():
    x = torch.randn(4, 4, device='cuda')
    m = nn.Sequential(nn.ReLU(), nn.Linear(4, 4), nn.Sigmoid()).cuda()
    _test_impl(m, (x,))

def test_next_fun_layernorm():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.ln = nn.LayerNorm(4)
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            x = self.relu(x)
            x = self.fc(x)
            x = F.layer_norm(x, self.ln.normalized_shape, self.ln.weight, self.ln.bias, self.ln.eps)
            return x

    x = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x,))

def test_next_fun_addcmul():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(4, 4)

        def forward(self, x, y, z):
            x = self.relu(x)
            x = self.fc(x)
            x = torch.addcmul(x, y, z)
            return x

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    z = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y, z))

def test_next_fun_addcmul_shared_input():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(4, 4)

        def forward(self, x, y):
            x = self.relu(x)
            x = self.fc(x)
            x = torch.addcmul(x, y, y)
            return x

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y))

def test_next_fun_cat():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.fc = nn.Linear(4, 4)

        def forward(self, x, y):
            x = self.relu(x)
            x = self.fc(x)
            x = torch.cat([x, y], dim=0)
            return x

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y))

def test_dual_linear():
    x = torch.randn(4, 4, device='cuda')
    m = nn.Sequential(nn.ReLU(), nn.Linear(4, 4), nn.Linear(4, 8), nn.Sigmoid()).cuda()
    _test_impl(m, (x,))

def test_addcmul_dual_linear():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)

        def forward(self, x, y):
            addcmul = torch.addcmul(x, x, y)
            fc1 = self.fc1(addcmul)
            fc2 = self.fc2(fc1)
            x = torch.addcmul(addcmul, x, fc2)
            return x

    x = torch.randn(4, 4, device='cuda')
    y = torch.randn(4, 4, device='cuda')
    m = M().cuda()
    _test_impl(m, (x, y))


def test_multiple_subgraphs():
    x = torch.randn(4, 4, device='cuda')
    m = nn.Sequential(nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 8), nn.Sigmoid()).cuda()
    _test_impl(m, (x,), num_subgraphs=2)

def test_placeholder_linear():
    x = torch.randn(4, 4, device='cuda')
    m = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).cuda()
    _test_impl(m, (x,), validate_subgraph_logs=False)
    
def test_skip_linear():
    x = torch.randn(4, 4, device='cuda')
    m = nn.Sequential(nn.ReLU(), nn.Linear(4, 4), nn.Sigmoid()).cuda()
    linear_mod_filter_fn = lambda mod: mod.in_features > 16
    _test_impl(
        m, (x,), linear_mod_filter_fn=linear_mod_filter_fn, 
        validate_subgraph_logs=False,
        validate_skip_logs=True,
    )
