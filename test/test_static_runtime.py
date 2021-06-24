import numpy as np
import torch
import unittest

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

from typing import Dict, Optional

class StaticModule:
    def __init__(self, scripted):
        # this is an nn.Module
        if hasattr(scripted, "_c"):
            self.static_module = torch._C._jit_to_static_module(scripted._c)
        else:
            self.static_module = torch._C._jit_to_static_module(scripted.graph)

    def __call__(self, *args, **kwargs):
        if not kwargs:
            return self.static_module(args)
        else:
            return self.static_module(args, kwargs)

    def benchmark(self, args, kwargs, warmup_runs, main_runs):
        self.static_module.benchmark(args, kwargs, warmup_runs, main_runs)

    def benchmark_individual_ops(self, args, kwargs, warmup_runs, main_runs):
        return self.static_module.benchmark_individual_ops(
            args, kwargs, warmup_runs, main_runs
        )


def linear_shim(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    ret = output
    return ret


torch.nn.functional.linear = linear_shim


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        # self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        # x = torch.matmul(self.dropout(attention), V)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


# Taken from https://github.com/facebookresearch/dlrm/blob/master/dlrm_s_pytorch.py
def create_mlp(ln, sigmoid_layer):
    layers = nn.ModuleList()
    for i in range(0, len(ln) - 1):
        n = ln[i]
        m = ln[i + 1]

        LL = nn.Linear(int(n), int(m), bias=True)

        mean = 0.0  # std_dev = np.sqrt(variance)
        std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
        std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
        bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
        LL.weight.data = torch.tensor(W, requires_grad=True)
        LL.bias.data = torch.tensor(bt, requires_grad=True)
        layers.append(LL)

        if i == sigmoid_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU())

    with torch.no_grad():
        s = torch.jit.script(torch.nn.Sequential(*layers))
    s.eval()
    return s


def trivial_graph(a, b, c):
    s = torch.tensor([[3, 3], [3, 3]])
    return a + b * c + s

def loop_graph(a, b, iters : int):
    c = a + b * 2
    for i in range(iters):
        c = c + b
        c *= 2
        c -= a
    return c

def output_graph(a, b, c, iters : int):
    s = torch.tensor([[3, 3], [3, 3]])
    k = a + b * c + s
    d : Dict[int, torch.Tensor] = {}
    for i in range(iters):
        d[i] = k + i
    return d

class TestStaticModule(TestCase):
    def test_multihead_attention_layer(self):
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device("cpu")
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        attention.eval()
        attention = torch.jit.script(attention)
        attention.eval()
        o_ref = attention(src, src, src, src_mask)

        attention_a = StaticModule(attention)
        o_test = attention_a(src, src, src, src_mask)
        o_test_kw = attention_a(src, src, value=src, mask=src_mask)

        for a, b in zip(o_ref, o_test):
            torch.testing.assert_allclose(a, b)

        for a, b in zip(o_ref, o_test_kw):
            torch.testing.assert_allclose(a, b)

    def test_multihead_attention_layer_benchmark(self):
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device("cpu")
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        attention.eval()
        attention = torch.jit.script(attention)
        attention_a = StaticModule(attention)

        attention_a.benchmark([src, src, src, src_mask], {}, 2, 2)
        metrics = attention_a.benchmark_individual_ops(
            [src, src, src, src_mask], {}, 2, 2
        )

    def test_mlp(self):
        # Arguments taken from benchmark script, ./bench/dlrm_s_benchmark.sh
        ln_bot = [512, 512, 64]
        sigmoid_bot = -1
        ln_top = [100, 1024, 1024, 1024, 1]
        sigmoid_top = 3
        bot_l = create_mlp(ln_bot, sigmoid_bot)
        bot_l_acc = StaticModule(bot_l)
        top_l = create_mlp(ln_top, sigmoid_top)
        top_l_acc = StaticModule(top_l)
        with torch.no_grad():
            bot_inp = torch.randn(2048, 512)  # torch.Size([2048, 512])
            top_inp = torch.randn(2048, 100)  # torch.Size([2048, 100])
        ref_bot = bot_l(bot_inp)
        acc_bot = bot_l_acc(bot_inp)[0]
        torch.testing.assert_allclose(acc_bot, ref_bot)
        ref_top = top_l(top_inp)
        acc_top = top_l_acc(top_inp)[0]
        torch.testing.assert_allclose(acc_top, ref_top)
        for _ in range(5):
            with torch.no_grad():
                bot_inp = torch.randn(2048, 512)  # torch.Size([2048, 512])
                top_inp = torch.randn(2048, 100)  # torch.Size([2048, 100])
            ref_bot = bot_l(bot_inp)
            acc_bot = bot_l_acc(bot_inp)[0]
            torch.testing.assert_allclose(acc_bot, ref_bot)
            ref_top = top_l(top_inp)
            acc_top = top_l_acc(top_inp)[0]
            torch.testing.assert_allclose(acc_top, ref_top)

    def test_trivial_graph(self):
        s = torch.full((2, 2), 2)
        tg = torch.jit.script(trivial_graph)
        o_ref = tg(s, s, s)
        tg_a = StaticModule(tg)
        o_test = tg_a(s, s, s)[0]
        torch.testing.assert_allclose(o_ref, o_test)

    def test_leaky_relu(self):
        s = torch.randn(5, 5)
        tg = torch.jit.script(nn.LeakyReLU(0.1))
        o_ref = tg(s)
        tg_a = StaticModule(tg)
        o_test = tg_a(s)[0]
        torch.testing.assert_allclose(o_ref, o_test)

    @unittest.skip("Temporarily disabled")
    def test_fusion_trivial_graph(self):
        s = torch.full((2, 2), 2)
        tg = torch.jit.script(trivial_graph)
        o_ref = tg(s, s, s)
        torch._C._fuse_to_static_module(tg.graph)
        assert "StaticSubgraph" in str(tg.graph)
        o_test = tg(s, s, s)
        torch.testing.assert_allclose(o_ref, o_test)

    @unittest.skip("Temporarily disabled")
    def test_fusion_multihead_attention_layer(self):
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device("cpu")
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        attention.eval()
        attention = torch.jit.script(attention)
        attention.eval()
        o_ref = attention(src, src, src, src_mask)

        torch._C._fuse_to_static_module(attention._c)
        o_test = attention(src, src, src, src_mask)

        for a, b in zip(o_ref, o_test):
            torch.testing.assert_allclose(a, b)

    @unittest.skip("Temporarily disabled")
    def test_fusion_loop(self):
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        c = 4
        lg = torch.jit.script(loop_graph)
        o_ref = lg(a, b, c)
        torch._C._fuse_to_static_module(lg.graph)
        assert "StaticSubgraph" in str(lg.graph)
        o_test = lg(a, b, c)
        torch.testing.assert_allclose(o_ref, o_test)

    @unittest.skip("Temporarily disabled")
    def test_fusion_outputs(self):
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        c = 4
        og = torch.jit.script(output_graph)
        o_ref = og(a, b, b, c)
        torch._C._fuse_to_static_module(og.graph)
        assert "StaticSubgraph" in str(og.graph)
        o_test = og(a, b, b, c)
        for i in o_ref.keys():
            torch.testing.assert_allclose(o_ref[i], o_test[i])



if __name__ == "__main__":
    run_tests()
