import torch
from torch.utils.flop_counter import FlopCounterMode
import torch._functorch.config as config
from torch.testing._internal.common_utils import run_tests, TestCase
torch.set_default_device('cuda')

def compile_with_ac(f, memory_budget):
    return torch.compile(f, backend="aot_eager_decomp_partition")

def get_act_mem(f):
    f().backward()
    start_mem = torch.cuda.memory_allocated()
    out = f()
    act_mem = (torch.cuda.memory_allocated() - start_mem) / (1024 * 1024)
    out.backward()
    return act_mem

def get_bw_flops(f):
    f().backward()
    out = f()
    with FlopCounterMode(display=False) as mode:
        out.backward()
    return mode.get_total_flops() / (1024 * 1024)

def get_mem_and_flops(f, memory_budget=None):
    # Returns megabytes rounded to 1 decimal point and FLOPs
    if memory_budget is not None:
        torch._dynamo.reset()
        with config.patch(memory_budget=memory_budget):
            f = torch.compile(f, backend="aot_eager_decomp_partition")

    return round(get_act_mem(f), 1), get_bw_flops(f)

class MemoryBudgetTest(TestCase):
    def test_rematerializes_cheap(self):
        def f(x, w):
            x = x.cos()
            x = torch.mm(x, w)
            return x.sum()
        x = torch.randn(512, 512, requires_grad=True)
        w = torch.randn(512, 512, requires_grad=True)
        call = lambda: f(x, w)
        eager_mem, eager_flops = get_mem_and_flops(call)
        self.assertEqual(eager_mem, 1.0)
        mem_10, flops_10 = get_mem_and_flops(call, memory_budget=1.0)
        self.assertEqual(mem_10, 1.0)
        self.assertEqual(eager_flops, flops_10)
        breakpoint()
        pass

if __name__ == "__main__":
    run_tests()
# def f(x, w1, w2):
#     x = torch.mm(x, w1)
#     x = x.cos()
#     x = torch.mm(x, w2)
#     return x.sum()

# x = torch.randn(512, 512, requires_grad=True)
# w1 = torch.randn(512, 512, requires_grad=True)
# w2 = torch.randn(512, 512, requires_grad=True)
# print(get_act_mem(lambda: f(x, w1, w2)))
# print(get_mem_and_flops(lambda: f(x, w1, w2), memory_budget=0.0))
# f1 = compile_with_ac(f, memory_budget=1.0)
# print(get_act_mem(lambda: f1(x, w1, w2)))
# print(get_bw_flops(lambda: f1(x, w1, w2)))
# f2 = compile_with_ac(f, memory_budget=0.0)
# print(get_act_mem(lambda: f2(x, w1, w2)))
# print(get_bw_flops(lambda: f2(x, w1, w2)))
