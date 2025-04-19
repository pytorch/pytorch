# Owner(s): ["module: inductor"]

import torch
import torch.utils.flop_counter
from torch._inductor.debug import DebugContext
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)


def get_total_flops(mode):
    return sum(v for _, v in mode.flop_counts["Global"].items())


def random_tensor(size, dtype, **kwargs):
    if dtype in [torch.half, torch.bfloat16, torch.float, torch.double]:
        return torch.randn(size, dtype=dtype, **kwargs)
    elif dtype in [torch.uint8, torch.int8, torch.short, torch.int, torch.long]:
        return torch.randint(0, 100, size, dtype=dtype, **kwargs)
    else:
        raise ValueError("Unsupported data type")


def cT(device, dtype):
    def T(*shape, requires_grad=False):
        return random_tensor(
            shape, requires_grad=requires_grad, device=device, dtype=dtype
        )

    return T


class TestScheduler(TestCase):
    @dtypes(torch.float, torch.double)
    def test_flop_counter_op(self, device, dtype):
        T = cT(device, dtype)

        def composite(x, y, z):
            tmp = torch.mm(x + 10, y / 12)
            return torch.mm(tmp, z)

        def composite_relu(x, y):
            tmp = torch.mm(x, y)
            return torch.relu(tmp)

        test_cases = [
            (torch.mm, [T(4, 5), T(5, 6)], {}),
            (torch.add, [T(4, 5), T(4, 5)], {}),
            (composite, [T(5, 4), T(4, 3), T(3, 12)], {}),
            (composite_relu, [T(5, 4), T(4, 3)], {}),
        ]
        for op, example_inputs, kwargs in test_cases:
            comp = torch.compile(op)
            with FlopCounterMode() as mode:
                comp(*example_inputs, **kwargs)
            gm = make_fx(op)(*example_inputs, **kwargs)
            reference_flops = get_total_flops(mode)

            graph = GraphLowering(gm)

            with V.set_graph_handler(graph), V.set_debug_handler(DebugContext()):
                graph.run(*example_inputs, **kwargs)
                graph.init_wrapper_code()
                graph._update_scheduler()
                scheduler_flops = 0
                for node in graph.scheduler.nodes:
                    flops = node.estimate_flops()
                    scheduler_flops += flops if flops is not None else 0
            self.assertEqual(reference_flops, scheduler_flops, msg=f"op = {op}")


instantiate_device_type_tests(TestScheduler, globals())

if __name__ == "__main__":
    run_tests()
