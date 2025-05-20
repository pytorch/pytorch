# Owner(s): ["module: inductor"]

import torch
import torch._inductor.metrics as metrics
import torch.utils.flop_counter
from torch._dynamo.utils import counters
from torch._inductor.ir import FixedLayout
from torch._inductor.utils import fresh_inductor_cache
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    skipCUDAIf,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


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


inductor_metrics_log = torch._logging.getArtifactLogger(__name__, "inductor_metrics")


def test_cases(device, dtype):
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
    return test_cases


class TestScheduler(TestCase):
    @dtypes(torch.float, torch.float16)
    def test_get_estimated_runtime_logging(self, device, dtype):
        tc = test_cases(device, dtype)
        expected_metrics = [
            # num_bytes_accessed, node_runtimes, nodes_num_elem
            (148, (0,), (37,)),
            (120, (0,), (30,)),
            (444, (0, 0, 0, 0), (20, 12, 23, 55)),
            (154, (0, 0), (23, 15)),
        ]
        tc_plus_metrics = zip(tc, expected_metrics)
        # turn off logging of inductor metrics so that they don't get logged
        torch._logging.set_logs(inductor_metrics=False)
        for tc, met in tc_plus_metrics:
            op, example_inputs, kwargs = tc
            enba, enr, enne = met
            comp = torch.compile(op)
            torch._dynamo.reset()
            with fresh_inductor_cache():
                comp(*example_inputs, **kwargs)
            self.assertEqual(enba, 0)
            self.assertEqual(enr, 0)
            self.assertEqual(enne, 0)
            metrics.reset()

        breakpoint()
        torch._logging.set_logs(inductor_metrics=True)
        for tc, met in tc_plus_metrics:
            op, example_inputs, kwargs = tc
            enba, enr, enne = met

            comp = torch.compile(op)
            # next two lines are required, otherwise the flops will be cached from pervious runs of this function.
            torch._dynamo.reset()
            with fresh_inductor_cache():
                # actually run to set the counters
                comp(*example_inputs, **kwargs)
            self.assertEqual(enba, metrics.num_bytes_accessed)
            self.assertEqual(enr, tuple(m[1] for m in metrics.node_runtimes))
            self.assertEqual(enne, tuple(m[1] for m in metrics.nodes_num_elem))
            metrics.reset()

    @dtypes(torch.float, torch.float16)
    @skipCUDAIf(not SM70OrLater, "GPU capability is < SM70")
    @parametrize(
        "options",
        [
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "force_disable_caches": True,
            },
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON,ATEN",
                "force_disable_caches": True,
            },
        ],
    )
    def test_flop_counter_op(self, device, dtype, options):
        if device == "cpu":
            return
        if (
            options["max_autotune_gemm_backends"] == "TRITON"
            and torch.cuda.is_available()
            and not torch._inductor.utils.use_triton_template(
                FixedLayout(torch.device("cuda"), torch.float16, [400, 800])
            )
        ):
            return

        tc = test_cases(device, dtype)

        for op, example_inputs, kwargs in tc:
            comp = torch.compile(op, options=options)
            # next two lines are required, otherwise the flops will be cached from pervious runs of this function.
            torch._dynamo.reset()
            with fresh_inductor_cache():
                # actually run to set the counters
                comp(*example_inputs, **kwargs)
                with FlopCounterMode() as mode:
                    comp(*example_inputs, **kwargs)
            reference_flops = get_total_flops(mode)

            self.assertEqual(
                reference_flops,
                counters["inductor"]["flop_count"],
                msg=f"op = {op} reference flops = {reference_flops} != counters {counters['inductor']['flop_count']}",
            )
            if op != torch.add:
                self.assertNotEqual(reference_flops, 0, msg=f"op = {op} is 0 flops")
            counters["inductor"]["flop_count"] = 0


instantiate_device_type_tests(TestScheduler, globals())

if __name__ == "__main__":
    run_tests()
