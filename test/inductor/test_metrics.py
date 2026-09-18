# Owner(s): ["module: inductor"]
import torch
from torch._inductor import config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import collect_defined_kernels
from torch._inductor.wrapper_benchmark import get_kernel_category_by_source_code
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
from torch.testing._internal.common_utils import HardwareClassification
from torch.utils._triton import has_triton


example_kernel = """
@triton_heuristics.reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={
        'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'},
        'device': 0,
        'device_type': 'GPU_TYPE',
        'constants': {},
        'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={
        'autotune_hints': set(),
        'kernel_name': 'triton_red_fused_add_sum_2',
        'mutated_arg_names': ['in_out_ptr0'],
        'no_x_dim': False,
        'kernel_num_gb': 0.0083968
    }
)
@triton.jit
def triton_red_fused_add_sum_2(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tmp4 + tmp2
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
"""


def replace_device_type(kernel_src, device):
    device_type = torch.device(device).type
    return kernel_src.replace("GPU_TYPE", device_type)


class TestMetrics(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_parse_proper_kernel_fn_code(self, device):
        new_kernel = replace_device_type(example_kernel, device)
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(new_kernel)
        if not proper_kernel_fn_code.startswith("def "):
            raise AssertionError

    def test_count_args(self, device):
        new_kernel = replace_device_type(example_kernel, device)
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(new_kernel)
        self.assertEqual(6, metrics._count_args(proper_kernel_fn_code))

    def test_count_pattern(self, device):
        new_kernel = replace_device_type(example_kernel, device)
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(new_kernel)
        self.assertEqual(2, metrics._count_pattern(proper_kernel_fn_code, "tl.load"))
        self.assertEqual(1, metrics._count_pattern(proper_kernel_fn_code, "tl.store"))
        self.assertEqual(1, metrics._count_pattern(proper_kernel_fn_code, "for "))

    def test_parse_reduction_hint(self, device):
        new_kernel = replace_device_type(example_kernel, device)
        kernel_category = get_kernel_category_by_source_code(new_kernel)
        self.assertEqual("reduction", kernel_category)
        self.assertEqual(
            "INNER", metrics._parse_reduction_hint(kernel_category, new_kernel)
        )

    @config.patch("fx_graph_remote_cache", False)
    @config.patch("partitioned_scatter_enabled", False)
    def test_atomic_add(self, device):
        @torch.compile
        def f(lhs, index, rhs):
            return lhs.index_put_([index], rhs, accumulate=True)

        lhs = torch.randn(1024, device=device)
        index = torch.randint(0, 1024, [32], device=device, dtype=torch.int32)
        rhs = torch.randn(32, device=device)

        kernel_list = []
        with collect_defined_kernels(kernel_list):
            f(lhs, index, rhs)

        self.assertEqual(len(kernel_list), 1)
        kernel_code = kernel_list[0]
        self.assertEqual(metrics._count_pattern(kernel_code, "tl.atomic_add"), 1)

    @largeTensorTest(25e7 * 2 * 4, inductor=True)
    @config.patch("fx_graph_remote_cache", False)
    @config.patch("benchmark_kernel", True)
    def test_kernel_args_num_gb(self, device):
        @torch.compile
        def f(x):
            return x + 1

        x = torch.randn(int(25e7), device=device)
        kernel_list = []
        with collect_defined_kernels(kernel_list):
            f(x)

        self.assertEqual(len(kernel_list), 1)
        kernel_code = kernel_list[0]
        self.assertEqual(
            metrics._parse_kernel_args_num_gb(kernel_code, "pointwise"), 2.0
        )


instantiate_device_type_tests(TestMetrics, globals(), except_for="cpu", allow_xpu=True)


if __name__ == "__main__":
    if has_triton():
        run_tests()
