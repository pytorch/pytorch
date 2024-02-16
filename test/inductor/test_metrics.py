# Owner(s): ["module: inductor"]
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import metrics
from torch._inductor.wrapper_benchmark import get_kernel_category_by_source_code

example_kernel = """
@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={
        'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'},
        'device': 0,
        'device_type': 'cuda',
        'constants': {},
        'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
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


class TestMetrics(TestCase):
    def test_parse_proper_kernel_fn_code(self):
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(example_kernel)
        assert proper_kernel_fn_code.startswith("def ")

    def test_count_args(self):
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(example_kernel)
        self.assertEqual(6, metrics._count_args(proper_kernel_fn_code))

    def test_count_pattern(self):
        proper_kernel_fn_code = metrics._parse_proper_kernel_fn_code(example_kernel)
        self.assertEqual(2, metrics._count_pattern(proper_kernel_fn_code, "tl.load"))
        self.assertEqual(1, metrics._count_pattern(proper_kernel_fn_code, "tl.store"))
        self.assertEqual(1, metrics._count_pattern(proper_kernel_fn_code, "for "))

    def test_parse_reduction_hint(self):
        kernel_category = get_kernel_category_by_source_code(example_kernel)
        self.assertEqual("reduction", kernel_category)
        self.assertEqual(
            "INNER", metrics._parse_reduction_hint(kernel_category, example_kernel)
        )


if __name__ == "__main__":
    run_tests()
