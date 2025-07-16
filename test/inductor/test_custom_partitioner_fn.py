# Owner(s): ["module: inductor"]
import torch
from functorch.compile import min_cut_rematerialization_partition
from torch._C import FileCheck
from torch._inductor.custom_partitioner_fn import CustomPartitionerFn
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class MyCustomPartitionerFn(CustomPartitionerFn):
    """
    A custom partitioner function with static_lifetime_input_indices overwrites.
    """

    def __init__(self):
        super().__init__()
        self.called = False

    def __call__(self, gm, joint_inputs, **kwargs):
        self.called = True
        kwargs["static_lifetime_input_indices"] = [0, 1]
        return min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)

    def uuid(self):
        return None


class TestCustomPartitionerFn(TestCase):
    def test_custom_partitioner_fn(self):
        """
        For function f(a, b), with the  partitioner in the compile_fx stack,
        the addtion `a+b` (equivalently `buf0`) is saved for backward.
        With the custom partitioner function, we indicate that
        `a` and `b` (equivalently `primals_1` and `primals_2`) do not take
        additional memory and thus, they are saved for backward.
        """

        # initialization
        @torch.compile
        def f(a, b):
            return (a + b).cos().cos()

        a = torch.randn((2, 2), requires_grad=True, device=GPU_TYPE)
        b = torch.randn((2, 2), requires_grad=True, device=GPU_TYPE)

        # CASE 1 -- default
        # addtion `a + b` (i.e, `buf0`) is saved for backward.
        code_og = run_fw_bw_and_get_code(lambda: f(a, b))
        fwd_code_og = code_og[1][0]
        FileCheck().check("return (buf1, buf0, )").run(fwd_code_og)

        # CASE 2 -- custom partitioner function
        # `a` and `b` (i.e., `primals_1` and `primals_2`) are saved for backward.
        custom_partitioner_fn = MyCustomPartitionerFn()
        self.assertFalse(custom_partitioner_fn.called)

        with torch._inductor.config.patch(custom_partitioner_fn=custom_partitioner_fn):
            code_cp = run_fw_bw_and_get_code(lambda: f(a, b))
        fwd_code_cp = code_cp[1][0]
        FileCheck().check("return (buf0, primals_1, primals_2, )").run(fwd_code_cp)

        # make sure the custom partitioner function is indeed invoked
        self.assertTrue(custom_partitioner_fn.called)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
