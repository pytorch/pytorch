from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
import torch
from torch import Tensor
import torch.nn.functional as F
import functools
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_dtype import integral_types
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.testing._internal.common_methods_invocations import op_db

aten = torch.ops.aten


def diff_arg(arg, requires_grad=True):
    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()
    if is_iterable_of_tensors(arg):
        if all([is_differentiable_arg(a) for a in arg]):
            return True
        if all([not is_differentiable_arg(a) for a in arg]):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_input_output2(f, args, kwargs, output_process_fn_grad=None, requires_grad=True):
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(i for i, arg in enumerate(flat_args) if diff_arg(arg, requires_grad=requires_grad))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        # import pdb; pdb.set_trace()
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            # TODO: Remove the following hack for namedtuples
            result = tuple(result)
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f, args, sample.kwargs, sample.output_process_fn_grad, requires_grad=requires_grad
    )

def is_inplace(op, variant):
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()

class InplaceError(Exception):
    def __repr__(self):
        return "Decomposition Tensor with no elem was created (probably due to an in-place op)"


def ref_vjp_no_create(f, *primals):
    result = f(*primals)

    return result, None


run_decompositions = set()
run_ops = set()


class TestDecompositionOpInfo(TestCase):

    @ops(
        op_db,
        allowed_dtypes=[torch.float32, torch.float64, torch.float16, torch.bfloat16] + [*integral_types()]
    )
    # entries in here need don't work and need to be fixed.
    # Each one of these is a bug (or needs to be investigated)
    def test_decomposition(self, device, dtype, op):
        # dtype is too confusing of a name for how we're using it
        TEST_DTYPE = dtype
        class DecompositionTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ['elem']

            @staticmethod
            def __new__(cls, elem):
                r = torch.Tensor._make_wrapper_subclass(
                    cls, elem.size(),
                    strides=elem.stride(), storage_offset=elem.storage_offset(),
                    dtype=elem.dtype, layout=elem.layout,
                    device=elem.device, requires_grad=elem.requires_grad
                )

                r.elem = elem
                return r

            def __repr__(self):
                return f"DecompositionTensor(elem={self.elem})"

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

                def unwrap_tensor(e):
                    if isinstance(e, DecompositionTensor):
                        if not hasattr(e, 'elem'):
                            raise InplaceError()
                        return e.elem
                    return e

                real_out = func(*tree_map(unwrap_tensor, args), **tree_map(unwrap_tensor, kwargs))

                def wrap_tensor(e):
                    if e is None:
                        return DecompositionTensor(torch.empty(()))
                    return DecompositionTensor(e) if type(e) == torch.Tensor else e
                wrapped_out = tree_map(wrap_tensor, real_out)
                return wrapped_out

        if TEST_DTYPE not in op.supported_dtypes(self.device_type):
            self.skipTest("Dtype not in op's supported dtypes")
            return
        if is_inplace(op, op.get_op()):
            self.skipTest("op is inplace")
            return
        _requires_grad = op.supports_autograd and TEST_DTYPE.is_floating_point

        samples = op.sample_inputs(device, TEST_DTYPE, requires_grad=_requires_grad)

        # Acquires variants to test
        def wrap_tensor(x):
            if type(x) == torch.Tensor:
                return DecompositionTensor(x)
            return x

        try:
            func = op.get_op()
            for sample_input in samples:
                if _requires_grad:
                    fn, primals = normalize_op_input_output(func, sample_input)
                    primals = tree_map(lambda x: x.abs() if isinstance(x, torch.Tensor) else x, primals)

                    decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *tree_map(wrap_tensor, primals))
                    cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)

                    _ = decomp_vjp_fn(cotangents)

                else:
                    args = [sample_input.input] + list(sample_input.args)
                    kwargs = sample_input.kwargs
                    _ = func(*args, **kwargs)

                    args = tree_map(wrap_tensor, args)
                    kwargs = tree_map(wrap_tensor, kwargs)
                    decomp_out = func(*args, **kwargs)

        except InplaceError:
            self.skipTest("op is inplace")
            return
        except RuntimeError as e:
            if "not implemented for" in str(e):
                self.skipTest(str(e))
                return
            if "Mismatch in shape: grad_output" in str(e):
                self.skipTest("Some weird issue with autograd engine and tensor subclasses")
                return
            raise e

only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestDecompositionOpInfo, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
