# Owner(s): ["module: dynamo"]
import functools
import itertools
import unittest
from functools import partial

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import normalize_gm
from torch._functorch._aot_autograd.utils import make_boxed_compiler
from torch._functorch.compilers import min_cut_rematerialization_partition
from torch._higher_order_ops.wrap import wrap
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.nested._internal.nested_tensor import (
    jagged_from_list,
    jagged_from_tensor_and_lengths,
    nested_view_from_values_offsets,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    NestedTensorTestCase,
    parametrize,
    subtest,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._python_dispatch import return_and_correct_aliasing


def traceable_subclass(c):
    return torch._dynamo.config.patch("traceable_tensor_subclasses", {c})


def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
    actual_recompiles = _recompiles_for_inputs(fn, inputs1, inputs2)
    self.assertEqual(actual_recompiles, expected_recompiles)


def get_jagged_tensor(nested_size, offsets, requires_grad=True):
    # Makes a jagged tensor with N constituent tensors with size
    # as specified ((S0, S1, S2), D)
    D = nested_size[1]
    out = []
    for s in nested_size[0]:
        out.append(torch.randn(s, D, requires_grad=requires_grad, dtype=torch.float64))
    return jagged_from_list(out, offsets)


def get_view_test_cases():
    # Test all cases with both an NT base and a dense base
    # Subclass -> Subclass
    # Dense -> Subclass

    # NB: Don't close over loop variables, they will not get copied into the
    # closure
    #
    # NB: These return functions so we don't generate tensors during test
    # collection time

    def mk_basic(base_is_nt):
        # There are three cases to consider here based on the logic in
        # meta_utils.py
        #
        # (1) basic case:
        # view is not a leaf and has the same requires grad as its basic case
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)
        x = x.clone() if base_is_nt else x
        assert not x.is_leaf
        return x.unsqueeze(-1)

    def mk_leaf(base_is_nt, requires_grad_1, requires_grad_2):
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=requires_grad_1)
        x = x.clone() if base_is_nt else x
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
            # The issue is this doesn't quite work
            x_view.requires_grad_(requires_grad_2)

        return x_view

    def mk_obscure(base_is_nt):
        x, _ = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=False)
        x = x.clone() if base_is_nt else x
        # intermediate leaf view
        with torch.no_grad():
            x_view = x.unsqueeze(-1)
        x_view.requires_grad_(True)
        x_view_view = x_view.unsqueeze(-1)
        return x_view_view

    for base_is_nt in [False, True]:
        prefix = f"base_is_nt_{base_is_nt}"

        yield partial(mk_basic, base_is_nt), f"{prefix}_basic"

        # (2) leaf view case:
        # the view has to be a leaf (w/ requires_grad True or requires_grad False)
        # base w/ requires_grad True or requires_grad False
        for requires_grad_1, requires_grad_2 in itertools.product(
            [True, False], repeat=2
        ):
            yield partial(
                mk_leaf, base_is_nt, requires_grad_1, requires_grad_2
            ), f"{prefix}_leaf_{requires_grad_1}_{requires_grad_2}"

        # (3) obscure case:
        # view is not a leaf (implies requires_grad True)
        # base w/ requires_grad False)
        yield partial(mk_obscure, base_is_nt), f"{prefix}_obscure"

    # Subclass -> Dense
    yield lambda: get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[
        0
    ].clone(), "subclass_dense"

    # Dense -> Subclass -> Dense -> Subclass
    def mk_dense_subclass_dense_subclass():
        values = torch.randn(10, 5)
        offsets = torch.tensor([0, 3, 6, 10])
        offsets2 = offsets.clone().detach()
        return nested_view_from_values_offsets(
            nested_view_from_values_offsets(values, offsets).values(), offsets
        )

    yield mk_dense_subclass_dense_subclass, "dense_subclass_dense_subclass"

    def mk_subclass_dense_subclass_dense():
        x = get_jagged_tensor(((2, 3, 4), 3), None, requires_grad=True)[0].clone()
        offsets2 = x.offsets().clone().detach()
        nt_view = nested_view_from_values_offsets(x.values(), offsets2).values()

    yield mk_subclass_dense_subclass_dense, "subclass_dense_subclass_dense"


VIEW_TEST_CASES = {k: v for v, k in get_view_test_cases()}


requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")

compile_full_eager = torch.compile(backend="eager", fullgraph=True)


class BaseTorchFunction(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)


class MockSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


class AttrSubclass(torch.Tensor):
    x: int = 10
    size: int = 10

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        return func(*args, **kwargs)


class DummyNDim(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.Tensor.ndim.__get__:
            return 10

        return super().__torch_function__(func, types, args, kwargs)


class WrapperSubclass:
    def __init__(self, tensor):
        self.tensor = tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        args = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, args)
        kwargs = pytree.tree_map_only(WrapperSubclass, lambda x: x.tensor, kwargs)

        return func(*args, **kwargs)


class SigmoidToExpSubclass(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func == torch.Tensor.sigmoid:
            return super().__torch_function__(torch.Tensor.exp, types, args, kwargs)

        return super().__torch_function__(func, types, args, kwargs)


# Wrapper subclass with two inner tensors: data and scale
# data has same shape as outer, and scale has single dim size
class ScaledTensor(torch.Tensor):
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        *,
        constant: int = 0,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale: torch.Tensor, constant: int = 0):
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        ctx = {"_constant": self._constant}
        return ["_data", "_scale"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        assert len(inner_tensors) == 2
        return ScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            constant=metadata["_constant"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        scaled_tensor = args[0]
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        return ScaledTensor(out, scaled_tensor._scale, constant=scaled_tensor._constant)

    def __repr__(self):
        return f"{self._data.__repr__()}\n{self._scale.__repr__()}"


class OptionalScaledTensor(torch.Tensor):
    def __new__(
        cls,
        data,
        scale,
        *,
        constant: int = 0,
    ):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=data.dtype,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )

    def __init__(self, data: torch.Tensor, scale, constant: int = 0):
        self._data = data
        self._scale = scale
        self._constant = constant

    def __tensor_flatten__(self):
        ctx = {"_constant": self._constant}
        if self._scale is not None:
            return ["_data", "_scale"], ctx
        else:
            return ["_data"], ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return OptionalScaledTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"] if "_scale" in inner_tensors else None,
            constant=metadata["_constant"],
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        scaled_tensor = args[0]
        out = func(scaled_tensor._data, *args[1:], **kwargs)
        if scaled_tensor._scale is not None:
            out = out * scaled_tensor._scale
        return OptionalScaledTensor(
            out, scaled_tensor._scale, constant=scaled_tensor._constant
        )

    def __repr__(self):
        return (
            f"OptionalScaledTensor({self._data.__repr__()}\n{self._scale.__repr__()})"
        )


class CtxSubclassTensor(torch.Tensor):
    """
    Class used to verify guarding on the subclass metadata
    """

    @staticmethod
    def __new__(cls, a, constant):
        shape = a.shape
        kwargs = {}
        kwargs["strides"] = a.stride()
        kwargs["storage_offset"] = a.storage_offset()
        kwargs["device"] = a.device
        kwargs["layout"] = a.layout
        kwargs["requires_grad"] = a.requires_grad
        kwargs["dtype"] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        return out

    def __init__(self, a, constant):
        self.a = a
        self.constant = constant

    def __repr__(self):
        a_repr = repr(self.a)
        return f"CtxSubclassTensor({a_repr})"

    def __tensor_flatten__(self):
        return ["a"], (self.constant,)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, sizes, strides):
        constant = meta[0]
        a = inner_tensors["a"]
        return CtxSubclassTensor(a, constant)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        biggest_constant = max(
            [
                x.constant
                for x in pytree.tree_flatten(args)[0]
                if isinstance(x, CtxSubclassTensor)
            ]
        )
        args_a = pytree.tree_map(
            lambda x: x.a if isinstance(x, CtxSubclassTensor) else x, args
        )
        kwargs_a = pytree.tree_map(
            lambda x: x.a if isinstance(x, CtxSubclassTensor) else x, kwargs
        )
        out_a = func(*args_a, **kwargs_a)
        out = pytree.tree_map(
            lambda x: (
                CtxSubclassTensor(x, biggest_constant)
                if isinstance(x, torch.Tensor)
                else x
            ),
            out_a,
        )

        if func == torch.ops.aten.mul.Tensor:
            out = out + out.constant

        return return_and_correct_aliasing(func, args, kwargs, out)


def func(a):
    return a.sin()


class EagerRecordGraphAndInputs:
    def __init__(self) -> None:
        self.graphs = []
        self.example_inputs = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.graphs.append(gm)
        self.example_inputs.append(example_inputs)
        return gm


GLOBAL_TEST_SUBCLASSES = {
    MockSubclass,
    DummyNDim,
    SigmoidToExpSubclass,
    BaseTorchFunction,
}


# Returns True if the function recompiles between inputs1 and inputs2 with the
# specified dynamic setting.
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    compile_count = [0]

    def counter(gm, example_inputs):
        compile_count[0] += 1
        return gm

    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)
    compiled_f(*inputs2)
    return compile_count[0] > 1


class SubclassTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch(
                "traceable_tensor_subclasses", GLOBAL_TEST_SUBCLASSES
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._exit_stack.close()

    def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
        _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles)

    def test_no_call_to_new(self):
        class BadNewTorchFunction(torch.Tensor):
            def __new__(cls, *args, **kwargs):
                raise RuntimeError("Oops!")

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {BadNewTorchFunction}
        ):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                return torch.add(x, 1)

            input = torch.ones(2, 2).as_subclass(BadNewTorchFunction)

            res = fn(input)
            self.assertIsInstance(res, BadNewTorchFunction)

    def test_no_torch_function_recompiles(self):
        class NJT:
            def __repr__(self):
                return f"NJT(shape={self.shape})"

            def __init__(self, values, offsets):
                self._values = values
                self._offsets = offsets

            def sin(self):
                return torch.sin(self)

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                if func == torch.sin:
                    self = args[0]
                    return NJT(func(self._values), self._offsets)
                raise AssertionError("should not get here")

        values1 = torch.randn(10, 3, 4, requires_grad=True)
        values2 = torch.randn(10, 3, 4, requires_grad=True)
        offsets = torch.tensor([0, 3, 10])
        njt1 = NJT(values1, offsets)
        njt2 = NJT(values2, offsets)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return torch.sin(x)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            f(njt1)
            f(njt2)

    def test_base_torch_function_tracing(self):
        def fn(x):
            return torch.add(x, 1)

        input = torch.ones(2, 2).as_subclass(BaseTorchFunction)
        out = fn(input)
        out_opt = compile_full_eager(fn)(input)
        self.assertIsInstance(out, BaseTorchFunction)
        self.assertEqual(out, out_opt)

    def test_torch_function_state_graph_break(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                torch._dynamo.graph_break()
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_torch_function_state_nested(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                with torch._C.DisableTorchFunctionSubclass():
                    x = x + 1
                # Should reset to the outer state (disabled) after exiting ctx manager
                return torch._C._is_torch_function_enabled(), torch.add(x, 1.0)

        input = torch.ones(2, 2)
        res, _ = fn(input)
        self.assertFalse(res)

    def test_torch_function_state_tracing(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            with torch._C.DisableTorchFunctionSubclass():
                torch.add(x, 1.0)

        input = torch.ones(2, 2)

        res = fn(input)

    def test_torch_function_state_guards(self):
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True)
        def fn(x):
            torch.add(x, 1.0)

        input = torch.ones(2, 2)

        with torch._C.DisableTorchFunctionSubclass():
            res = fn(input)

        res = fn(input)

        self.assertEqual(cnt.frame_count, 2)

    def test_return_subclass(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return MockSubclass(torch.add(x, 1.0))

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)

    def test_return_as_subclass(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return torch.add(x, 1.0).as_subclass(MockSubclass)

        input = torch.ones(2, 2)

        res = fn(input)
        self.assertIsInstance(res, MockSubclass)

    def test_return_local_subclass(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with torch._dynamo.config.patch("traceable_tensor_subclasses", {LocalSubclass}):

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                return LocalSubclass(torch.add(x, 1.0))

            input = torch.ones(2, 2)

            res = fn(input)
            self.assertIsInstance(res, LocalSubclass)

    def test_torch_function_list_args(self):
        HANDLED_FUNCTIONS = {}

        class MyClass:
            def __init__(self, foo):
                self.foo = foo

            @classmethod
            def __torch_function__(
                cls,
                func,
                types,
                args=(),
                kwargs=None,
            ):
                if kwargs is None:
                    kwargs = {}
                if func not in HANDLED_FUNCTIONS or not all(  # noqa: C419
                    [  # noqa: C419
                        issubclass(t, (torch.Tensor, MyClass)) for t in types
                    ]
                ):
                    return NotImplemented
                return HANDLED_FUNCTIONS[func](*args, **kwargs)

        def _stack(input, dim=0, *, out=None):
            return MyClass(sum([x.foo for x in input]))

        HANDLED_FUNCTIONS[torch.stack] = _stack

        @torch.compile(backend="eager", fullgraph=True)
        def fn(v0, v1):
            return torch.stack([v0, v1])

        ret = fn(MyClass(1), MyClass(1))
        self.assertEqual(ret.foo, 2)

    @parametrize(
        "comparison",
        [
            subtest(isinstance, "isinstance"),
            subtest(lambda instance, type_: type(instance) == type_, "equality"),
            subtest(lambda instance, type_: type(instance) is type_, "identity"),
        ],
    )
    @parametrize(
        "input_type",
        [
            subtest(torch.Tensor, "tensor"),
            subtest(DummyNDim, "subclass"),
        ],
    )
    def test_type_check(self, comparison, input_type):
        with torch._dynamo.config.patch("traceable_tensor_subclasses", {DummyNDim}):

            def fn(x):
                if comparison(x, DummyNDim):
                    return torch.ones(1, 1)
                else:
                    return torch.zeros(2, 2)

            input = torch.ones(2, 2).as_subclass(input_type)
            exp_res = fn(input)
            act_res = torch.compile(backend="eager", fullgraph=True)(fn)(input)
            self.assertEqual(exp_res, act_res)

    def test_torch_function_call_on_method(self):
        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = torch.ones(2, 2)
        wrapped = x.as_subclass(SigmoidToExpSubclass)
        wrapped2 = y.as_subclass(SigmoidToExpSubclass)

        def fn(w):
            return w.exp()

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped2)
        res_exp2 = z.exp()

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, res_exp2)

    def test_torch_function_call_on_method_arg(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if func == torch._C.TensorBase.add_:
                    func = torch._C.TensorBase.sub_

                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            def sigmoid(self):
                return None

        x = torch.ones(2, 2)
        y = torch.ones(2, 2)
        z = torch.ones(2, 2)
        wrapped = y.as_subclass(LocalSubclass)
        wrapped2 = z.as_subclass(LocalSubclass)

        def fn(a, w):
            a.add_(w)
            return a

        fn_opt = torch.compile(fn)

        with torch._dynamo.config.patch("traceable_tensor_subclasses", {LocalSubclass}):
            res_exp = fn(x, wrapped)
            res_act = fn_opt(y, wrapped2)

        self.assertEqual(res_exp, res_act)

    def test_user_overidden_method_unsupported(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            def sigmoid(self):
                return None

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            x.sigmoid()

        msg = (
            "Accessing overridden method/attribute sigmoid on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_user_overidden_attr_unsupported(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            ndim = 10

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.ndim

        msg = (
            "Accessing overridden method/attribute ndim on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_user_overidden_property_unsupported(self):
        class LocalSubclass(torch.Tensor):
            def __init__(self) -> None:
                self._ndim = 10

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

            @property
            def ndim(self):
                return self._ndim

            @ndim.setter
            def ndim(self, value):
                self._ndim = value

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.ndim

        msg = (
            "Accessing overridden method/attribute ndim on a tensor"
            " subclass with a __torch_function__ override is not supported"
        )
        with torch._dynamo.config.patch(
            "traceable_tensor_subclasses", {LocalSubclass}
        ), self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

    def test_overridden_method_guarding(self):
        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager")
        def fn(x):
            return x.sigmoid()

        with torch._dynamo.config.patch(
            error_on_recompile=True, traceable_tensor_subclasses={LocalSubclass}
        ):
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)
            fn(x)
            x = torch.ones(2, 2).as_subclass(LocalSubclass)
            fn(x)

        with torch._dynamo.config.patch(
            traceable_tensor_subclasses={LocalSubclass}
        ), self.assertRaisesRegex(
            TypeError,
            "'bool' object is not callable",
        ):
            LocalSubclass.sigmoid = False
            fn(x)

    def test_torch_function_call_on_attr(self):
        x = torch.ones(2, 2)
        wrapped = x.as_subclass(DummyNDim)

        def fn(w):
            return w.ndim + torch.ones(2)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)

        self.assertEqual(res_exp, res_act)
        self.assertEqual(res_exp, torch.ones(2) + 10)

    def test_torch_function_wrapper_class(self):
        x = torch.ones(2, 2)
        wrapped = WrapperSubclass(x)

        def fn(w):
            return torch.add(w, 1.0)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)
        self.assertEqual(res_exp, res_act)

    def test_no_torch_function_on_size_bytecode(self):
        class TestTensor(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                with torch._C.DisableTorchFunctionSubclass():
                    out = func(*args, **kwargs)

                    if func == torch.clone:
                        return out * 2
                    else:
                        return out

        def fn(x):
            return torch.clone(x)

        with torch._dynamo.config.patch(traceable_tensor_subclasses={TestTensor}):
            inp = torch.ones(4, 4)
            x = inp.as_subclass(TestTensor)
            torch._dynamo.mark_dynamic(x, 0)
            compiled_fn = torch.compile(fn, fullgraph=True)
            out = compiled_fn(x)
            self.assertEqual(out, torch.ones(4, 4) * 2)

    def test_torch_function_wrapper_class_with_kwargs(self):
        x = torch.ones(2, 2)
        wrapped = WrapperSubclass(x)

        def fn(w):
            return torch.add(w, 1.0, alpha=2.0)

        fn_opt = compile_full_eager(fn)

        res_exp = fn(wrapped)
        res_act = fn_opt(wrapped)
        self.assertEqual(res_exp, res_act)

    def test_tensor_subclass_custom_attr(self):
        class AttrSubclass(torch.Tensor):
            x: int = 10

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                return super().__torch_function__(func, types, args, kwargs)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            return x.x + torch.ones(2, 2)

        with traceable_subclass(AttrSubclass):
            input = torch.ones(2, 2).as_subclass(AttrSubclass)
            fn_opt = compile_full_eager(fn)

            res_exp = fn(input)
            res_act = fn_opt(input)
            self.assertEqual(res_exp, res_act)

    def test_compile_with_fake_tensor_dynamic_dim(self):
        x = torch.randn([3, 4])

        def f(x):
            return torch.sin(x)

        def test_dynamic_dim(f, x, dim_dynamic, exp_frame_count, exp_op_count):
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()

            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            x1 = torch.rand_like(x)
            f(x)
            f(torch.randn([4, 3]))
            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                x_fake = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                x1_fake = fake_mode.from_tensor(
                    x1,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[dim_dynamic for i in range(x.dim())]
                    ),
                )
                opt_f(x_fake)
                opt_f(x1_fake)

            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        test_dynamic_dim(f, x, DimDynamic.DYNAMIC, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.DUCK, 1, 1)
        test_dynamic_dim(f, x, DimDynamic.STATIC, 1, 1)

    def test_compile_with_fake_tensor_automatic_dynamic(self):
        def f(x):
            return torch.sin(x)

        def test_automatic_dynamic(f, inps, dim_dynamic, exp_frame_count, exp_op_count):
            torch._dynamo.reset()
            cnt = torch._dynamo.testing.CompileCounter()
            opt_f = torch.compile(f, backend=cnt, fullgraph=True)

            shape_env = ShapeEnv()
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                for inp in inps:
                    fake_inp = fake_mode.from_tensor(
                        inp,
                        symbolic_context=StatelessSymbolicContext(
                            [dim_dynamic for i in range(x.dim())]
                        ),
                    )
                    opt_f(fake_inp)
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)

        x = torch.randn([3, 4])
        y = torch.randn([4, 5])
        z = torch.randn([5, 6])
        a = torch.randn([3, 5])
        b = torch.randn([4, 4])
        # When inputs' DimDynamic is DYNAMIC or DUCK, the inputs
        # to opt_f will be tensors with SymInt sizes. Dynamo will treat input
        # as dynamic automatically and will only compile once
        for dim_dynamic in [DimDynamic.DYNAMIC, DimDynamic.DUCK]:
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 1, 1)
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 1, 1)

        for dim_dynamic in [DimDynamic.STATIC]:
            # Recompile once, first with dim 0 and 1 become Dynamic
            test_automatic_dynamic(f, [x, y, z], dim_dynamic, 2, 2)
            # Recompile 2 times, first with dim 1 become Dynamic, second with dim 0 becomes Dynamic.
            test_automatic_dynamic(f, [x, a, z], dim_dynamic, 3, 3)
            # Recompile 2 times, first with dim 0 become Dynamic, second with dim 1 becomes Dynamic.
            test_automatic_dynamic(f, [x, b, z], dim_dynamic, 3, 3)

    def test_compile_with_functionalization(self):
        x = torch.randn([3, 4])
        x_clone = x.clone()
        x_clone2 = x.clone()
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return x.add_(1.0) + torch.nn.functional.relu_(x)

        f_out = f(x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 3)
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.example_inputs), 1)

        actual = normalize_gm(backend.graphs[0].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )

        ff = torch.func.functionalize(f)
        ff_out = ff(x_clone)

        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 6)
        self.assertEqual(len(backend.graphs), 2)
        self.assertEqual(len(backend.example_inputs), 2)
        actual = normalize_gm(backend.graphs[1].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        # Cannot re-use the version from AOTAutograd, since that uses python functional tensors.
        def to_fun(x):
            x_functional = torch._to_functional_tensor(x)
            torch._mirror_autograd_meta_to(x, x_functional)
            return x_functional

        def aot_f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(to_fun, args)
                    func_kwargs = pytree.tree_map(to_fun, kwargs)
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        aot_ff = aot_f_wrapper(f)
        aot_ff_out = aot_ff(x_clone2)

        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 9)
        self.assertEqual(len(backend.graphs), 3)
        self.assertEqual(len(backend.example_inputs), 3)
        actual = normalize_gm(backend.graphs[2].print_readable(print_output=False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        add_: "f32[3, 4]" = l_x_.add_(1.0)
        relu_: "f32[3, 4]" = torch.relu_(l_x_);  l_x_ = None
        add: "f32[3, 4]" = add_ + relu_;  add_ = relu_ = None
        return (add,)
""",
        )
        self.assertTrue(torch._is_functional_tensor(backend.example_inputs[1][0]))

        self.assertEqual(f_out, ff_out)
        self.assertEqual(f_out, aot_ff_out)

        try:
            torch._enable_functionalization(reapply_views=False)
            xf = pytree.tree_map(to_fun, x)
            x_view = xf.t()
            with self.assertRaisesRegex(RuntimeError, "Cannot safely fakify a view"):
                f(x_view)
        finally:
            torch._disable_functionalization()

    def test_compile_higher_order_with_functionalization(self):
        backend = EagerRecordGraphAndInputs()
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            return wrap(lambda x: x.add_(1.0), x)

        def check_count_and_graph(
            exp_frame_count, exp_op_count, exp_n_graph, exp_graph
        ):
            self.assertEqual(cnt.frame_count, exp_frame_count)
            self.assertEqual(cnt.op_count, exp_op_count)
            self.assertEqual(len(backend.graphs), exp_n_graph)
            actual = normalize_gm(
                backend.graphs[exp_n_graph - 1].print_readable(print_output=False)
            )
            self.assertExpectedInline(actual, exp_graph, skip=1)

        t = torch.randn([3, 4])
        t_clone = t.clone()
        t_clone2 = t.clone()
        f(t)

        check_count_and_graph(
            1,
            2,
            1,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

        ff = torch.func.functionalize(f)
        ff_out = ff(t_clone)
        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(
            2,
            4,
            2,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

        try:
            x = torch._to_functional_tensor(t_clone2)
            torch._mirror_autograd_meta_to(t_clone2, x)
            torch._enable_functionalization(reapply_views=False)
            aot_f_out = f(x)
        finally:
            torch._disable_functionalization()

        # frame count and op count are incremented due to re-compilation
        check_count_and_graph(
            3,
            6,
            3,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        wrap = torch.ops.higher_order.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None
        getitem: "f32[3, 4]" = wrap[0];  wrap = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[3, 4]"):
            add_: "f32[3, 4]" = l_x_.add_(1.0);  l_x_ = None
            return (add_,)
""",
        )

    def test_has_torch_function(self):
        class MyTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                if func is torch.max:
                    return torch.tensor(123)
                return func(*args, **kwargs)

        class LocalSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        def fn(x):
            return torch.overrides.has_torch_function_unary(
                x
            ), torch.overrides.has_torch_function_variadic(x)

        for test_class in [MyTensor, LocalSubclass]:
            x = test_class()
            ref0 = fn(x)
            ref1 = fn(4)
            opt_fn = torch._dynamo.optimize("eager")(fn)
            res0 = opt_fn(x)
            res1 = opt_fn(4)
            self.assertEqual(ref0, res0)
            self.assertEqual(ref1, res1)

    def test_wrapper_subclass_guards_on_inner_tensor(self):
        # Holds an inner tensor, that has a distinct shape from the outer wrapper tensor.
        # Also adds additional guards on the inner tensor's sizes.
        # When the first input to an op has x.shape[0] > 5, we insert an extra add node.
        class DoubleSizeMaybeAddGeThreeTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, inner):
                # Double the outer-most dimension
                outer_shape = (inner.shape[0] * 2,) + inner.shape[1:]
                return torch.Tensor._make_wrapper_subclass(
                    # TODO: right now, _make_wrapper_subclass's dynamic shape interaction is not great.
                    # Calling the overload that has kwargs causes us to go down the first overload path,
                    # which will **always** specialize sizes.
                    # We should probably eventually fix this so that the first overload can just handle dynamic shapes.
                    cls,
                    outer_shape,
                    inner.stride(),
                    None,
                    None,
                    inner.dtype,
                    inner.layout,
                    inner.device,
                    False,
                    inner.requires_grad,
                )

            def __init__(self, inner):
                self.inner_elem = inner

            def __tensor_flatten__(self):
                return ["inner_elem"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, _, outer_size, outer_stride):
                return DoubleSizeMaybeAddGeThreeTensor(inner_tensors["inner_elem"])

            def __repr__(self):
                return f"DoubleSizeMayberAddGeThreeTensor({repr(self.inner_elem)})"

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                args_inner = torch.utils._pytree.tree_map_only(
                    DoubleSizeMaybeAddGeThreeTensor, lambda x: x.inner_elem, args
                )
                out_inner = func(*args_inner, **kwargs)

                # Add guards on the  inner tensor's sizes
                if args_inner[0].shape[0] > 3:
                    out_inner += 2

                return DoubleSizeMaybeAddGeThreeTensor(out_inner)

        curr_var_to_val = None
        curr_var_to_sources = None
        guards = None

        def backend(gm, args):
            context = torch._guards.TracingContext.get()

            # Grab info on sources and guards from the shapeenv
            nonlocal curr_var_to_val
            nonlocal curr_var_to_sources
            nonlocal guards

            guards = [str(g.expr) for g in context.fake_mode.shape_env.guards]
            curr_var_to_val = {
                str(k): v for k, v in context.fake_mode.shape_env.var_to_val.items()
            }
            curr_var_to_sources = {
                str(k): v[0].name()
                for k, v in context.fake_mode.shape_env.var_to_sources.items()
            }
            return gm

        @torch.compile(backend=backend)
        def fn(x):
            if x.shape[0] < 13:
                return torch.mul(x, x)
            else:
                return torch.div(x, x)

        inp = torch.ones(4, 4)

        x = DoubleSizeMaybeAddGeThreeTensor(inp)
        torch._dynamo.mark_dynamic(x, 0)
        res = fn(x)
        # During fakeifying, we end up allocating a separate symint
        # for the outer and inner tensor (in this test, s0 is unused).
        expected_var_to_val = {
            "s0": 8,
            "s1": 4,
        }
        expected_var_to_sources = {
            "s0": "L['x'].size()[0]",
            "s1": "L['x'].inner_elem.size()[0]",
        }
        self.assertEqual(curr_var_to_val, expected_var_to_val)
        self.assertEqual(curr_var_to_sources, expected_var_to_sources)
        self.assertExpectedInline(
            "\n".join(guards),
            """\
Eq(2*s1, s0)
2*s1 < 13
s1 > 3""",
        )

    def test_wrapper_subclass_with_same_sized_inner_tensor(self):
        # shouldn't recompile for different sizes when dynamic=True
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(7))
        self.assertFalse(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=True))

        # should recompile for different data size when dynamic=False
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(6))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

        # avoid recompile using manual mark_dynamic() for different data size
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(6))
        # NB: mark_dynamic() on outer tensor should translate to inner tensors of the same size
        torch._dynamo.mark_dynamic(sub1, 0)
        torch._dynamo.mark_dynamic(sub1, 1)
        sub2 = ScaledTensor(torch.randn(3, 5), torch.randn(6))
        self.assertFalse(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

    def test_wrapper_subclass_with_differently_sized_inner_tensor(self):
        # should recompile for different scale size when dynamic=False
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(3))
        sub2 = ScaledTensor(torch.randn(2, 4), torch.randn(5))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

        # still recompiles using manual mark_dynamic() on outer for different scale size
        sub1 = ScaledTensor(torch.randn(2, 4), torch.randn(3))
        # NB: mark_dynamic() on outer tensor doesn't translate to inner tensors of different size
        torch._dynamo.mark_dynamic(sub1, 0)
        torch._dynamo.mark_dynamic(sub1, 1)
        sub2 = ScaledTensor(torch.randn(2, 4), torch.randn(5))
        self.assertTrue(_recompiles_for_inputs(func, (sub1,), (sub2,), dynamic=False))

    def test_recompiles_with_optional_inner_tensor(self):
        def f(x):
            return x + 1

        # sub1 does not have the optional tensor specified while sub2 does
        sub1 = OptionalScaledTensor(torch.randn(2, 4), None)
        sub2 = OptionalScaledTensor(torch.randn(2, 4), torch.randn(2, 4))

        # sanity check; don't recompile for same input
        self.assertFalse(_recompiles_for_inputs(f, (sub1,), (sub1,), dynamic=True))
        self.assertFalse(_recompiles_for_inputs(f, (sub2,), (sub2,), dynamic=True))

        # these should recompile; optional tensor changes between specified and unspecified
        self.assertTrue(_recompiles_for_inputs(f, (sub1,), (sub2,), dynamic=True))
        self.assertTrue(_recompiles_for_inputs(f, (sub2,), (sub1,), dynamic=True))

        f_compiled = torch.compile(f, backend="aot_eager")
        self.assertEqual(f(sub1)._data, f_compiled(sub1)._data)
        self.assertEqual(f(sub2)._data, f_compiled(sub2)._data)

    def test_torch_dispatch_subclass_guard_recompile(self):
        x = torch.ones(2, 2)
        x_two = TwoTensor(x.clone(), x.clone())

        def fn(w):
            return torch.add(w, 1.0)

        fn_opt = torch.compile(backend="eager")(fn)

        ref = fn(x_two)
        res = fn_opt(x_two)
        self.assertEqual(ref, res)

        # ensure no recompilation on same input type
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn_opt(TwoTensor(x + 1, x + 2))

        # recompile!
        ref = fn(x)
        res = fn_opt(x)
        self.assertEqual(ref, res)

    def test_tensor_subclass_ctx_guards(self):
        x = CtxSubclassTensor(torch.ones(2), 3)
        x2 = CtxSubclassTensor(torch.ones(2), 3)
        x3 = CtxSubclassTensor(torch.ones(2), 4)
        _check_recompiles(self, lambda x: x * x, (x,), (x2,), False)
        _check_recompiles(self, lambda x: x * x, (x,), (x3,), True)

    def test_tensor_subclass_ctx_recursive_guards(self):
        x0 = torch.ones(2, 2)
        x1 = CtxSubclassTensor(x0.clone(), 2)
        x2 = CtxSubclassTensor(x0.clone(), 3)
        tt0 = TwoTensor(x0.clone(), x1)
        tt1 = TwoTensor(x0.clone(), x2)

        _check_recompiles(self, lambda x: x * x, (tt0,), (tt1,), True)

    def test_tensor_subclass_ctx_custom_guards_override(self):
        class CtxSubclassTensorCustomGuardFn(CtxSubclassTensor):
            @classmethod
            def __metadata_guard__(cls, orig_data, other):
                return orig_data[0] <= other[0]

        x = CtxSubclassTensorCustomGuardFn(torch.ones(2), 2)
        x2 = CtxSubclassTensorCustomGuardFn(torch.ones(2), 3)
        x3 = CtxSubclassTensorCustomGuardFn(torch.ones(2), 1)
        _check_recompiles(self, lambda x: x * x, (x,), (x2,), False)
        _check_recompiles(self, lambda x: x * x, (x,), (x3,), True)

    def test_tensor_subclass_ctx_custom_guards_error_arg_num(self):
        import torch._dynamo.exc

        class CtxSubclassTensorCustomGuardFn(CtxSubclassTensor):
            @classmethod
            def __metadata_guard__(cls, y):
                # Shouldn't reach here
                return False

        x = CtxSubclassTensorCustomGuardFn(torch.ones(2), 3)
        self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "Tensor subclass method __metadata_guard__ must take exactly two subclass metadata arguments",
            lambda: torch.compile(lambda x: x * x)(x),
        )

    def test_tensor_subclass_ctx_custom_guards_error_not_classmethod(self):
        import torch._dynamo.exc

        class CtxSubclassTensorCustomGuardFn(CtxSubclassTensor):
            def __metadata_guard__(self, x, y):
                return False

        x = CtxSubclassTensorCustomGuardFn(torch.ones(2), 3)
        self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "Tensor subclass method __metadata_guard__ must be a classmethod",
            lambda: torch.compile(lambda x: x * x)(x),
        )

    def test_subclass_constructor_proxying(self):
        import dataclasses
        from collections import namedtuple
        from typing import Any

        @dataclasses.dataclass(frozen=True)
        class SubclassTensorArgs:
            original_shape: torch.Size
            device: torch.device
            inner_meta: Any

        SubclassTensorArgs2 = namedtuple(
            "SubclassTensorArgs2",
            [
                "original_shape",
                "device",
                "inner_meta",
            ],
        )

        class SubclassTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, a, meta):
                shape = a.shape
                kwargs = {}
                kwargs["strides"] = a.stride()
                kwargs["storage_offset"] = a.storage_offset()
                kwargs["device"] = a.device
                kwargs["layout"] = a.layout
                kwargs["requires_grad"] = a.requires_grad
                kwargs["dtype"] = a.dtype
                out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
                return out

            def __init__(self, a, meta):
                self.a = a
                self.meta = meta

            def __repr__(self):
                a_repr = repr(self.a)
                return f"SubclassTensor({a_repr})"

            def __tensor_flatten__(self):
                return ["a"], self.meta

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, _, __):
                a = inner_tensors["a"]
                return SubclassTensor(a, meta)

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs):
                if kwargs is None:
                    kwargs = {}
                args_a = pytree.tree_map(
                    lambda x: x.a if isinstance(x, SubclassTensor) else x, args
                )
                kwargs_a = pytree.tree_map(
                    lambda x: x.a if isinstance(x, SubclassTensor) else x, kwargs
                )
                out_a = func(*args_a, **kwargs_a)
                out = pytree.tree_map(
                    lambda x: (
                        SubclassTensor(x, SubclassTensorArgs2(x.shape, x.device, None))
                        if isinstance(x, torch.Tensor)
                        else x
                    ),
                    out_a,
                )
                return return_and_correct_aliasing(func, args, kwargs, out)

        @torch.compile(fullgraph=True)
        def f1(x):
            meta = SubclassTensorArgs(
                x.shape, x.device, SubclassTensorArgs(x.shape, x.device, None)
            )
            out = SubclassTensor(x, meta)
            return out * out

        x = torch.randn(3, 3)
        f1(x)

        @torch.compile(fullgraph=True)
        def f1(x):
            meta = SubclassTensorArgs2(
                x.shape, x.device, SubclassTensorArgs2(x.shape, x.device, None)
            )
            out = SubclassTensor(x, meta)
            return out * out

        x = torch.randn(3, 3)
        f1(x)

    def test_torch_function_subclass_survives_into_aot_autograd(self):
        # If you have a tensor subclass that relies on dispatch into the same op
        # without unwrapping and calling torch._C.DisableTorchFunctionSubclass(),
        # the torch function-ness will survive into AOTAutograd. Today, NestedTensor
        # actually relies on this behavior! Because that torch function logic
        # runs during AOTAutograd, this test tests that there is no logic below
        # that relies torch function that gets unexpectedly disabled after we
        # redispatch from the subclass's torch function.
        class SubTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, t):
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    t.shape,
                    t.stride(),
                    t.storage_offset(),
                    torch.contiguous_format,
                    t.dtype,
                    torch.strided,
                    t.device,
                    False,
                    t.requires_grad,
                    "sizes",
                    False,
                    False,
                    None,
                )

            def __init__(self, t):
                super().__init__()
                self._t = t

            def __tensor_flatten__(self):
                return ["_t"], {}

            @staticmethod
            def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
                t = inner_tensors["_t"]
                return SubTensor(t)

            def __repr__(self):
                return f"SubTensor({self._t})"

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                with torch._C.DisableTorchFunctionSubclass():
                    return func(*args, **kwargs)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                new_args = pytree.tree_map_only(SubTensor, lambda s: s._t, args)
                output = func(*new_args, **kwargs)
                output = pytree.tree_map_only(
                    torch.Tensor, lambda t: SubTensor(t), output
                )
                return output

        @torch.compile(dynamic=True)
        def f(x):
            return x.unflatten(-1, [2, 5])

        s = SubTensor(torch.randn(3, 10))
        f(s)

    # Guard validation upsets the guard
    # https://github.com/pytorch/pytorch/issues/129936
    @unittest.expectedFailure
    def test_recompile_with_symbool_inputs(self):
        def f(pred: bool):
            if pred:
                return torch.ones([3, 4])
            else:
                return torch.ones([4, 3])

        def test_recompilation(
            f, x, sizes, exp_graphs, exp_frame_count, exp_shape_env_guards
        ):
            torch._dynamo.reset()
            shape_env = ShapeEnv()
            backend = torch._dynamo.testing.EagerAndRecordGraphs()
            cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)
            f_cond = torch.compile(f, backend=cnt, fullgraph=True)
            with torch._subclasses.fake_tensor.FakeTensorMode(
                shape_env=shape_env
            ) as fake_mode:
                fake_inp = fake_mode.from_tensor(
                    x,
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[DimDynamic.DYNAMIC for i in range(x.dim())]
                    ),
                )
                for i, size in enumerate(sizes):
                    pred = fake_inp.size(0) == size
                    f_cond(pred)
                    actual = normalize_gm(
                        backend.graphs[exp_frame_count[i] - 1].print_readable(
                            print_output=False
                        )
                    )
                    actual_guard_str = [str(guard.expr) for guard in shape_env.guards]
                    self.assertExpectedInline(actual, exp_graphs[i])
                    self.assertEqual(cnt.frame_count, exp_frame_count[i])
                    self.assertEqual(actual_guard_str, exp_shape_env_guards[i])

        true_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        ones: "f32[3, 4]" = torch.ones([3, 4])
        return (ones,)
"""
        false_graph = """\
class GraphModule(torch.nn.Module):
    def forward(self):
        ones: "f32[4, 3]" = torch.ones([4, 3])
        return (ones,)
"""
        test_recompilation(
            f,
            torch.randn([3, 4]),
            [3, 3, 4, 5],
            exp_graphs=[true_graph, true_graph, false_graph, false_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 is specialized and guarded in outter shape_env when dynamo checks the guards
                ["Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)"],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                ],
                [
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 4)), (0, True)), 1)",
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                ],
            ],
        )

        test_recompilation(
            f,
            torch.randn([3, 4]),
            [4, 5, 3, 3],
            exp_graphs=[false_graph, false_graph, true_graph, true_graph],
            exp_frame_count=[1, 1, 2, 2],
            exp_shape_env_guards=[
                [],
                # s0 is specialized and guarded in outter shape_env when dynamo checks the guards
                ["Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)"],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
                [
                    "Ne(Piecewise((1, Eq(s0, 5)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                    "Eq(Piecewise((1, Eq(s0, 3)), (0, True)), 1)",
                ],
            ],
        )

    def test_wrapper_subclass_dynamo_attribute_access_on_intermediate(self):
        def f(x_subclass):
            tmp_subclass = torch.add(x, 1)
            return torch.mul(tmp_subclass._scale, tmp_subclass._constant)

        x = ScaledTensor(torch.randn(2, 4), torch.randn(3), constant=2)
        out_ref = f(x)
        out_test = torch.compile(f, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(out_ref, out_test)

    def test_support_bases(self):
        import abc

        import torch.fx._symbolic_trace

        class Meta(abc.ABCMeta, torch.fx._symbolic_trace.ProxyableClassMeta):
            def __new__(cls, name, bases, dct):
                x = super().__new__(cls, name, bases, dct)
                x.attr = 100
                return x

        class Multistreamable(abc.ABC):  # noqa: B024
            pass

        class Foo(Multistreamable, metaclass=Meta):
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            typ = type(Foo())
            typ.__bases__
            return typ.__bases__

        self.assertEqual(f(torch.randn(1)), (Multistreamable,))

        @torch.compile(backend="eager", fullgraph=True)
        def g(x):
            typ = type(Foo())
            typ.__base__
            return typ.__base__

        self.assertEqual(g(torch.randn(1)), Multistreamable)

    @parametrize("dynamic", [False, True])
    def test_subclass_views(self, dynamic):
        def _get_views(t):  # returns (view: Tensor, expects_raises_false)
            # Note that any closed-over SymInts will be symbolicized during fake-ification.
            yield t.narrow(dim=-1, start=3, length=8), False
            yield t.split(5, -1)[2], False
            yield t.split_with_sizes([9, 6], -1)[1], False
            yield t.unsqueeze(-1).expand(4, 15, 10), False
            yield t.select(-1, 6), False
            # https://github.com/pytorch/pytorch/issues/128649
            yield t[2:3, 5:9], dynamic
            yield t.view(-1, 15), False

        def f(x):
            return x * 2

        compiled_f = torch.compile(
            f, backend="aot_eager", fullgraph=True, dynamic=dynamic
        )

        # Take a view of a subclass to pass as input.
        t = TwoTensor(torch.randn(4, 15), torch.randn(4, 15))
        for view, expects_raises in _get_views(t):
            torch._dynamo.reset()
            out_ref = f(view)
            if expects_raises:
                with self.assertRaises(AssertionError):
                    out_test = compiled_f(view)
            else:
                out_test = compiled_f(view)
                self.assertEqual(out_ref, out_test)

    @torch._dynamo.config.patch("inline_inbuilt_nn_modules", True)
    @parametrize("dynamic", [True, False])
    def test_mark_static_with_subclass_desugaring(self, dynamic):
        from typing import Any, Callable, Dict, List, Optional

        from torch._dynamo.decorators import mark_static_address
        from torch._inductor.compile_fx import compile_fx
        from torch._inductor.cudagraph_utils import BoxedDeviceIndex
        from torch._inductor.utils import BoxedBool

        x_inner = torch.ones(4)
        x = TwoTensor(x_inner, x_inner)
        mark_static_address(x, guard=False)

        def inner_compile(
            gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor],
            cudagraphs: Optional[BoxedBool] = None,
            static_input_idxs: Optional[List[int]] = None,
            is_backward: bool = False,
            graph_id: Optional[int] = None,
            cpp_wrapper: bool = False,
            aot_mode: bool = False,
            is_inference: bool = False,
            boxed_forward_device_index: Optional[BoxedDeviceIndex] = None,
            user_visible_outputs: Optional[Dict[str, None]] = None,
            layout_opt: Optional[bool] = None,
            extern_node_serializer: Optional[Callable[[List[Any]], Any]] = None,
        ):
            if dynamic:
                self.assertEqual(static_input_idxs, [2, 3, 4])
            else:
                self.assertEqual(static_input_idxs, [1, 2])
            return gm

        compiler = functools.partial(compile_fx, inner_compile=inner_compile)

        @torch.compile(backend=compiler, dynamic=dynamic)
        def fn(t0, t1, t2):
            return t0 + t1 + t2 + 2

        fn(torch.ones(4), x, torch.ones(4))

    # copied from common_utils.py::NestedTensorTestCase
    def assertEqualIgnoringNestedInts(self, a, b):
        # unbinding NJTs allows us to compare them as essentially equal without
        # caring about exact nested int comparison
        def _unbind_njts(x):
            if isinstance(x, torch.Tensor) and x.is_nested and x.layout == torch.jagged:
                return x.unbind()
            else:
                return x

        self.assertEqual(
            pytree.tree_map(_unbind_njts, a), pytree.tree_map(_unbind_njts, b)
        )

    def _compile_check(
        self,
        fn,
        inps,
        *,
        dynamic=True,
        fullgraph=True,
        call_backward=False,
    ):
        def call_backward_fn(t):
            if t.is_nested:
                from torch.nested._internal.nested_tensor import buffer_from_jagged

                t = buffer_from_jagged(t)
            return t.sum().backward(retain_graph=True)

        torch.manual_seed(0)
        fw_compiler = EagerRecordGraphAndInputs()
        bw_compiler = EagerRecordGraphAndInputs()
        compiler_fn = aot_autograd(
            fw_compiler=make_boxed_compiler(fw_compiler),
            bw_compiler=make_boxed_compiler(bw_compiler),
            partition_fn=min_cut_rematerialization_partition,
            keep_inference_input_mutations=True,
        )

        c = torch.compile(backend=compiler_fn, dynamic=dynamic, fullgraph=fullgraph)(fn)
        for inp in inps:
            expected = fn(*inp)
            # reset the seed for randn to generate the same tensor
            torch.manual_seed(0)
            got = c(*inp)
            self.assertEqualIgnoringNestedInts(expected, got)

            if call_backward:
                re = pytree.tree_map_only(
                    lambda x: isinstance(x, torch.Tensor) and x.requires_grad,
                    call_backward_fn,
                    expected,
                )
                rg = pytree.tree_map_only(
                    lambda x: isinstance(x, torch.Tensor) and x.requires_grad,
                    call_backward_fn,
                    got,
                )
                self.assertEqualIgnoringNestedInts(re, rg)

        if call_backward:
            return fw_compiler.graphs, bw_compiler.graphs
        return fw_compiler.graphs, None

    def test_tensor_subclass_TwoTensor_simple(self):
        def f(tt):
            return tt * tt.size()[0]

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone().detach().requires_grad_(True)
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[s0, s1]", primals_2: "f32[s0, s1]", primals_3: "Sym(s0)", primals_4: "Sym(s1)", primals_5: "Sym(s1)", primals_6: "Sym(s0)", primals_7: "Sym(s1)"):
        mul: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(primals_1, primals_3);  primals_1 = None
        mul_3: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(primals_2, primals_3);  primals_2 = None
        return (mul, mul_3, primals_6, primals_7, primals_7, primals_3, primals_6, primals_7)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "Sym(s0)", primals_6: "Sym(s0)", primals_7: "Sym(s1)", tangents_1: "f32[s0, s1]", tangents_2: "f32[s0, s1]"):
        mul_8: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(tangents_1, primals_3);  tangents_1 = None
        mul_9: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(tangents_2, primals_3);  tangents_2 = primals_3 = None
        return (mul_8, mul_9, primals_6, primals_7, primals_7, None, None)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_clone_view(self):
        def f(tt):
            y = tt.clone()
            return y.view(y.shape[1], y.shape[0])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[s0, s1]", primals_2: "f32[s0, s1]", primals_3: "Sym(s0)", primals_4: "Sym(s1)", primals_5: "Sym(s1)", primals_6: "Sym(s0)", primals_7: "Sym(s1)"):
        clone: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[s1, s0]" = torch.ops.aten.view.default(clone, [primals_4, primals_3]);  clone = None
        view_1: "f32[s1, s0]" = torch.ops.aten.view.default(clone_1, [primals_4, primals_3]);  clone_1 = primals_3 = None
        return (view, view_1, primals_4, primals_6, primals_6, primals_6, primals_7)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "Sym(s0)", primals_7: "Sym(s1)", tangents_1: "f32[s1, s0]", tangents_2: "f32[s1, s0]"):
        view_2: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_1, [primals_6, primals_7]);  tangents_1 = None
        view_3: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_2, [primals_6, primals_7]);  tangents_2 = None
        return (view_2, view_3, primals_6, primals_7, primals_7, None, None)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_mul(self):
        def f(tt, a, b):
            s0, s1 = a.size()
            s2, s3 = b.size()
            # return tt * a.size()[1]
            return tt * s0 * s1 * s2 * s3

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt, a, b)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s0)", primals_2: "Sym(s1)", primals_3: "f32[s0, s1]", primals_4: "f32[s0, s1]", primals_5: "Sym(s0)", primals_6: "Sym(s1)", primals_7: "Sym(s1)"):
        mul: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(primals_3, primals_1);  primals_3 = None
        mul_3: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(primals_4, primals_1);  primals_4 = None
        mul_8: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul, primals_2);  mul = None
        mul_11: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_3, primals_2);  mul_3 = None
        mul_16: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_8, primals_1);  mul_8 = None
        mul_19: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_11, primals_1);  mul_11 = None
        mul_24: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_16, primals_2);  mul_16 = None
        mul_27: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_19, primals_2);  mul_19 = None
        return (mul_24, mul_27, primals_5, primals_7, primals_7, primals_1, primals_2, primals_5, primals_7)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s0)", primals_2: "Sym(s1)", primals_5: "Sym(s0)", primals_7: "Sym(s1)", tangents_1: "f32[s0, s1]", tangents_2: "f32[s0, s1]"):
        mul_32: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  tangents_1 = None
        mul_33: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(tangents_2, primals_2);  tangents_2 = None
        mul_34: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_32, primals_1);  mul_32 = None
        mul_35: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_33, primals_1);  mul_33 = None
        mul_36: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_34, primals_2);  mul_34 = None
        mul_37: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_35, primals_2);  mul_35 = primals_2 = None
        mul_38: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_36, primals_1);  mul_36 = None
        mul_39: "f32[s0, s1]" = torch.ops.aten.mul.Tensor(mul_37, primals_1);  mul_37 = primals_1 = None
        return (None, None, mul_38, mul_39, primals_5, primals_7, primals_7)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_view(self):
        def f(tt):
            y = tt.clone()
            return y.view(y.shape[0], y.shape[1])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[s0, s1]", primals_2: "f32[s0, s1]", primals_3: "Sym(s0)", primals_4: "Sym(s1)", primals_5: "Sym(s1)", primals_6: "Sym(s0)", primals_7: "Sym(s1)"):
        clone: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[s0, s1]" = torch.ops.aten.view.default(clone, [primals_3, primals_4]);  clone = None
        view_1: "f32[s0, s1]" = torch.ops.aten.view.default(clone_1, [primals_3, primals_4]);  clone_1 = primals_3 = primals_4 = None
        return (view, view_1, primals_6, primals_7, primals_7, primals_6, primals_7)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "Sym(s0)", primals_7: "Sym(s1)", tangents_1: "f32[s0, s1]", tangents_2: "f32[s0, s1]"):
        view_2: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_1, [primals_6, primals_7]);  tangents_1 = None
        view_3: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_2, [primals_6, primals_7]);  tangents_2 = None
        return (view_2, view_3, primals_6, primals_7, primals_7, None, None)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_view_mul(self):
        def f(tt):
            y = tt.clone()
            return y.view(y.shape[0] * y.shape[1])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[s0, s1]", primals_2: "f32[s0, s1]", primals_3: "Sym(s0)", primals_4: "Sym(s1)", primals_5: "Sym(s1)", primals_6: "Sym(s0)", primals_7: "Sym(s1)"):
        clone: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        mul_6: "Sym(s0*s1)" = primals_3 * primals_4;  primals_3 = primals_4 = None
        view: "f32[s0*s1]" = torch.ops.aten.view.default(clone, [mul_6]);  clone = None
        view_1: "f32[s0*s1]" = torch.ops.aten.view.default(clone_1, [mul_6]);  clone_1 = None
        return (view, view_1, mul_6, primals_6, primals_7)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "Sym(s0)", primals_7: "Sym(s1)", tangents_1: "f32[s0*s1]", tangents_2: "f32[s0*s1]"):
        view_2: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_1, [primals_6, primals_7]);  tangents_1 = None
        view_3: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_2, [primals_6, primals_7]);  tangents_2 = None
        return (view_2, view_3, primals_6, primals_7, primals_7, None, None)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_return_tensor_and_subclass(self):
        def f(tt):
            y = tt.clone()
            return y.a, y.view(y.shape[0] * y.shape[1])

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[s0, s1]", primals_2: "f32[s0, s1]", primals_3: "Sym(s0)", primals_4: "Sym(s1)", primals_5: "Sym(s1)", primals_6: "Sym(s0)", primals_7: "Sym(s1)"):
        clone: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[s0, s1]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        mul_6: "Sym(s0*s1)" = primals_3 * primals_4;  primals_3 = primals_4 = None
        view: "f32[s0*s1]" = torch.ops.aten.view.default(clone, [mul_6])
        view_1: "f32[s0*s1]" = torch.ops.aten.view.default(clone_1, [mul_6]);  clone_1 = None
        return (clone, view, view_1, mul_6, primals_6, primals_7)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "Sym(s0)", primals_7: "Sym(s1)", tangents_1: "f32[s0*s1]", tangents_2: "f32[s0*s1]"):
        view_2: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_1, [primals_6, primals_7]);  tangents_1 = None
        view_3: "f32[s0, s1]" = torch.ops.aten.view.default(tangents_2, [primals_6, primals_7]);  tangents_2 = None
        return (view_2, view_3, primals_6, primals_7, primals_7, None, None)
""",  # noqa: B950
        )

    @unittest.expectedFailure
    def test_tensor_subclass_TwoTensor_return_multiple(self):
        def f(tt):
            y = tt.clone()
            z = tt.clone()
            return y.a, y.view(y.shape[0] * y.shape[1]), y.b, z.view(-1)

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 4]", primals_2: "f32[3, 4]", primals_3: "Sym(3)", primals_4: "Sym(4)", primals_5: "Sym(3)", primals_6: "Sym(4)"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        mul: "Sym(12)" = primals_5 * primals_6
        view: "f32[12]" = torch.ops.aten.view.default(clone, [mul])
        view_1: "f32[12]" = torch.ops.aten.view.default(clone_1, [mul]);  clone_1 = None
        return [clone, view, view_1, mul, primals_5, primals_6]
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_5: "Sym(3)", primals_6: "Sym(4)", tangents_1: "f32[12]", tangents_2: "f32[12]"):
        view_2: "f32[3, 4]" = torch.ops.aten.view.default(tangents_1, [primals_5, primals_6]);  tangents_1 = None
        view_3: "f32[3, 4]" = torch.ops.aten.view.default(tangents_2, [primals_5, primals_6]);  tangents_2 = primals_5 = primals_6 = None
        return [view_2, view_3, None, None]
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_automatic_dynamic_shapes(self):
        def f(tt):
            y = tt.clone()
            return y.a, y.view(-1), y.b

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt1 = TwoTensor(a, b)

        a = torch.ones(3, 5, requires_grad=True)
        b = a.clone()
        tt2 = TwoTensor(a, b)

        fw, bw = self._compile_check(
            f, [(tt1,), (tt2,)], dynamic=None, call_backward=True
        )

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, 4]", primals_2: "f32[3, 4]"):
        clone: "f32[3, 4]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, 4]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[12]" = torch.ops.aten.view.default(clone, [-1])
        view_1: "f32[12]" = torch.ops.aten.view.default(clone_1, [-1])
        return (clone, view, view_1, clone_1)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(fw[1].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, s0]", primals_2: "f32[3, s0]", primals_3: "Sym(s0)", primals_4: "Sym(s0)", primals_5: "Sym(s0)"):
        clone: "f32[3, s0]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, s0]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[3*s0]" = torch.ops.aten.view.default(clone, [-1])
        sym_numel_default: "Sym(3*s0)" = torch.ops.aten.sym_numel.default(clone)
        view_1: "f32[3*s0]" = torch.ops.aten.view.default(clone_1, [-1])
        return (clone, view, view_1, sym_numel_default, clone_1, primals_5)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[12]", tangents_2: "f32[12]"):
        view_2: "f32[3, 4]" = torch.ops.aten.view.default(tangents_1, [3, 4]);  tangents_1 = None
        view_3: "f32[3, 4]" = torch.ops.aten.view.default(tangents_2, [3, 4]);  tangents_2 = None
        return (view_2, view_3)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[1].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_5: "Sym(s0)", tangents_1: "f32[3*s0]", tangents_2: "f32[3*s0]"):
        view_2: "f32[3, s0]" = torch.ops.aten.view.default(tangents_1, [3, primals_5]);  tangents_1 = None
        view_3: "f32[3, s0]" = torch.ops.aten.view.default(tangents_2, [3, primals_5]);  tangents_2 = None
        return (view_2, view_3, primals_5, primals_5, None)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_mark_dynamic_shapes(self):
        def f(tt):
            y = tt.clone()
            return y.a, y.view(-1), y.b

        a = torch.ones(3, 4, requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)
        torch._dynamo.mark_dynamic(tt, 1)

        fw, bw = self._compile_check(
            f,
            [
                (tt,),
            ],
            dynamic=None,
            call_backward=True,
        )

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[3, s0]", primals_2: "f32[3, s0]", primals_3: "Sym(s0)", primals_4: "Sym(s0)", primals_5: "Sym(s0)"):
        clone: "f32[3, s0]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[3, s0]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[3*s0]" = torch.ops.aten.view.default(clone, [-1])
        sym_numel_default: "Sym(3*s0)" = torch.ops.aten.sym_numel.default(clone)
        view_1: "f32[3*s0]" = torch.ops.aten.view.default(clone_1, [-1])
        return (clone, view, view_1, sym_numel_default, clone_1, primals_5)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_5: "Sym(s0)", tangents_1: "f32[3*s0]", tangents_2: "f32[3*s0]"):
        view_2: "f32[3, s0]" = torch.ops.aten.view.default(tangents_1, [3, primals_5]);  tangents_1 = None
        view_3: "f32[3, s0]" = torch.ops.aten.view.default(tangents_2, [3, primals_5]);  tangents_2 = None
        return (view_2, view_3, primals_5, primals_5, None)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_different_shape(self):
        def f(tt):
            y = tt.clone()
            return y.view(3, 2, 4)

        a = torch.ones((2 * 4 * 3), requires_grad=True)
        b = a.clone()
        tt = TwoTensor(a, b)

        fw, bw = self._compile_check(f, [(tt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[24]", primals_2: "f32[24]"):
        clone: "f32[24]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None
        clone_1: "f32[24]" = torch.ops.aten.clone.default(primals_2);  primals_2 = None

        view: "f32[3, 2, 4]" = torch.ops.aten.view.default(clone, [3, 2, 4]);  clone = None
        view_1: "f32[3, 2, 4]" = torch.ops.aten.view.default(clone_1, [3, 2, 4]);  clone_1 = None
        return (view, view_1)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[3, 2, 4]", tangents_2: "f32[3, 2, 4]"):
        view_2: "f32[24]" = torch.ops.aten.view.default(tangents_1, [24]);  tangents_1 = None
        view_3: "f32[24]" = torch.ops.aten.view.default(tangents_2, [24]);  tangents_2 = None
        return (view_2, view_3)
""",  # noqa: B950
        )

    def test_tensor_subclass_TwoTensor_return_shape(self):
        @torch.compile(backend="aot_eager", dynamic=True)
        def fn(x):
            return x.clone().view(x.shape[0] * x.shape[1])

        a = torch.ones(2, 3)
        b = a.clone()
        tt = TwoTensor(a, b)
        out = fn(tt)
        self.assertEqual(tt.view(2 * 3), out)
        self.assertEqual(out.shape, (6,))

    def test_tensor_subclass_TwoTensor_nested(self):
        @torch.compile(backend="aot_eager", dynamic=True)
        def f(x, i, y):
            out1 = x.sin() + i.sin() + y.sin()
            val1 = x.shape[0] * i.shape[1] * y.shape[0]
            return out1 * val1

        i = torch.randn(2, 2, requires_grad=True)
        x = TwoTensor(i, i.clone())
        y = TwoTensor(x.clone(), x.clone())

        out = f(x, i, y)

        x_test = x.clone().detach().requires_grad_(True)
        i_test = i.clone().detach().requires_grad_(True)
        y_test = y.clone().detach().requires_grad_(True)

        out_test = f(x_test, i_test, y_test)
        torch.allclose(out, out_test)

        out.sum().backward()
        out_test.sum().backward()
        torch.allclose(x.grad, x_test.grad)
        torch.allclose(i.grad, i_test.grad)
        torch.allclose(y.grad, y_test.grad)

    def test_subclass_TwoTensor_TwoTensor_TwoTensor(self):
        @torch.compile(backend="aot_eager", dynamic=True)
        def f(x):
            return x.sin()

        data = torch.randn(2, 3)
        s = TwoTensor(data, data.clone())
        y = TwoTensor(s, s.clone())
        z = TwoTensor(s, y)
        out = f(z)
        self.assertEqual(out, z.sin())

    @unittest.expectedFailure
    def test_njt_subclass_simple(self):
        def f(nt):
            y = nt.clone()
            return y * y.size(0) * y.size(1)

        nt, _ = get_jagged_tensor(((2, 3, 4), 5), None, True)

        fw, bw = self._compile_check(f, [(nt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[s0, s1]", primals_2: "i64[s2 + 1]", primals_3: "f32[s6, 0]", primals_4: "f32[s7, 0]", primals_5: "Sym(s2)", primals_6: "Sym(s1)", primals_7: "Sym(s1)", primals_8: "Sym(s1)", primals_9: "Sym(s2)", primals_10: "Sym(s3)"):
        clone: "f64[s0, s1]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None

        mul: "f64[s0, s1]" = torch.ops.aten.mul.Tensor(clone, primals_9);  clone = None
        mul_1: "f64[s0, s1]" = torch.ops.aten.mul.Tensor(mul, primals_10);  mul = None
        return (mul_1, primals_2, primals_3, primals_4, primals_9, primals_8, primals_8, primals_8, primals_9, primals_10)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_8: "Sym(s1)", primals_9: "Sym(s2)", primals_10: "Sym(s3)", tangents_1: "f64[s0, s1]", tangents_2: "i64[s2 + 1]", tangents_3: "f32[s6, 0]", tangents_4: "f32[s7, 0]"):
        mul_2: "f64[s0, s1]" = torch.ops.aten.mul.Tensor(tangents_1, primals_10);  tangents_1 = primals_10 = None
        mul_3: "f64[s0, s1]" = torch.ops.aten.mul.Tensor(mul_2, primals_9);  mul_2 = None
        return (mul_3, tangents_2, tangents_3, tangents_4, primals_9, primals_8, primals_8, None, None, None)
""",  # noqa: B950
        )

    def test_njt_subclass_from_cat(self):
        # create from an existing NJT
        def f(nt):
            y = nt.clone()
            z = torch.cat([y, y], dim=-1)
            return z

        nt, _ = get_jagged_tensor(((2, 3, 4), 5), None, True)

        fw, bw = self._compile_check(f, [(nt,)], dynamic=True, call_backward=True)

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[s0, s1]", primals_2: "i64[s2 + 1]", primals_3: "f32[s6, 0]", primals_4: "f32[s7, 0]", primals_5: "Sym(s2)", primals_6: "Sym(s1)", primals_7: "Sym(s1)", primals_8: "Sym(s1)", primals_9: "Sym(s2)", primals_10: "Sym(s3)"):
        clone: "f64[s0, s1]" = torch.ops.aten.clone.default(primals_1);  primals_1 = None

        cat: "f64[s0, 2*s1]" = torch.ops.aten.cat.default([clone, clone], 1);  clone = None
        add_2: "Sym(2*s1)" = primals_8 + primals_8
        return (cat, primals_2, primals_3, primals_4, primals_9, add_2, add_2, primals_8, primals_9, add_2)
""",  # noqa: B950
        )

        self.assertExpectedInline(
            normalize_gm(bw[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_8: "Sym(s1)", primals_9: "Sym(s2)", add_2: "Sym(2*s1)", tangents_1: "f64[s0, 2*s1]", tangents_2: "i64[s2 + 1]", tangents_3: "f32[s6, 0]", tangents_4: "f32[s7, 0]"):
        slice_1: "f64[s0, s1]" = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, primals_8)
        slice_2: "f64[s0, s1]" = torch.ops.aten.slice.Tensor(tangents_1, 1, primals_8, add_2);  tangents_1 = add_2 = None

        add_4: "f64[s0, s1]" = torch.ops.aten.add.Tensor(slice_1, slice_2);  slice_1 = slice_2 = None
        return (add_4, tangents_2, tangents_3, tangents_4, primals_9, primals_8, primals_8, None, None, None)
""",  # noqa: B950
        )

    @unittest.expectedFailure
    def test_njt_subclass_from_buffer(self):
        # create the NJT from a buffer(?)
        def f(nt):
            nested_size = ((2, 3, 4), 5)
            offsets = None
            nt2, _ = get_jagged_tensor(nested_size, offsets, requires_grad=False)
            nt3 = torch.cat([nt2, nt], dim=-1)
            return nt3.sin() * nt3.size(1)

        nested_size = ((2, 3, 4), 5)
        offsets = None
        nt, _ = get_jagged_tensor(nested_size, offsets, requires_grad=False)

        fw, _ = self._compile_check(
            f,
            [(nt,)],
            dynamic=True,
            call_backward=False,  # we cannot set requires_grad=True inside a compile region
        )

        self.assertExpectedInline(
            normalize_gm(fw[0].print_readable(print_output=False)),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f64[9, s2]", arg1_1: "i64[s3 + 1]", arg2_1: "f32[s7, 0]", arg3_1: "f32[s8, 0]", arg4_1: "Sym(s3)", arg5_1: "Sym(s2)", arg6_1: "Sym(s2)", arg7_1: "Sym(s2)", arg8_1: "Sym(s3)", arg9_1: "Sym(s4)"):
        randn: "f64[2, 5]" = torch.ops.aten.randn.default([2, 5], dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
        randn_1: "f64[3, 5]" = torch.ops.aten.randn.default([3, 5], dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
        randn_2: "f64[4, 5]" = torch.ops.aten.randn.default([4, 5], dtype = torch.float64, device = device(type='cpu'), pin_memory = False)

        cat: "f64[9, 5]" = torch.ops.aten.cat.default([randn, randn_1, randn_2]);  randn = randn_1 = randn_2 = None

        cat_2: "f64[9, s2 + 5]" = torch.ops.aten.cat.default([arg0_1, cat], 1);  arg0_1 = cat = None

        sin: "f64[9, s2 + 5]" = torch.ops.aten.sin.default(cat_2)
        mul: "f64[9, s2 + 5]" = torch.ops.aten.mul.Tensor(sin, arg9_1);  sin = arg9_1 = None

        sym_size_int_1: "Sym(s2 + 5)" = torch.ops.aten.sym_size.int(cat_2, 1);  cat_2 = None
        sym_stride_int: "Sym(s2 + 5)" = torch.ops.aten.sym_stride.int(mul, 0)
        return (mul, arg1_1, arg2_1, arg3_1, arg8_1, sym_size_int_1, sym_stride_int)
""",  # noqa: B950
        )


instantiate_parametrized_tests(SubclassTests)


class TestNestedTensor(torch._dynamo.test_case.TestCase, NestedTensorTestCase):
    def _get_jagged_tensor(self, nested_size, offsets, requires_grad=True):
        return get_jagged_tensor(nested_size, offsets, requires_grad)

    def _get_nc_jagged_tensor(self, inner_dim, starts, lengths, requires_grad=True):
        # Makes a jagged tensor with N constituent tensors with size
        # as specified ((S0, S1, S2), D)
        max_dim = (starts + lengths).max()
        values_tensor = torch.randn(
            starts.shape[0],
            max_dim.item(),
            inner_dim,
            requires_grad=requires_grad,
            dtype=torch.float64,
        )
        return jagged_from_tensor_and_lengths(values_tensor, starts, lengths)

    def _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles):
        _check_recompiles(self, fn, inputs1, inputs2, expected_recompiles)

    def test_unary_does_not_recompile(self):
        nt1, _ = self._get_jagged_tensor(((2, 3, 4), 3), None)
        nt2, _ = self._get_jagged_tensor(((3, 4, 5, 6), 4), None)
        self._check_recompiles(lambda nt1: nt1.sin(), (nt1,), (nt2,), False)

    def test_binary_does_not_recompile(self):
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # NB: If we have shape e.g. (3, j0, 3), duck sizing will give us (s0, s1, s0).
        # This causes a recompile later on when it realizes the batch and last dim
        # should not always be equal. To avoid that, we use (3, j0, 5) here.
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, offsets = self._get_jagged_tensor(((3, 4, 5), 4), None)
        nt4, _ = self._get_jagged_tensor(((3, 4, 5), 4), offsets)
        self._check_recompiles(binary, (nt1, nt2), (nt3, nt4), False)

    def test_binary_recompiles(self):
        def binary(nt1, nt2):
            if nt1.shape == nt2.shape:
                return nt1 + nt2
            else:
                return nt1.sin()

        # Binary recompiles because singleton ints no longer match
        nt1, offsets = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 4), 5), offsets)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        self._check_recompiles(binary, (nt1, nt2), (nt1, nt3), True)

    def _validate_compile(self, fn, arg_fn):
        def _gen_grad_outputs(out_val):
            if isinstance(out_val, (list, tuple)):
                return tuple(torch.ones_like(c) for c in out_val)
            else:
                return (torch.ones_like(out_val),)

        with self.branch_nested_state():
            from torch.nested._internal.nested_tensor import _tensor_symint_registry

            # Validate that compilation does not modify eager state
            registry_before = list(_tensor_symint_registry.items())
            count_before = torch.nested._internal.nested_tensor._tensor_id_counter

            guards_exported = []
            guards_failed = []

            def append_guard_export(guards):
                for g in guards:
                    if g.code_list is not None:
                        guards_exported.append(g.code_list[0])

            def append_guard_fail(guards):
                guards_failed.extend(guards)

            compiled = torch._dynamo.optimize(
                nopython=True,
                backend="aot_eager",
                guard_export_fn=append_guard_export,
                guard_fail_fn=append_guard_fail,
            )(fn)
            registry_after = list(_tensor_symint_registry.items())
            count_after = torch.nested._internal.nested_tensor._tensor_id_counter
            self.assertEqual(registry_before, registry_after)
            self.assertEqual(count_before, count_after)

            args = arg_fn()
            compile_out = compiled(*args)
            compile_grads = []
            g_args = [arg for arg in args if arg.requires_grad]
            if len(g_args) > 0:
                compile_grad_outputs = _gen_grad_outputs(compile_out)
                compile_grads = torch.autograd.grad(
                    compile_out, inputs=g_args, grad_outputs=compile_grad_outputs
                )

        with self.branch_nested_state():
            args = arg_fn()
            ref_out = fn(*args)
            ref_grads = []
            g_args = [arg for arg in args if arg.requires_grad]
            if len(g_args) > 0:
                ref_grad_outputs = _gen_grad_outputs(ref_out)
                ref_grads = torch.autograd.grad(
                    ref_out, inputs=g_args, grad_outputs=ref_grad_outputs
                )

        # Validate correctness forward
        if isinstance(compile_out, (list, tuple)):
            # TODO: Fix assertEqual() to support NJTs so this isn't necessary
            self.assertEqual(len(compile_out), len(ref_out))
            for c, r in zip(compile_out, ref_out):
                self.assertEqualIgnoringNestedInts(c, r)
        else:
            self.assertEqualIgnoringNestedInts(compile_out, ref_out)

        # Validate correctness backward
        for compile_grad, ref_grad in zip(compile_grads, ref_grads):
            self.assertEqualIgnoringNestedInts(compile_grad, ref_grad)

        return guards_exported, guards_failed

    # Note: [What kind of guards are involved in nested tensor compilation]
    #
    # Until we implement UnionFind, dynamic shapes guards are not involved.
    # we rely only on dynamo's tensor aliasing guards.
    #
    # This is possible because dynamo able to generate tensor aliasing guards
    # not only for the outer tensor, but also for the inner tensor.
    #
    # The case where dynamic shapes guards would eventually come into play is
    # when my inputs are (1) two non-aliased tensors, but (2) declared as
    # equal using a "trust me assert equal" API.

    # Note: [Compiling nested tensor global state]
    #
    # Today there are two pieces of global eager state that NJTs deals with:
    # - tensor_id_counter: a global counter that assigns unique ids to tensors
    # - tensor_symint_registry: maps tensor to nested int
    #   - this is used in eager only (we should get rid of this because it is
    #     not necessary to cache nested int in eager)
    #   - during tracing, we DO need to cache nested int, but we do so on
    #     the FakeTensor.
    #
    # Ideally we would like to satisfy the following:
    # - (1) The eager state is not mutated during tracing
    # - (2) Running the compiled function should mutate the eager state in the
    #       same way that running the eager function would
    #       (a) The global counter should be incremented
    #       (b) The registry is updated in the same way
    #
    # Today we can satisfy (1) and (2a) but cannot satisfy (2b)
    #
    # Today, (1) is satisfied because we maintain a separate counter during
    # tracing, and cache nested int on FakeTensor instead of relying on
    # tensor_symint_registry.
    #
    # (2) is cannot be completely satisfied because we trace away the
    # side-effectful operations (which we can fix this by wrapping the
    # side-effectful operations in a custom op, and threading through effect
    # tokens.) The current plan is to do that in the UnionFind impl.
    #
    # Interestingly, despite this, the state is mutated in a way that is somewhat
    # close to what we want, e.g. if I construct a nested tensor using an
    # offsets in the compiled region and return it, AOTAutograd runtime wrapper
    # must rewrap the inner->inner graph outputs back into subclass. This
    # triggers the eager logic to run, updating the counter and registry.
    #
    # Notably however, compile differs in two ways from eager:
    # (1) The order in which the offsets are assigned ids is differnet
    #     the registry would be set in the order the offsets are returned
    #     which is not necessarily the same order as they were constructed.
    # (2) If a NestedTensor is not returned, then the AOTAutograd wrapping
    #     logic will not be triggered.
    #
    # I claim that correctness is not affected by these differences today.
    # e.g. there is never the case where two distinct offsets silently share
    # the same id.
    #
    # (1) is clearly not a problem, and (2) should only be a problem if
    # the nested int is returned on its own, without the corresponding NJT
    # being returned. This is not a problem in the current implementation
    # because returning only a shape is not supported!

    # Note: [Creating symbolic nested int]
    #
    # We must create a symbolic nested int when we construct a nested tensor
    # from a tensor. There are two main cases:
    #
    # 1. The offsets has NOT been used to construct a NJT
    #    - Create a new plain nested int with current val of fake nt id counter
    #    - Increment the fake nt id counter
    #    - Create a new symint with plain nested int as hint
    # 2. The offsets HAS been used to construct a NJT
    #    - Create a new symint with plain nested int as hint
    #
    # More details on case 2:
    # - During fakification of the offsets, we check the eager registry, and
    #   if the tensor HAS been used to construct a NJT,
    #   we create a symint, with the existing nested int as hint, and cache
    #   it on to the FakeTensor.
    #
    # [ Always use ephemeral source ]
    #
    # We create the new symint ALWAYS with ephemeral source whether that is
    # in case (1) or (2) even though we could've had a proper source for case (2).
    # Using a proper source would enable a few more (edge) cases, but since
    # we plan to handle things more holistically in the future anyway, we don't
    # bother doing so today.
    #
    # Using an ephemeral source has some consequences. But we are happy if
    # - We do not silently miss recompiles, e.g. we guard when necessary.
    #   We know that this is true, because dynamo guards alone are already
    #   sufficient.
    # - We are not producing errors for the cases we care about
    #
    # The main case we care about is when we guard that two shapes are equal.
    # In this case, the replacements logic would simplify away the ephemeral
    # symbol, and there is no error produced.
    # The unsupported case is when we guard that two shapes are not equal, in
    # which, we will try and fail to generate a guard.

    #
    # Case 1: in-graph construction where the offsets are passed as inputs
    #
    def test_in_graph_construction_from_input(self):
        # The offsets is passed as an input
        def fn(values, offsets):
            return torch.nested.nested_tensor_from_jagged(values * 2, offsets) * 2

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        self._validate_compile(fn, arg_fn=lambda: (values, offsets))

        # Do not specialize on the offsets
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            different_offsets = torch.tensor([0, 1, 5, 10], dtype=torch.int64)
            self._validate_compile(fn, arg_fn=lambda: (values, different_offsets))

    def test_in_graph_construction_from_input_2(self):
        # Construct two NJTs, both are passed as inputs
        def fn(values, offsets1, offsets2):
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets1)
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets2)
            return nt2, nt1

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        offsets2 = torch.tensor([0, 1, 4, 10], dtype=torch.int64)
        # 1. Offsets are different
        guards_exported, guards_failed = self._validate_compile(
            fn, arg_fn=lambda: (values, offsets, offsets2)
        )
        self.assertEqual(len(guards_failed), 0)
        self.assertNotIn("L['offsets1'] is L['offsets2']", guards_exported)

        # TODO
        # 2. Offsets are the same
        new_guards_exported, _ = self._validate_compile(
            fn, arg_fn=lambda: (values, offsets, offsets)
        )
        self.assertTrue(any("Duplicate tensors found" in g for g in guards_failed))
        self.assertIn("L['offsets1'] is L['offsets2']", new_guards_exported)

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            offsets3 = offsets.clone()
            self._validate_compile(fn, arg_fn=lambda: (values, offsets3, offsets3))

        # Do a binary op
        def fn(values, offsets, offsets2):
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets2)
            return nt1 * nt2

        self._validate_compile(fn, arg_fn=lambda: (values, offsets, offsets))

    def test_in_graph_construction_from_input_4(self):
        # The offsets is taken from an NJT input
        def fn(nt, other_values):
            nt2 = torch.nested.nested_tensor_from_jagged(other_values, nt.offsets())
            return nt + nt2

        values = torch.randn(9, 5, requires_grad=True)
        other_values = torch.randn(9, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 9], dtype=torch.int64)

        def arg_fn(values=values, other_values=other_values, offsets=offsets):
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt, other_values

        self._validate_compile(fn, arg_fn=arg_fn)

        # Do not specialize on the offsets
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            different_offsets = offsets.clone()

            def arg_fn(
                values=values, other_values=other_values, offsets=different_offsets
            ):
                nt = torch.nested.nested_tensor_from_jagged(values, different_offsets)
                return nt, other_values

            self._validate_compile(fn, arg_fn=arg_fn)

    def test_in_graph_construction_from_input_5(self):
        # Construct from lengths instead of offsets
        def fn(values, lengths):
            nt = torch.nested.nested_tensor_from_jagged(values, lengths=lengths)
            return nt.sin()

        values = torch.randn(9, 5, requires_grad=True)
        lengths = torch.tensor([2, 4, 3])
        self._validate_compile(fn, arg_fn=lambda: (values, lengths))

    #
    # Case 2: in-graph construction where offsets are graph intermediates
    #
    def test_in_graph_construction_from_intermediate(self):
        # offsets is an intermediate computed from lengths
        def fn(values, lengths):
            offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values, offsets)
            return (nt * nt2).sin()

        values = torch.randn(9, 5, requires_grad=True)
        lengths = torch.tensor([2, 4, 3])
        self._validate_compile(fn, arg_fn=lambda: (values, lengths))

        # Do not specialize on the lengths
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            different_lengths = lengths.clone()
            self._validate_compile(fn, arg_fn=lambda: (values, different_lengths))

    def test_in_graph_construction_from_intermediate_2(self):
        def fn(values, offsets):
            return torch.nested.nested_tensor_from_jagged(values * 2, offsets.clone())

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        self._validate_compile(fn, arg_fn=lambda: (values, offsets))

    def test_in_graph_construction_from_intermediate_3(self):
        # Note that due to CSE, clone is not necessarily called twice!
        def fn(values, offsets):
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets.clone())
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets.clone())
            return nt2, nt1

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        self._validate_compile(fn, arg_fn=lambda: (values, offsets))

    def test_in_graph_construction_from_intermediate_4(self):
        # Shared intermediate (should be same as case #1)
        def fn(values):
            offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            values2 = torch.ones_like(values)
            nt2 = torch.nested.nested_tensor_from_jagged(values2, offsets)
            return nt * nt2

        values = torch.randn(10, 5).requires_grad_(True)
        self._validate_compile(fn, arg_fn=lambda: (values,))

    # AssertionError: s2 (could be from ['<ephemeral: intermediate_offsets_or_lengths>',
    @unittest.expectedFailure
    def test_in_graph_construction_from_intermediate_5(self):
        # non-shared intermediate
        def fn(values):
            offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            values2 = torch.ones_like(values)
            nt2 = torch.nested.nested_tensor_from_jagged(values2, offsets.clone())
            if nt2.shape[1] != nt.shape[1]:
                return nt * 2
            else:
                return nt * 3

        values = torch.randn(10, 5).requires_grad_(True)
        self._validate_compile(fn, arg_fn=lambda: (values,))

    #
    # Case 3: in-graph construction where offsets are both direct graph inputs
    #         and passed in as part of an NJT's offsets.
    #
    def test_in_graph_construction_mixed(self):
        def fn(nt, values, offsets):
            nt2 = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt * nt2

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)

        def arg_fn(values=values, offsets=offsets):
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt, values, offsets

        self._validate_compile(fn, arg_fn)

    # See Note: [Creating symbolic nested int]
    # AssertionError: s2 (could be from ['<ephemeral: intermediate_offsets_or_lengths>',
    @unittest.expectedFailure
    def test_in_graph_construction_mixed_2(self):
        def fn(nt, values, offsets, nt2):
            # Intermediate offsets has ephemeral source
            intermediate_nt = torch.nested.nested_tensor_from_jagged(
                values, offsets.clone()
            )
            # This creates a dynamic shapes neq guard
            if nt2.shape[1] != intermediate_nt.shape[1]:
                # We should always go here.
                nt = nt * 2
            return nt

        values = torch.randn(10, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 10], dtype=torch.int64)
        offsets2 = torch.tensor([0, 1, 4, 10], dtype=torch.int64)

        def arg_fn(values=values, offsets=offsets, offsets2=offsets2):
            # Values is shared, but it shouldn't matter
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values, offsets2)
            return nt, values, offsets, nt2

        self._validate_compile(fn, arg_fn)

    def test_in_graph_construction_mixed_3(self):
        # More involved mixed case
        def fn(nt, values, offsets):
            nt1 = torch.nested.nested_tensor_from_jagged(values * 2, offsets)
            nt2 = torch.nested.nested_tensor_from_jagged(values * 3, offsets)
            return nt1 + nt2 + nt

        values = torch.randn(9, 5, requires_grad=True)
        offsets = torch.tensor([0, 2, 6, 9], dtype=torch.int64)

        def arg_fn(values=values, offsets=offsets):
            nt = torch.nested.nested_tensor_from_jagged(values, offsets)
            return nt, values, offsets

        self._validate_compile(fn, arg_fn)

    def test_return_shape(self):
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(nt):
            return (nt * 2).shape

        compiled = torch.compile(fn, fullgraph=True, backend="aot_eager")
        compiled(nt)

    def test_inference_tensor(self):
        with torch.inference_mode():
            nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(n):
            return n * 2

        torch.compile(fn, backend="eager")(nt)

    # TODO: cannot parametrize this test class with device for some reason
    def _test_autograd(self, backend):
        a = torch.randn(2, 3, requires_grad=True, dtype=torch.float64)
        b = torch.randn(3, 3, requires_grad=True, dtype=torch.float64)
        c = torch.randn(4, 3, requires_grad=True, dtype=torch.float64)
        nt = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
        # TODO: Switch to public API when it exists
        nt2, _ = jagged_from_list([a, b, c], nt.offsets())

        def fn1(nt1, nt2):
            return (nt1 + nt2).sin().cos()

        compiled_f = torch.compile(fn1, fullgraph=True, backend=backend, dynamic=True)
        out = compiled_f(nt, nt2)
        out_buffer = out.values()
        ga, gb, gc = torch.autograd.grad(out_buffer.sum(), (a, b, c))

        out_ref = fn1(nt, nt2)
        out_buffer_ref = out_ref.values()
        ga_ref, gb_ref, gc_ref = torch.autograd.grad(out_buffer_ref.sum(), (a, b, c))

        self.assertTrue(torch.allclose(ga, ga_ref))
        self.assertTrue(torch.allclose(gb, gb_ref))
        self.assertTrue(torch.allclose(gc, gc_ref))

    def test_basic_autograd(self):
        self._test_autograd("aot_eager")

    @requires_cuda
    def test_basic_autograd_inductor(self):
        self._test_autograd("inductor")

    def test_subclass_with_mutation_in_graph(self):
        # In this graph, we have an in-graph mutation, i.e. a mutation that is allowed
        # to remain in the graph. Normally this is allowed, but it's not allowed if
        # the graph handles subclasses at all.
        # Whether the mutation is allowed or not allowed in the graph alters the number
        # of outputs from the forward graph. Previously, a bug in this handling meant
        # that sometimes the expected number and actual number of outputs from the
        # joint graph did not match, causing assertion failures.
        def fn(x, y):
            z = x.sin()
            y.sin_()
            return z.cos(), y.cos()

        fn_c = torch.compile(fn, backend="inductor")

        values = [torch.rand((i, 8), requires_grad=True) for i in range(1, 6)]
        values_copy = [x.detach().clone().requires_grad_(True) for x in values]

        nt, offsets = jagged_from_list(values, None)
        nt_copy, offsets = jagged_from_list(values_copy, offsets)
        y = torch.rand((4, 8))
        y_copy = y.clone()

        ret = fn_c(nt, y)[0]
        ref = fn(nt_copy, y_copy)[0]

        self.assertEqual(ret.values(), ref.values())

        ret.values().sum().backward()
        ref.values().sum().backward()
        for ref_v, res_v in zip(values_copy, values):
            self.assertEqual(ref_v.grad, res_v.grad)

    @torch._dynamo.config.patch({"capture_scalar_outputs": True})
    def test_unbind(self):
        # NB: If we have shape e.g. (3, j0, 3), duck sizing will give us (s0, s1, s0).
        # This causes a recompile later on when it realizes the batch and last dim
        # should not always be equal. To avoid that, we use (3, j0, 5) here.
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)
        nt2, _ = self._get_jagged_tensor(((2, 3, 5), 2), None)
        nt3, _ = self._get_jagged_tensor(((2, 3, 4, 5), 3), None)

        def fn(x):
            return x.unbind()

        compiled_f = torch.compile(fn, fullgraph=True, backend="eager", dynamic=True)
        out = compiled_f(nt)

        out_ref = fn(nt)

        # correctness
        self.assertEqual(len(out), len(out_ref))
        for x, x_ref in zip(out, out_ref):
            self.assertTrue(torch.allclose(x, x_ref))

        # We specialize on the length of offsets, e.g. (1) we recompile if the
        # length of the offsets is different. (2) we don't recompile if the
        # length of the offsets is the same, even if the size of the constituent
        # tensors are different.
        self._check_recompiles(fn, (nt,), (nt2,), False)
        self._check_recompiles(fn, (nt,), (nt3,), True)

    def test_inline_nested_tensor_from_jagged(self):
        nt, _ = self._get_jagged_tensor(((2, 3, 4), 5), None)

        def fn(x):
            return torch.nested.nested_tensor_from_jagged(x.values() * 2, x.offsets())

        torch.compile(fn, fullgraph=True, backend="aot_eager")(nt)

    # The test here: nn.Parameters that are secretly subclasses
    # have a metaclass that overrides __isinstance__,
    # that dynamo needs to respect when it inlines the if statement.
    def test_param_subclass_isinstance_input(self):
        x_inner = torch.randn(16, 16, requires_grad=True)
        x = torch.nn.Parameter(TwoTensor(x_inner, x_inner))
        m = torch.nn.Linear(16, 16)
        m.weight = x

        def fn():
            if isinstance(m.weight, torch.nn.Parameter):
                return m.weight + 1
            else:
                return m.weight + 2

        out_ref = fn()
        out_test = torch.compile(fn, backend="aot_eager")()
        self.assertEqual(out_ref, out_test)

    def _input_view_test(self, nt_view_name):
        nt_view = VIEW_TEST_CASES[nt_view_name]()

        def fn(x):
            return x.sin()

        out_ref = fn(nt_view)
        torch._dynamo.reset()
        compile_fn = torch.compile(
            fn, fullgraph=True, backend="aot_eager", dynamic=True
        )
        out = compile_fn(nt_view)

        # Check metadata and values are correct
        self.assertTrue(out.size() == out_ref.size())
        self.assertTrue(out.stride() == out_ref.stride())
        if out.is_nested:
            self.assertTrue(torch.allclose(out.values(), out_ref.values()))
        else:
            self.assertTrue(torch.allclose(out, out_ref))

        # Check that no upper/lower bound guards are incurred
        def backend(gm, args):
            context = torch._guards.TracingContext.get()
            guards = [str(g.expr) for g in context.fake_mode.shape_env.guards]

            # varies based on the type of view
            guard_str = "\n".join(guards)
            if nt_view_name == "subclass_dense":
                self.assertExpectedInline(guard_str, """Eq(s3 - 1, s0)""")
            elif nt_view_name == "dense_subclass_dense_subclass":
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s5 - 1, s2)
Eq(s12 - 1, s7)
Eq(s11, s9)""",
                )
            elif nt_view_name.startswith("base_is_nt_True"):
                self.assertExpectedInline(
                    guard_str,
                    """Eq(s3 - 1, s0)""",
                )
            else:
                self.assertExpectedInline(
                    guard_str,
                    """\
Eq(s4 - 1, s1)
Eq(s13 - 1, s8)
Eq(s12, s10)""",
                )
            return gm

        torch._dynamo.reset()
        compile_fn = torch.compile(fn, fullgraph=True, backend=backend, dynamic=True)
        out = compile_fn(nt_view)

    @parametrize(
        "nt_view_name",
        [k for k in VIEW_TEST_CASES.keys() if k != "subclass_dense_subclass_dense"],
    )
    def test_inputs_to_compiled_fn_are_views(self, nt_view_name):
        self._input_view_test(nt_view_name)

    def test_subclass_gives_static_shapes_when_dynamic_false(self):
        def check_graph(gm, *args):
            first_node_example_val = next(iter(gm.graph.nodes)).meta["example_value"]
            # We compiled with dynamic=False, expect no SymInt sizes on our placeholders
            self.assertTrue(
                all(isinstance(x, int) for x in first_node_example_val.shape)
            )
            return gm

        @torch.compile(backend=check_graph, dynamic=False)
        def f(x):
            return x + 1

        x_inner = torch.ones(4)
        x = TwoTensor(x_inner, x_inner)
        x_view = x.view(2, 2)
        out = f(x_view)

    # NJT1 -> Dense -> NJT2 -> Dense view
    # During view replay, the Dense -> NJT2 part will construct an intermediate,
    # symbolically-sized NJT that is immediately deconstructed to return the final dense
    # view. To construct this intermediate properly, we need the associated nested int
    # to be symbolic. This view is expected to fail compilation until symbolic nested ints
    # are cached onto fake offsets to solve this problem.
    @unittest.expectedFailure
    def test_subclass_dense_subclass_dense_view(self):
        self._input_view_test("subclass_dense_subclass_dense")


instantiate_parametrized_tests(TestNestedTensor)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
