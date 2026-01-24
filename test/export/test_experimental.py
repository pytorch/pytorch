# Owner(s): ["oncall: export"]
# flake8: noqa
import copy
import types
import unittest
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch._dynamo
from torch._dynamo.functional_export import (
    _dynamo_graph_capture_for_export,
    dynamo_graph_capture_for_export,
)
from torch._dynamo.test_case import run_tests, TestCase
from torch._functorch.aot_autograd import aot_export_module
from torch.export import export
from torch.export.experimental import _export_forward_backward, _sticky_export
from torch.export.graph_signature import OutputKind
from torch.testing import FileCheck
from torch.testing._internal.common_utils import TEST_CUDA
from torch.utils import _pytree as pytree


GLOBAL_LIST = []


def _register_blockmask_pytree():
    """Register BlockMask as a pytree node if not already registered."""
    from torch.nn.attention.flex_attention import BlockMask
    from torch.utils._pytree import register_pytree_node, SUPPORTED_NODES

    if BlockMask not in SUPPORTED_NODES:
        register_pytree_node(
            BlockMask,
            BlockMask._flatten,
            BlockMask._unflatten,
            flatten_with_keys_fn=BlockMask._flatten_with_keys,
            serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
        )


class GlobalContext:
    def __init__(self) -> None:
        self._summaries: dict[str, MetricValue] = {}
        self._tensors: dict[str, Tensor] = {}

    def __flatten__(self):
        """Flattens into (leaves, ctx)."""
        summary_leaves, summary_spec = pytree.tree_flatten(self._summaries)
        tensor_leaves, tensor_spec = pytree.tree_flatten(self._tensors)
        leaves = (*summary_leaves, *tensor_leaves)
        ctx = (summary_spec, tensor_spec)
        return leaves, ctx

    @classmethod
    def __unflatten__(cls, leaves, ctx: tuple[pytree.TreeSpec, pytree.TreeSpec]):
        """Reconstructs from (leaves, ctx)."""
        output = cls()
        summary_spec, tensor_spec = ctx
        assert len(leaves) == summary_spec.num_leaves + tensor_spec.num_leaves
        output._summaries = pytree.tree_unflatten(
            leaves[: summary_spec.num_leaves], summary_spec
        )
        output._tensors = pytree.tree_unflatten(
            leaves[summary_spec.num_leaves :], tensor_spec
        )
        return output

    def __enter__(self) -> "GlobalContext":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass


@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestExperiment(TestCase):
    def test_joint_basic(self) -> None:
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.loss = torch.nn.CrossEntropyLoss()

            def forward(self, x):
                return self.loss(
                    self.linear(x).softmax(dim=0), torch.tensor([1.0, 0.0, 0.0])
                )

        m = Module()
        example_inputs = (torch.randn(3),)
        m(*example_inputs)
        with torch._export.config.patch(use_new_tracer_experimental=True):
            ep = torch.export.export(m, example_inputs, strict=True)
        joint_ep = _export_forward_backward(ep)
        self.assertExpectedInline(
            str(joint_ep.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, c_lifted_tensor_0, x):
    view = torch.ops.aten.view.default(x, [1, 3]);  x = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, view, permute);  p_linear_bias = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [3]);  addmm = None
    _softmax = torch.ops.aten._softmax.default(view_1, 0, False);  view_1 = None
    alias = torch.ops.aten.alias.default(_softmax)
    clone = torch.ops.aten.clone.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_1 = torch.ops.aten.alias.default(_log_softmax)
    mul = torch.ops.aten.mul.Tensor(_log_softmax, clone);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, clone);  expand = clone = None
    alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    exp = torch.ops.aten.exp.default(alias_2);  alias_2 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_3 = torch.ops.aten.alias.default(alias);  alias = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_3);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_3, sum_3);  alias_3 = sum_3 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    view_2 = torch.ops.aten.view.default(sub_1, [1, 3]);  sub_1 = None
    permute_1 = torch.ops.aten.permute.default(view_2, [1, 0])
    mm = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
    sum_4 = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
    view_3 = torch.ops.aten.view.default(sum_4, [3]);  sum_4 = None
    permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return (div, permute_3, view_3)""",
        )
        ep = joint_ep.run_decompositions()
        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, c_lifted_tensor_0, x):
    view = torch.ops.aten.view.default(x, [1, 3]);  x = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, view, permute);  p_linear_bias = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [3]);  addmm = None
    _softmax = torch.ops.aten._softmax.default(view_1, 0, False);  view_1 = None
    alias = torch.ops.aten.alias.default(_softmax)
    clone = torch.ops.aten.clone.default(c_lifted_tensor_0);  c_lifted_tensor_0 = None
    _log_softmax = torch.ops.aten._log_softmax.default(_softmax, 0, False);  _softmax = None
    alias_1 = torch.ops.aten.alias.default(_log_softmax)
    mul = torch.ops.aten.mul.Tensor(_log_softmax, clone);  _log_softmax = None
    sum_1 = torch.ops.aten.sum.dim_IntList(mul, []);  mul = None
    neg = torch.ops.aten.neg.default(sum_1);  sum_1 = None
    div = torch.ops.aten.div.Scalar(neg, 1);  neg = None
    full_like = torch.ops.aten.full_like.default(div, 1, pin_memory = False, memory_format = torch.preserve_format)
    div_1 = torch.ops.aten.div.Scalar(full_like, 1);  full_like = None
    neg_1 = torch.ops.aten.neg.default(div_1);  div_1 = None
    expand = torch.ops.aten.expand.default(neg_1, [3]);  neg_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, clone);  expand = clone = None
    alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    exp = torch.ops.aten.exp.default(alias_2);  alias_2 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True)
    mul_2 = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
    sub = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    alias_3 = torch.ops.aten.alias.default(alias);  alias = None
    mul_3 = torch.ops.aten.mul.Tensor(sub, alias_3);  sub = None
    sum_3 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True)
    mul_4 = torch.ops.aten.mul.Tensor(alias_3, sum_3);  alias_3 = sum_3 = None
    sub_1 = torch.ops.aten.sub.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    view_2 = torch.ops.aten.view.default(sub_1, [1, 3]);  sub_1 = None
    permute_1 = torch.ops.aten.permute.default(view_2, [1, 0])
    mm = torch.ops.aten.mm.default(permute_1, view);  permute_1 = view = None
    permute_2 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
    sum_4 = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
    view_3 = torch.ops.aten.view.default(sum_4, [3]);  sum_4 = None
    permute_3 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    return (div, permute_3, view_3)""",
        )

    def _test_export_blockmask_with_mask_fn(self, make_mask_fn):
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        class Model(torch.nn.Module):
            def __init__(self, mask_fn_factory):
                super().__init__()
                self.mask_fn_factory = mask_fn_factory

            def forward(self, x):
                mask_fn = self.mask_fn_factory()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model(make_mask_fn)

        out_eager, mask_eager = module(x)

        compiled = _dynamo_graph_capture_for_export(module)(x)
        out_compiled, mask_compiled = compiled(x)

        self.assertEqual(out_eager, out_compiled)
        self.assertEqual(
            mask_eager.mask_mod(1, 1, 64, 64),
            mask_compiled.mask_mod(1, 1, 64, 64),
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask(self):
        def make_mask_fn():
            res = 4

            def fn(b, h, q, k):
                return q >= k + res

            return fn

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_mutated_closure(self):
        def make_mask_fn():
            res = 1

            def fn(b, h, q, k):
                return q >= k + res

            res = 4  # mutation after function definition
            return fn

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_with_containers(self):
        def make_mask_fn():
            offsets = [1, 2, 3]
            config = {"base": 4, "nested": {"scale": 2}}

            def fn(b, h, q, k):
                return q >= k + config["base"] + sum(offsets)

            return fn

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_triple_nested(self):
        def make_mask_fn():
            a = 1

            def level1():
                b = 2

                def level2():
                    c = 3

                    def fn(bx, h, q, k):
                        return q >= k + a + b + c

                    return fn

                return level2()

            return level1()

        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_self_recursive(self):
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn():
            # Self-referential: fn captures itself through the closure
            def fn(b, h, q, k):
                _ = fn  # self-reference
                return q >= k + 4

            return fn

        class Model(torch.nn.Module):
            def forward(self, x):
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_tensor(self):
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn():
            tensor = torch.ones(2, 2)

            def fn(b, h, q, k):
                _ = fn
                return q >= k + 4 + tensor.sum()

            return fn

        class Model(torch.nn.Module):
            def forward(self, x):
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_unsupported_class_instance(self):
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        class MaskConfig:
            def __init__(self, offset):
                self.offset = offset

        def make_mask_fn():
            cfg = MaskConfig(offset=5)

            def fn(b, h, q, k):
                return q >= k + cfg.offset

            return fn

        class Model(torch.nn.Module):
            def forward(self, x):
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask_closure_mutually_recursive(self):
        from torch.nn.attention.flex_attention import create_block_mask

        _register_blockmask_pytree()

        def make_mask_fn():
            # Create mutually recursive closures: fn_a references fn_b, fn_b references fn_a
            # This is non-constructible because we cannot serialize mutually recursive closures
            def fn_a(b, h, q, k):
                _ = fn_b  # reference to fn_b
                return q >= k

            def fn_b(b, h, q, k):
                _ = fn_a  # reference to fn_a
                return q >= k + 1

            return fn_a

        class Model(torch.nn.Module):
            def forward(self, x):
                mask_fn = make_mask_fn()
                block_mask = create_block_mask(
                    mask_fn, B=1, H=1, Q_LEN=64, KV_LEN=64, device=x.device
                )
                return x, block_mask

        x = torch.randn(2, 128, device="cuda")
        module = Model()

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "nested function with non-constructible closure in output",
        ):
            _dynamo_graph_capture_for_export(module)(x)

    def test_joint_dynamic(self) -> None:
        from torch.export import Dim

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.y = torch.nn.Parameter(torch.randn(3))

            def forward(self, x):
                x = torch.ones(x.shape[0], 3)
                return (self.y + x).sum()

        m = Module()
        example_inputs = (torch.randn(3),)
        m(*example_inputs)
        ep = torch.export.export(
            m, example_inputs, dynamic_shapes={"x": {0: Dim("x0")}}, strict=True
        )
        _export_forward_backward(ep)

    def test_joint_cifar10_backwards(self) -> None:
        import torch.nn as nn
        import torch.nn.functional as F

        # From Pytorch's CIFAR10 example:
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)
                self.loss = nn.CrossEntropyLoss()

            def forward(self, x, labels):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)  # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return self.loss(x, labels)

        net = Net()
        x = torch.randn(4, 3, 32, 32)
        labels = torch.ones(4, dtype=torch.int64)
        inputs = (x, labels)

        ep = export(net, inputs, strict=True)
        ep = _export_forward_backward(ep)

    def test_joint_loss_index(self):
        class Foo(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                self.l = torch.nn.Linear(4, 4)
                self.index = index

            def forward(self, x):
                x = self.l(x)
                x = x.sum()
                if self.index == 0:
                    return x, -x.detach()
                else:
                    return x.detach(), x

        inputs = (torch.randn(4, 4),)
        for i in [0, 1]:
            ep = export(Foo(i), inputs, strict=True)
            ep_joint = _export_forward_backward(ep, joint_loss_index=i)
            for j, spec in enumerate(ep_joint.graph_signature.output_specs):
                if i == j:
                    self.assertTrue(spec.kind == OutputKind.LOSS_OUTPUT)
                else:
                    self.assertTrue(spec.kind != OutputKind.LOSS_OUTPUT)

    def test_joint_buffer_input_mutations(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = torch.nn.Linear(4, 4)
                self.register_buffer("buf", torch.randn(4))
                self.loss = torch.nn.CrossEntropyLoss()

            def forward(self, x, label):
                x.add_(self.buf)
                x = self.l(x)
                self.buf.add_(2.0)
                return self.loss(x, label)

        inputs = (
            torch.randn(4, 4),
            torch.randint(0, 4, (4,)),
        )
        ep = export(Foo(), inputs)
        ep_joint = _export_forward_backward(ep)
        self.assertEqual(len(ep_joint.graph_signature.output_specs), 5)
        self.assertEqual(
            ep_joint.graph_signature.output_specs[0].kind,
            OutputKind.BUFFER_MUTATION,
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[0].target,
            "buf",
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[1].kind,
            OutputKind.USER_INPUT_MUTATION,
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[1].target,
            "x",
        )
        self.assertEqual(
            ep_joint.graph_signature.output_specs[2].kind,
            OutputKind.LOSS_OUTPUT,
        )

    def test_sticky_export(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        class Pipeline:
            def __init__(self, model):
                self.model = model

            def generate(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        inp = torch.randn(4, 4)

        p = Pipeline(Model())
        orig_forward = p.model.forward
        p.model.forward = _sticky_export(p.model.forward)
        res = p.generate(inp)

        p.model.forward = orig_forward
        res2 = p.generate(inp)
        self.assertTrue(torch.allclose(res, res2))

    def test_sticky_export_dynamic(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                if x.shape[0] < 5:
                    return self.linear(x)
                return x.sin()

        class Pipeline:
            def __init__(self, model):
                self.model = model

            def generate(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        inp = torch.randn(4, 4)

        def callback(*args, **kwargs):
            # I think it is bit weird to use the forward arg name here, so
            # lets just use ShapeCollections

            flat_args, _ = torch.utils._pytree.tree_flatten((args, kwargs))
            collections = torch.export.ShapesCollection()
            for arg in flat_args:
                if isinstance(arg, torch.Tensor):
                    collections[arg] = {
                        i: torch.export.Dim.AUTO for i in range(len(arg.shape))
                    }
            return collections

        p = Pipeline(Model())
        p.model.forward = _sticky_export(
            p.model.forward, dynamic_shapes_callback=callback
        )
        _ = p.generate(inp)
        self.assertExpectedInline(
            str(p.model.forward._exported_artifact.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    linear_weight = self.linear.weight
    linear_bias = self.linear.bias
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    linear = torch.ops.aten.linear.default(x, linear_weight, linear_bias);  x = linear_weight = linear_bias = None
    return pytree.tree_unflatten((linear,), self._out_spec)""",
        )

    def test_sticky_export_nested_inp(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, *, inputs):
                return self.linear(inputs[0]) + self.linear(inputs[1])

        class Pipeline:
            def __init__(self, model):
                self.model = model

            def generate(self, *, input_tensor, input_tensor2):
                inputs = [input_tensor, input_tensor2]
                return self.model(inputs=inputs)

        inp = torch.randn(4, 4)
        inp2 = torch.randn(4, 4)

        p = Pipeline(Model())
        orig_forward = p.model.forward
        p.model.forward = _sticky_export(p.model.forward)
        res = p.generate(input_tensor=inp, input_tensor2=inp2)

        p.model.forward = orig_forward
        res2 = p.generate(input_tensor=inp, input_tensor2=inp2)
        self.assertTrue(torch.allclose(res, res2))

    def test_side_effect(self):
        global_env = []

        class Foo(torch.nn.Module):
            def forward(self, x):
                global_env.append(x)
                return x.sin()

        with torch._dynamo.config.patch(replay_side_effects=False):
            _ = dynamo_graph_capture_for_export(Foo())(torch.randn(4, 4))
            self.assertEqual(len(global_env), 0)

    def test_export_add_in_out_info(self):
        class Foo(torch.nn.Module):
            def forward(self, dct, lst, bleh):
                x = dct["a"] * lst[1][0]
                y = dct["b"] * lst[0]
                out_dict = {}
                # Mutate and get a new entry in there
                lst_copy = lst.copy()
                lst_copy.append(lst[0])
                out_dict["a"] = x
                out_dict["b"] = y
                return (
                    dct["a"],
                    out_dict["b"],
                    bleh,
                    lst_copy[-1],
                    out_dict["a"],
                    [5, 6],
                )

        dct = {"a": torch.randn(2, 3), "b": torch.randn(2, 3)}
        lst = [torch.randn(2, 3), [torch.randn(2, 3), torch.randn(2, 3)]]

        export_inputs = ((dct, lst, 56), {})
        eager_inputs = copy.deepcopy(export_inputs)

        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        graph_module = dynamo_graph_capture_for_export(Foo())(
            *export_inputs[0], **export_inputs[1]
        )

        res_export = graph_module(*export_inputs[0], **export_inputs[1])
        res_eager = Foo()(*eager_inputs[0], **eager_inputs[1])

        self.assertEqual(res_export, res_eager)

    def test_export_leaf(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        export_inputs = ((torch.randn(4, 4),), {})
        eager_inputs = copy.deepcopy(export_inputs)

        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        graph_module = dynamo_graph_capture_for_export(Foo())(
            *export_inputs[0], **export_inputs[1]
        )

        res_export = graph_module(*export_inputs[0], **export_inputs[1])
        res_eager = Foo()(*eager_inputs[0], **eager_inputs[1])

        self.assertEqual(res_export, res_eager)

    def test_dynamo_graph_capture(self):
        class Foo(torch.nn.Module):
            def forward(self, dct, lst, bleh):
                x = dct["a"] * lst[1][0]
                y = dct["b"] * lst[0]
                out_dict = {}

                # Mutate and get a new entry in there
                lst_copy = lst.copy()
                lst_copy.append(lst[0])
                out_dict["a"] = x
                out_dict["b"] = y
                return (
                    dct["a"],
                    out_dict["b"],
                    bleh,
                    lst_copy[-1],
                    out_dict["a"],
                    [5, 6],
                )

        foo = Foo()

        def make_inputs():
            return (
                {"a": torch.randn(2, 3), "b": torch.randn(2, 3)},
                [torch.randn(2, 3), (torch.randn(2, 3),)],
                torch.randn(2, 3),
            )

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))

    def test_dynamo_graph_capture_with_call_override(self):
        class _InterestingModule(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self._module = module

            def __call__(self, *args, **kwargs):
                return self._module(*args, **kwargs)

        class MyModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        foo = _InterestingModule(MyModel())

        def make_inputs():
            return (torch.randn(2, 3),)

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        self.assertEqual(len(list(gm.buffers())), len(list(foo.buffers())))
        self.assertEqual(len(list(gm.parameters())), len(list(foo.parameters())))

    def test_dynamo_graph_capture_custom_pytree_type(self):
        import torch.utils._pytree as pytree

        @dataclass
        class Bar:
            x: torch.Tensor
            y: torch.Tensor

        class Foo(torch.nn.Module):
            def forward(self, bar: Bar):
                return bar.x + bar.y

        foo = Foo()

        def make_inputs():
            return (Bar(torch.randn(2, 3), torch.randn(2, 3)),)

        pytree.register_dataclass(Bar)
        try:
            trace_inputs = make_inputs()
            gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
            test_inputs = make_inputs()
            self.assertExpectedInline(
                gm.code.strip("\r\n "),
                """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_bar_x , L_bar_y , = self._dynamo_bytecode_flatten(*_fn_args)
    l_bar_x = L_bar_x
    l_bar_y = L_bar_y
    add = l_bar_x + l_bar_y;  l_bar_x = l_bar_y = None
    return self._dynamo_bytecode_unflatten((add,), _fn_args)""",
            )
            self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        finally:
            pytree._deregister_pytree_node(Bar)

    def test_dynamo_graph_capture_closure(self):
        from torch.export import Dim

        N = 3
        outer = torch.randn(10, 32)

        class MyModel(torch.nn.Module):
            def forward(self, x):
                z = x + outer
                y = z[:-1, :]  # [s0 - 1, 32]
                stacked = torch.stack([y] * N, dim=0)  # [N * (s0 - 1), 32]
                reshaped = stacked.reshape(-1, N, 32)  # [(s0 - 1), N, 32]
                return reshaped

        inps = (torch.randn(10, 32),)
        ep = dynamo_graph_capture_for_export(MyModel())(*inps)
        self.assertExpectedInline(
            ep.code.strip("\r\n "),
            """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_x_ , L_outer_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_x_ = L_x_
    l_outer_ = L_outer_
    z = l_x_ + l_outer_;  l_x_ = l_outer_ = None
    y = z[(slice(None, -1, None), slice(None, None, None))];  z = None
    stacked = torch.stack([y, y, y], dim = 0);  y = None
    reshaped = stacked.reshape(-1, 3, 32);  stacked = None
    return self._dynamo_bytecode_unflatten((reshaped,), _fn_args)""",
        )
        self.assertEqual(ep(*inps), MyModel()(*inps))

    def test_dynamo_graph_capture_full_tracing_context(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + x.shape[0]

        foo = Foo()

        def make_inputs(b: int):
            ret = (torch.randn(b, 3),)
            torch._dynamo.mark_dynamic(ret[0], 0)
            return ret

        trace_inputs = make_inputs(2)
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs(3)
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        self.assertIsNotNone(gm.meta["tracing_context"].fake_mode)
        self.assertEqual(len(gm.meta["tracing_context"].tensor_to_context), 1)

    def test_dynamo_graph_capture_ctx_return(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                with GlobalContext() as ctx:
                    z = x + 1
                    ctx._tensors["6"] = x + 2
                return z, ctx

        def make_inputs():
            return (torch.randn(2, 3),)

        try:
            pytree.register_pytree_node(
                GlobalContext,
                lambda x: x.__flatten__(),
                GlobalContext.__unflatten__,
            )
            mod = Module()

            gm = dynamo_graph_capture_for_export(mod)(*make_inputs())
            test_inputs = make_inputs()
            actual_outputs = pytree.tree_leaves(gm(*test_inputs))
            expected_outputs = pytree.tree_leaves(mod(*test_inputs))
            self.assertEqual(actual_outputs, expected_outputs)
        finally:
            pytree._deregister_pytree_node(GlobalContext)

    def test_dynamo_graph_capture_dict_keys_getitem(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x * 2

        foo = Module()

        class BlockMask:
            def __init__(self, d):
                self.d = d

        block_mask = BlockMask(torch.randn(4))

        def pre_hook_function(m, input):
            block_mask.d = input[0] + 1
            return input  # Return a tuple of modified inputs

        foo.register_forward_pre_hook(pre_hook_function)

        def make_inputs():
            return (torch.randn(4),)

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertExpectedInline(
            gm.code.strip("\r\n "),
            """\
def forward(self, args_0):
    _fn_args = (args_0, )
    L_args_0_ , = self._dynamo_bytecode_flatten(*_fn_args)
    l_args_0_ = L_args_0_
    add = l_args_0_ + 1;  add = None
    mul = l_args_0_ * 2;  l_args_0_ = None
    return self._dynamo_bytecode_unflatten((mul,), _fn_args)""",
        )
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))

    def test_dynamo_graph_capture_with_tensor_constant(self):
        outer = torch.randn(2, 3)

        class MyModel(torch.nn.Module):
            def forward(self, x):
                z = x + outer
                return z

        foo = MyModel()

        def make_inputs():
            return (torch.randn(2, 3),)

        trace_inputs = make_inputs()
        gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
        test_inputs = make_inputs()
        self.assertEqual(gm(*test_inputs), foo(*test_inputs))
        self.assertEqual(len(list(gm.buffers())), len(list(foo.buffers())))
        self.assertEqual(len(list(gm.parameters())), len(list(foo.parameters())))

    def test_dynamo_graph_capture_side_effects(self):
        GLOBAL_LIST.clear()

        def foo(x):
            z = x + 1
            GLOBAL_LIST.append(z)
            return z

        def make_inputs():
            return (torch.randn(2, 3),)

        trace_inputs = make_inputs()
        with (
            torch._dynamo.config.patch(replay_side_effects=False),
            warnings.catch_warnings(record=True) as w,
        ):
            gm = dynamo_graph_capture_for_export(foo)(*trace_inputs)
            cnt = 0
            for entry in w:
                if "While compiling, we found certain side effects happened" in str(
                    entry.message
                ):
                    cnt += 1
            self.assertEqual(cnt, 1)
        self.assertEqual(len(GLOBAL_LIST), 0)
        test_inputs = make_inputs()
        gm_results = gm(*test_inputs)
        self.assertEqual(len(GLOBAL_LIST), 0)
        self.assertEqual(gm_results, foo(*test_inputs))
        self.assertEqual(len(GLOBAL_LIST), 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_dynamo_graph_capture_fx_graph_annotate_overlap_pass(self):
        class DummyOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, scalar):
                ctx.save_for_backward(x)
                return x + scalar

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out, None

        def mock_fw_compute(x):
            with fx_traceback.annotate({"compute": 0}):
                return DummyOp.apply(x, 10)

        def mock_bw_comm(x):
            with fx_traceback.annotate({"comm": 0}):
                return DummyOp.apply(x, 20)

        def mock_bw_compute(x):
            return DummyOp.apply(x, 30)

        class Model(torch.nn.Module):
            def forward(self, fw_in, bw_in):
                fw_out = mock_fw_compute(fw_in)
                # bw_in blocks bw_out
                bw_in = mock_bw_comm(bw_in)
                bw_out = mock_bw_compute(bw_in)
                return fw_out, bw_out

        def input_fn():
            inputs = (torch.rand(2, 128, device="cuda", requires_grad=True),)
            grad_ins = (torch.rand(2, 128, device="cuda"),)
            return (
                *inputs,
                *grad_ins,
            )

        with torch.device("meta"):
            model = Model()

        import torch.fx.traceback as fx_traceback

        with fx_traceback.preserve_node_meta():
            gm = dynamo_graph_capture_for_export(model)(*input_fn())

        """
        def forward(self, args_0, args_1):
            _tree_leaf_0, _tree_leaf_1, _tree_leaf_2, = pytree.tree_leaves((self, args_0, args_1,))
            L_fw_in_ , L_bw_in_ , = self._in_shuffle_graph(_tree_leaf_0, _tree_leaf_1, _tree_leaf_2)
            l_fw_in_ = L_fw_in_
            l_bw_in_ = L_bw_in_
            fwd_body_0 = self.fwd_body_0
            bwd_body_0 = self.bwd_body_0
            fw_out = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_fw_in_, args_tensor_mask = [True, False], non_differentiable_idx = []);  fwd_body_0 = bwd_body_0 = l_fw_in_ = None
            bw_in = l_bw_in_ + 20;  l_bw_in_ = None
            bw_out = bw_in + 30;  bw_in = None
            return pytree.tree_unflatten(self._out_shuffle_graph(_tree_leaf_0, _tree_leaf_1, _tree_leaf_2, fw_out, bw_out), self._out_spec)
        """
        test_inputs = input_fn()
        self.assertEqual(gm(*test_inputs), model(*test_inputs))

    def test_dynamo_graph_capture_default_args(self):
        class Module(torch.nn.Module):
            def forward(self, x, y=1):
                return x + y

        m = Module()
        ep = dynamo_graph_capture_for_export(m)(torch.randn(2, 3))
        test_inputs = (torch.randn(2, 3),)
        self.assertEqual(ep(*test_inputs), m(*test_inputs))

    def test_restore_state_dict_basic(self):
        """Test basic state dict restoration with a simple model."""
        from torch.export import _restore_state_dict

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.register_buffer("buf", torch.randn(4))

            def forward(self, x):
                return self.linear(x) + self.buf

        model = Model()
        gm = dynamo_graph_capture_for_export(model)(torch.randn(2, 4))

        # Before restoration, FQNs may be flattened
        state_dict_before = dict(gm.named_parameters())
        buffer_dict_before = dict(gm.named_buffers())

        _restore_state_dict(model, gm)

        # After restoration, FQNs should match original module
        state_dict_after = dict(gm.named_parameters())
        buffer_dict_after = dict(gm.named_buffers())

        # Check that the parameter names match the original
        original_param_names = set(dict(model.named_parameters()).keys())
        restored_param_names = set(state_dict_after.keys())
        self.assertEqual(original_param_names, restored_param_names)

        # Check that the buffer names match the original
        original_buffer_names = set(dict(model.named_buffers()).keys())
        restored_buffer_names = set(buffer_dict_after.keys())
        self.assertEqual(original_buffer_names, restored_buffer_names)

        # Verify the model still works correctly
        test_input = torch.randn(2, 4)
        expected = model(test_input)
        actual = gm(test_input)
        self.assertEqual(expected, actual)

    def test_restore_state_dict_nested_modules(self):
        """Test state dict restoration with nested modules."""
        from torch.export import _restore_state_dict

        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 8)
                self.register_buffer("scale", torch.tensor(2.0))

            def forward(self, x):
                return self.fc(x) * self.scale

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule()

            def forward(self, x):
                return self.sub1(x) + self.sub2(x)

        model = Model()
        gm = dynamo_graph_capture_for_export(model)(torch.randn(2, 8))

        _restore_state_dict(model, gm)

        # Check that nested FQNs are properly restored
        original_param_names = set(dict(model.named_parameters()).keys())
        restored_param_names = set(dict(gm.named_parameters()).keys())
        self.assertEqual(original_param_names, restored_param_names)

        # Check buffers too
        original_buffer_names = set(dict(model.named_buffers()).keys())
        restored_buffer_names = set(dict(gm.named_buffers()).keys())
        self.assertEqual(original_buffer_names, restored_buffer_names)

        # Verify functionality
        test_input = torch.randn(2, 8)
        expected = model(test_input)
        actual = gm(test_input)
        self.assertEqual(expected, actual)

    def test_restore_state_dict_with_bound_method(self):
        """Test state dict restoration when passing a bound method."""
        from torch.export import _restore_state_dict

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x)

        model = Model()
        gm = dynamo_graph_capture_for_export(model.forward)(torch.randn(2, 4))

        # Pass bound method instead of module
        _restore_state_dict(model.forward, gm)

        # Verify FQNs match
        original_param_names = set(dict(model.named_parameters()).keys())
        restored_param_names = set(dict(gm.named_parameters()).keys())
        self.assertEqual(original_param_names, restored_param_names)

    def test_restore_state_dict_type_error(self):
        """Test that _restore_state_dict raises TypeError for invalid input."""
        from torch.export import _restore_state_dict

        gm = torch.fx.GraphModule({}, torch.fx.Graph())

        # Should raise TypeError when given a non-module, non-bound-method
        with self.assertRaises(TypeError):
            _restore_state_dict(lambda x: x, gm)


if __name__ == "__main__":
    run_tests()
