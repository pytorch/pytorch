# Owner(s): ["module: functorch"]

import logging
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch._functorch.config
from torch._functorch._aot_autograd.schemas import PlainTensorMeta
from torch._functorch._aot_autograd.subclass_codegen import (
    _codegen_subclass_wrapper_source,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.two_tensor import TwoTensor


trace_log = logging.getLogger("torch.__trace")


@dataclass
class _TestSubclassMeta:
    flat_tensor_start_idx: int
    arg_count: int
    included_subclass_symints: bool
    attrs: dict
    outer_size: tuple
    outer_stride: tuple
    meta: Any
    original_subclass: Any
    original_subclass_type: type
    outer_size_from_attr: str | None = None
    outer_stride_from_attr: str | None = None


class TestSubclassCodegen(TestCase):
    def _two_tensor_meta(self) -> _TestSubclassMeta:
        return _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=2,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(4,),
            outer_stride=(1,),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
        )

    @contextmanager
    def _capture_wrapper_source(self):
        """Capture subclass_wrapper artifacts from the structured trace log."""
        captured: list[str] = []

        class _ArtifactHandler(logging.Handler):
            def emit(self, record):
                metadata = getattr(record, "metadata", {})
                if (
                    "artifact" in metadata
                    and metadata["artifact"].get("name") == "subclass_wrapper"
                ):
                    payload = getattr(record, "payload", None)
                    if payload is not None:
                        captured.append(payload)

        handler = _ArtifactHandler()
        handler.setLevel(logging.DEBUG)
        old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)
        trace_log.addHandler(handler)
        try:
            yield captured
        finally:
            trace_log.removeHandler(handler)
            trace_log.setLevel(old_level)

    def test_compile_simple(self):
        """torch.compile with TwoTensor produces correct wrapper and output."""
        with self._capture_wrapper_source() as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x * 2

            a = torch.randn(4)
            b = torch.randn(4)
            tt = TwoTensor(a, b)
            out = f(tt)

        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, a * 2)
        self.assertEqual(out.b, b * 2)

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_0 = args[0]
    assert type(_inp_0) is _expected_type_1, f'expected {_expected_type_1}, got {type(_inp_0)}'
    unwrapped_args.append(_inp_0.a)
    unwrapped_args.append(_inp_0.b)
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _has_subclass_symint_outputs = len(unwrapped_outs) == 2
    _out_attr_3 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_attr_4 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_inner_2 = {'a': _out_attr_3, 'b': _out_attr_4}
    _out_7 = _subclass_type_5.__tensor_unflatten__(_out_inner_2, _meta_6, _out_attr_3.size(), _out_attr_3.stride())
    return (_out_7,)""",
        )

    def test_compile_nested_subclass(self):
        """torch.compile with nested TwoTensor produces correct recursive wrapper."""
        with self._capture_wrapper_source() as captured:

            @torch.compile(backend="aot_eager")
            def f(x):
                return x.sin().cos()

            a1 = torch.randn(4)
            a2 = torch.randn(4)
            a3 = torch.randn(4)
            a4 = torch.randn(4)
            inner_a = TwoTensor(a1, a2)
            inner_b = TwoTensor(a3, a4)
            tt = TwoTensor(inner_a, inner_b)
            out = f(tt)

        self.assertIsInstance(out, TwoTensor)
        self.assertIsInstance(out.a, TwoTensor)
        self.assertIsInstance(out.b, TwoTensor)
        self.assertEqual(out.a.a, inner_a.a.sin().cos())
        self.assertEqual(out.b.b, inner_b.b.sin().cos())

        self.assertEqual(len(captured), 1)
        self.assertExpectedInline(
            captured[0],
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_0 = args[0]
    assert type(_inp_0) is _expected_type_1, f'expected {_expected_type_1}, got {type(_inp_0)}'
    _inner_2 = _inp_0.a
    unwrapped_args.append(_inner_2.a)
    unwrapped_args.append(_inner_2.b)
    _inner_3 = _inp_0.b
    unwrapped_args.append(_inner_3.a)
    unwrapped_args.append(_inner_3.b)
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _has_subclass_symint_outputs = len(unwrapped_outs) == 4
    _out_attr_6 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_attr_7 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_inner_5 = {'a': _out_attr_6, 'b': _out_attr_7}
    _out_10 = _subclass_type_8.__tensor_unflatten__(_out_inner_5, _meta_9, _out_attr_6.size(), _out_attr_6.stride())
    _out_attr_12 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_attr_13 = unwrapped_outs[_out_idx]
    _out_idx += 1
    _out_inner_11 = {'a': _out_attr_12, 'b': _out_attr_13}
    _out_16 = _subclass_type_14.__tensor_unflatten__(_out_inner_11, _meta_15, _out_attr_12.size(), _out_attr_12.stride())
    _out_inner_4 = {'a': _out_10, 'b': _out_16}
    _out_19 = _subclass_type_17.__tensor_unflatten__(_out_inner_4, _meta_18, _out_10.size(), _out_10.stride())
    return (_out_19,)""",
        )

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_act_wait_top_level_codegen(self):
        """ACT paths on plain inputs emit waits during input unwrapping."""
        source, _ = _codegen_subclass_wrapper_source(
            inp_metas=[PlainTensorMeta(unwrapped_idx=0)],
            out_metas=[PlainTensorMeta(unwrapped_idx=0)],
            num_fw_outs_saved_for_bw=None,
            act_input_paths=[(0, ())],
        )

        self.assertExpectedInline(
            source,
            """\
def inner_fn(args):
    unwrapped_args = []
    unwrapped_args.append(_maybe_wait_act_0(args[0]))
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _has_subclass_symint_outputs = len(unwrapped_outs) == 1
    _out_idx = max(_out_idx, 1)
    return (unwrapped_outs[0],)""",
        )

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_act_wait_nested_codegen(self):
        """ACT paths inside subclasses emit waits at the matching leaf."""
        source, _ = _codegen_subclass_wrapper_source(
            inp_metas=[self._two_tensor_meta()],
            out_metas=[PlainTensorMeta(unwrapped_idx=0)],
            num_fw_outs_saved_for_bw=None,
            act_input_paths=[(0, ("a",))],
        )

        self.assertExpectedInline(
            source,
            """\
def inner_fn(args):
    unwrapped_args = []
    _inp_1 = args[0]
    assert type(_inp_1) is _expected_type_2, f'expected {_expected_type_2}, got {type(_inp_1)}'
    _resolved_3 = _maybe_wait_act_0(_inp_1.a)
    unwrapped_args.append(_resolved_3)
    unwrapped_args.append(_inp_1.b)
    unwrapped_args.extend(args[1:])
    args.clear()
    unwrapped_outs = compiled_fn(unwrapped_args)
    _out_idx = 0
    _has_subclass_symint_outputs = len(unwrapped_outs) == 1
    _out_idx = max(_out_idx, 1)
    return (unwrapped_outs[0],)""",
        )

    def test_trailing_args_forwarded(self):
        """Extra trailing args (e.g. rng seed/offset) are forwarded to compiled_fn."""
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        inp_meta = self._two_tensor_meta()
        out_meta = self._two_tensor_meta()

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[inp_meta],
            out_metas=[out_meta],
            num_fw_outs_saved_for_bw=None,
        )

        received_args = []

        def mock_compiled_fn(args):
            received_args.extend(args)
            return [args[0] * 2, args[1] * 2]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        a = torch.randn(4)
        b = torch.randn(4)
        tt = TwoTensor(a, b)
        seed = torch.tensor(42)
        offset = torch.tensor(0)
        # Simulate FunctionalizedRngRuntimeWrapper appending rng state
        wrapper([tt, seed, offset])

        self.assertEqual(len(received_args), 4)
        self.assertEqual(received_args[0], a)
        self.assertEqual(received_args[1], b)
        self.assertIs(received_args[2], seed)
        self.assertIs(received_args[3], offset)

    def test_attr_derived_metadata_does_not_skip_outputs(self):
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        out_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[],
            out_metas=[out_meta],
            num_fw_outs_saved_for_bw=1,
        )

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        activation = torch.randn(4)

        def mock_compiled_fn(args):
            return [a, b, a.shape[0], a.shape[1], a.stride()[0], activation]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        out, saved_activation = wrapper([])

        self.assertIsInstance(out, TwoTensor)
        self.assertEqual(out.a, a)
        self.assertEqual(out.b, b)
        self.assertIs(saved_activation, activation)

    def test_attr_derived_metadata_consumes_symints_between_outputs(self):
        # Build SubclassCreationMeta manually to avoid __post_init__ fake tensor check
        first_meta = _TestSubclassMeta(
            flat_tensor_start_idx=0,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=0),
                "b": PlainTensorMeta(unwrapped_idx=1),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )
        second_meta = _TestSubclassMeta(
            flat_tensor_start_idx=5,
            arg_count=5,
            included_subclass_symints=True,
            attrs={
                "a": PlainTensorMeta(unwrapped_idx=5),
                "b": PlainTensorMeta(unwrapped_idx=6),
            },
            outer_size=(None, None),
            outer_stride=(None, 1),
            meta=None,
            original_subclass=None,
            original_subclass_type=TwoTensor,
            outer_size_from_attr="a",
            outer_stride_from_attr="a",
        )

        source, globals_dict = _codegen_subclass_wrapper_source(
            inp_metas=[],
            out_metas=[first_meta, second_meta],
            num_fw_outs_saved_for_bw=None,
        )

        a0 = torch.randn(2, 3)
        b0 = torch.randn(2, 3)
        a1 = torch.randn(4, 5)
        b1 = torch.randn(4, 5)

        def mock_compiled_fn(args):
            return [
                a0,
                b0,
                a0.shape[0],
                a0.shape[1],
                a0.stride()[0],
                a1,
                b1,
                a1.shape[0],
                a1.shape[1],
                a1.stride()[0],
            ]

        globals_dict["compiled_fn"] = mock_compiled_fn
        local_dict = {}
        exec(compile(source, "<test>", "exec"), globals_dict, local_dict)
        wrapper = local_dict["inner_fn"]

        first, second = wrapper([])

        self.assertIsInstance(first, TwoTensor)
        self.assertEqual(first.a, a0)
        self.assertEqual(first.b, b0)
        self.assertIsInstance(second, TwoTensor)
        self.assertEqual(second.a, a1)
        self.assertEqual(second.b, b1)


if __name__ == "__main__":
    run_tests()
