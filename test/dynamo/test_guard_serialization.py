# Owner(s): ["module: dynamo"]

import dataclasses
import importlib
import pickle
import sys
import types

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.bytecode_transformation import transform_code_object
from torch._dynamo.guards import CheckFunctionManager, CompileId
from torch._dynamo.symbolic_convert import (
    ExceptionStack,
    InstructionTranslator,
    SpeculationLog,
)
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._guards import compile_context, CompileContext, tracing
from torch.utils import _pytree as pytree


@dataclasses.dataclass
class _FrameState:
    f_locals: dict
    f_globals: dict
    f_code: types.CodeType
    f_builtins: dict


class GlobalModule(torch.nn.Module):
    def forward(self, x):
        return x + 1


class TestGuardSerialization(torch._inductor.test_case.TestCase):
    def _tracefunc(self, frame, event, arg):
        if event != "call":
            return

        if self._frame_state is not None:
            return

        self._frame_state = _FrameState(
            f_locals=dict(frame.f_locals),
            f_globals=dict(frame.f_globals),
            f_code=frame.f_code,
            f_builtins=frame.f_builtins,
        )

    def _test_serialization(self, guard_type, fn, *args, **kwargs):
        self._frame_state = None
        sys.settrace(self._tracefunc)
        if isinstance(fn, torch.nn.Module):
            fn = fn.forward
        try:
            fn(*args, **kwargs)
        finally:
            sys.settrace(None)

        assert self._frame_state is not None

        def guard_filter_fn(guards):
            ret = [
                g.guard_type == guard_type or guard_type in g.derived_guard_types
                for g in guards
            ]
            self.assertTrue(any(ret))
            return ret

        ref_gm = None
        loaded_gm = None

        def transform(instructions: list, code_options: dict[str, object]):
            """
            The goal is here is not to reimplement dynamo, but just to have a
            simplified version to extract the state from symbolic convert.
            Should not work on all cases, but should work on simple functions
            in this test file.
            """
            nonlocal ref_gm
            nonlocal loaded_gm

            tracer = InstructionTranslator(
                instructions,
                self._frame_state.f_code,
                self._frame_state.f_locals,
                self._frame_state.f_globals,
                self._frame_state.f_builtins,
                fn.__closure__ or (),
                [],  # TODO tf_mode_stack,
                code_options,
                lambda gm, *args, **kwargs: gm.forward,
                one_graph=False,
                export=False,
                export_constraints=None,
                frame_state=None,
                speculation_log=SpeculationLog(),
                exn_vt_stack=ExceptionStack(),
                distributed_state=None,
            )
            with compile_context(CompileContext(CompileId(0, 0))), tracing(
                tracer.output.tracing_context
            ), tracer.set_current_tx(), get_metrics_context(), dynamo_timed(""):
                tracer.run()

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    tracer.output,
                    guard_filter_fn=guard_filter_fn,
                    guards_serialization_mode="save",
                )
                ref_gm = check_fn_manager.guard_manager
                guards_state = check_fn_manager.guards_state
                self.assertIsNotNone(guards_state)
                guards_state = pickle.loads(guards_state)

                check_fn_manager = CheckFunctionManager(
                    self._frame_state.f_code,
                    guards_state.output_graph,
                    guards_serialization_mode="load",
                )
                loaded_gm = check_fn_manager.guard_manager

        try:
            transform_code_object(self._frame_state.f_code, transform)
        finally:
            self._frame_state = None

        self.assertIsNotNone(ref_gm)
        self.assertIsNotNone(loaded_gm)
        return ref_gm, loaded_gm

    def _test_check_fn(self, ref, loaded, inputs, expected):
        self.assertIsInstance(inputs, dict)
        self.assertEqual(ref.check(inputs), expected)
        self.assertEqual(ref.check(inputs), loaded.check(inputs))

    def test_tensor_match(self):
        def f(x: torch.Tensor):
            return x + 1

        ref, loaded = self._test_serialization(
            "TENSOR_MATCH", f, torch.ones(2, dtype=torch.float32)
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(2, dtype=torch.float32)}, True
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(3, dtype=torch.float32)}, False
        )
        self._test_check_fn(
            ref, loaded, {"x": torch.randn(2, dtype=torch.float64)}, False
        )
        self._test_check_fn(ref, loaded, {"x": None}, False)

    def test_not_present_in_generic_dict(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x + 1

        m = Module()

        def fn(x):
            return m(x)

        ref, loaded = self._test_serialization(
            "NOT_PRESENT_IN_GENERIC_DICT", fn, torch.ones(2, dtype=torch.float32)
        )
        self._test_check_fn(ref, loaded, {"m": m}, True)

        m.forward = types.MethodType(lambda x: x + 2, m)
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_hasattr_serialization(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = 1

            def forward(self, x: torch.Tensor):
                if hasattr(self, "a"):
                    return x + self.a
                else:
                    return x + 2

        m = Module()

        def fn(x):
            return m(x)

        ref, loaded = self._test_serialization("HASATTR", fn, torch.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        delattr(m, "a")
        self._test_check_fn(ref, loaded, {"m": m}, False)

    def test_type_match(self):
        class LocalModule(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x + 1

        m = LocalModule()

        def fn(m, x):
            return m(x)

        with self.assertRaisesRegex(
            TypeError, "Please define the class at global scope"
        ):
            self._test_serialization("TYPE_MATCH", fn, m, torch.randn(3))

        m = GlobalModule()
        ref, loaded = self._test_serialization("TYPE_MATCH", fn, m, torch.randn(3))
        self._test_check_fn(ref, loaded, {"m": m}, True)
        self._test_check_fn(ref, loaded, {"m": GlobalModule()}, True)
        self._test_check_fn(ref, loaded, {"m": torch.nn.Module()}, False)

    def test_dict_version(self):
        def fn(x):
            return pytree.tree_leaves(x)[0] + 1

        with self.assertRaisesRegex(
            RuntimeError, "DICT_VERSION guard cannot be serialized."
        ):
            self._test_serialization("DICT_VERSION", fn, {"t": torch.randn(3)})

    def test_dict_contains(self):
        def fn(x):
            if x.__contains__("t"):
                return x["t"] + 1
            else:
                return torch.ones(3)

        ref, loaded = self._test_serialization(
            "DICT_CONTAINS", fn, {"t": torch.randn(3)}
        )

        self._test_check_fn(ref, loaded, {"x": {"t": torch.randn(3)}}, True)
        self._test_check_fn(ref, loaded, {"x": {}}, False)
        self._test_check_fn(
            ref, loaded, {"x": {"t": torch.randn(3), "d": torch.randn(3)}}, True
        )

    def test_bool_match(self):
        def fn(x, b):
            if b:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("BOOL_MATCH", fn, torch.randn(3), True)

        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": True}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": None}, False)

    def test_none_match(self):
        def fn(x, b):
            if b is None:
                return x + 1
            else:
                return x + 2

        ref, loaded = self._test_serialization("NONE_MATCH", fn, torch.randn(3), None)

        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": None}, True)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": False}, False)
        self._test_check_fn(ref, loaded, {"x": torch.randn(3), "b": True}, False)

    def test_id_match(self):
        def fn(x):
            return x + id(x)

        with self.assertRaisesRegex(
            RuntimeError, "ID_MATCH guard cannot be serialized."
        ):
            self._test_serialization("ID_MATCH", fn, torch.randn(3))

    def test_dispatch_key_set_match(self):
        def fn(x, dks):
            if dks.has("CPU"):
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x = torch.randn(3)
        dks = torch._C._dispatch_keys(x)
        ref, loaded = self._test_serialization("DISPATCH_KEY_SET_MATCH", fn, x, dks)

        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, True)

        x = torch.randn(3, device="meta")
        dks = torch._C._dispatch_keys(x)
        self._test_check_fn(ref, loaded, {"x": x, "dks": dks}, False)

    def test_name_match(self):
        def fn(x, y):
            return torch.cond(x, lambda x: y + 1, lambda x: y - 1, (y,))

        x = torch.tensor(True)
        y = torch.randn(3)
        ref, loaded = self._test_serialization("NAME_MATCH", fn, x, y)

        self._test_check_fn(ref, loaded, {"x": x, "y": y}, True)

        op = importlib.import_module("torch._higher_order_ops.cond").cond_op
        prev, op.__name__ = op.__name__, ""
        try:
            self._test_check_fn(ref, loaded, {"x": x, "y": y}, False)
        finally:
            op.__name__ = prev


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
