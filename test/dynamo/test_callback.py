# Owner(s): ["module: dynamo"]

import unittest
from unittest.mock import Mock

import torch
from torch._dynamo.callback import callback_handler, CallbackArgs, CallbackTrigger
from torch._dynamo.test_case import run_tests, TestCase
from torch._guards import CompileId
from torch.testing._internal.common_utils import TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import HAS_CUDA


class CallbackTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._on_compile_start = Mock()
        self._on_compile_end = Mock()
        callback_handler.register_start_callback(self._on_compile_start)
        callback_handler.register_end_callback(self._on_compile_end)

    def tearDown(self) -> None:
        callback_handler.clear()
        return super().tearDown()

    def test_callbacks_with_duplicate_prevention(self) -> None:
        trigger = CallbackTrigger.DYNAMO
        compile_id = CompileId(0, 0)
        with (
            callback_handler.install_callbacks(trigger, compile_id),
            callback_handler.install_callbacks(trigger, compile_id),
        ):
            self._on_compile_start.assert_called_once()
        self._on_compile_end.assert_called_once()

    def test_counter(self) -> None:
        trigger = CallbackTrigger.DYNAMO
        compile_id = CompileId(0, 0)
        with callback_handler.install_callbacks(trigger, compile_id):
            self.assertEqual(
                callback_handler._CompilationCallbackHandler__pending_callbacks_counter,
                1,
            )
        self.assertEqual(
            callback_handler._CompilationCallbackHandler__pending_callbacks_counter, 0
        )

    def test_counter_assertion(self) -> None:
        callback_handler._CompilationCallbackHandler__pending_callbacks_counter -= 1
        with self.assertRaisesRegex(
            AssertionError, "Pending callbacks counter cannot become negative."
        ):
            trigger = CallbackTrigger.DYNAMO
            compile_id = CompileId(0, 0)
            with callback_handler.install_callbacks(trigger, str(compile_id)):
                pass
        self.assertEqual(
            callback_handler._CompilationCallbackHandler__pending_callbacks_counter, 0
        )

    @unittest.skipIf(
        TEST_WITH_ROCM, "ROCm outputs a different number of autotuning logs"
    )
    @unittest.skipIf(not HAS_CUDA, "requires triton")
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_triggers(self) -> None:
        torch._dynamo.reset()
        order = []

        def on_start(args: CallbackArgs):
            nonlocal order
            order.append(f"start={args}")

        def on_end(args: CallbackArgs):
            nonlocal order
            order.append(f"end={args}")

        torch._dynamo.callback.on_compile_start(on_start)
        torch._dynamo.callback.on_compile_start(on_end)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                temp = self.fc1(x)
                temp = self.relu(temp)
                torch._dynamo.graph_break()
                return self.fc2(temp)

        model = TinyModel().to("cuda")
        compiled_model = torch.compile(model, mode="max-autotune")
        x = torch.randn(10, 10, device="cuda")

        loss = compiled_model(x).sum()
        loss.backward()
        self.assertExpectedInline(
            "\n".join(order),
            """\
start=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='0/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.TRITON_AUTOTUNING: 3>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.TRITON_AUTOTUNING: 3>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='0/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.TRITON_AUTOTUNING: 3>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.TRITON_AUTOTUNING: 3>, compile_id='0/0')""",  # noqa: B950
        )
        order.clear()

        compiled_model.zero_grad()
        loss = compiled_model(x).sum()
        loss.backward()
        self.assertExpectedInline(
            "\n".join(order),
            """\
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')""",  # noqa: B950
        )
        order.clear()

        compiled_model.zero_grad()
        loss = compiled_model(x).sum()
        loss.backward()
        self.assertEqual(len(order), 0)


if __name__ == "__main__":
    run_tests()
