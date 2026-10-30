# Owner(s): ["module: dynamo"]
import types
from unittest.mock import patch

import torch
import torch._dynamo.test_case
from torch._dynamo import trace_rules
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
from torch.testing._internal.common_utils import run_tests


class _RecordOnlyBackend:
    # Compiles the graph without executing it: running synchronize_device at
    # runtime would fail on a host without an "npu" device (torch.accelerator
    # validates the device type against the current accelerator).
    def __init__(self) -> None:
        self.frame_count = 0
        self.graphs: list[torch.fx.GraphModule] = []

    def __call__(self, gm, example_inputs):
        self.frame_count += 1
        self.graphs.append(gm)
        return lambda *args, **kwargs: None


def _make_privateuse1_module():
    # Every function the Dynamo probe looks for. current_stream/synchronize
    # RAISE if called directly: a compile that succeeds proves the call was
    # dispatched to the handler table instead of being inlined.
    calls = {"current_device": 0, "set_device": 0}

    def _raise(name):
        def _fn(*args, **kwargs):
            raise RuntimeError(
                f"torch.npu.{name} was called during tracing; Dynamo inlined it "
                "instead of dispatching to the handler table"
            )

        return _fn

    def _current_device():
        calls["current_device"] += 1
        return 0

    def _set_device(i):
        calls["set_device"] += 1
        return None

    module = types.ModuleType("torch.npu")
    module.current_stream = _raise("current_stream")
    module.synchronize = _raise("synchronize")
    # exchange_device_helper calls the exchange fn at trace time and installs a
    # guard on current_device plus a set_device cleanup hook.
    module._exchange_device = lambda i: 1
    module._maybe_exchange_device = lambda i: 1
    module.current_device = _current_device
    module.set_device = _set_device
    return module, calls


class PrivateUse1RoutingTests(torch._dynamo.test_case.TestCase):
    def test_privateuse1_routing(self):
        # The backend getter is patched rather than really renamed: a rename is
        # once-per-process and would leak into every later test.
        module, calls = _make_privateuse1_module()
        fns = (
            module.current_stream,
            module.synchronize,
            module._exchange_device,
            module._maybe_exchange_device,
        )
        with (
            patch.object(torch._C, "_get_privateuse1_backend_name", return_value="npu"),
            patch.object(torch, "npu", module, create=True),
        ):
            trace_rules.get_torch_obj_rule_map.cache_clear()
            trace_rules.get_privateuse1_torch_obj_rule_map.cache_clear()
            TorchInGraphFunctionVariable._get_handlers.cache_clear()
            try:
                # The backend call is the first torch call traced, so this
                # compile is also the ordering regression: routing must be
                # resolved when the callable is first looked up, not later by
                # _get_handlers() (which runs only on the first dispatch).
                # current_stream is deliberately not traced here: its handler
                # needs an entered stream, and none exists on a host without a
                # real accelerator (its routing is still asserted below).
                cnt = _RecordOnlyBackend()

                @torch.compile(backend=cnt, fullgraph=True)
                def compiled_fn():
                    torch.npu.synchronize()
                    torch.npu._exchange_device(0)

                compiled_fn()
                compiled_fn()  # second call evaluates the exchange guard
                self.assertEqual(cnt.frame_count, 1)

                # synchronize dispatches to its handler, which emits the
                # device-type-specific synchronize_device op.
                graph = cnt.graphs[0]
                self.assertTrue(
                    any(
                        node.op == "call_function"
                        and node.target is torch.ops.streams.synchronize_device
                        and node.args[0] == "npu"
                        for node in graph.graph.nodes
                    )
                )

                # _exchange_device installs a guard on torch.npu.current_device
                # and restores the previous device through torch.npu.set_device.
                self.assertGreaterEqual(calls["current_device"], 1)
                self.assertGreaterEqual(calls["set_device"], 1)

                for fn in fns:
                    self.assertIn(fn, TorchInGraphFunctionVariable._get_handlers())
                    self.assertIs(
                        trace_rules.lookup(fn), TorchInGraphFunctionVariable
                    )

                # Routing is re-established after rule-map cache clears, e.g.
                # from torch.distributed.init_process_group.
                trace_rules.clear_lru_cache()
                for fn in fns:
                    self.assertIs(
                        trace_rules.lookup(fn), TorchInGraphFunctionVariable
                    )
            finally:
                trace_rules.get_torch_obj_rule_map.cache_clear()
                trace_rules.get_privateuse1_torch_obj_rule_map.cache_clear()
                TorchInGraphFunctionVariable._get_handlers.cache_clear()


if __name__ == "__main__":
    run_tests()
