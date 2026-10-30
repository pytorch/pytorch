# Owner(s): ["module: dynamo"]
import types
from unittest.mock import patch

import torch
import torch._dynamo.test_case
from torch._dynamo import trace_rules
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable
from torch.testing._internal.common_utils import run_tests


class _NoRunBackend:
    # Compiles the graph without executing it: running synchronize_device at
    # runtime would fail on a host without an "npu" device (torch.accelerator
    # validates the device type against the current accelerator).
    def __init__(self) -> None:
        self.frame_count = 0

    def __call__(self, gm, example_inputs):
        self.frame_count += 1
        return lambda *args, **kwargs: None


def _make_privateuse1_module():
    # Every function the Dynamo probe looks for. current_stream/synchronize
    # RAISE if called directly: a compile that succeeds proves the call was
    # dispatched to the handler table instead of being inlined.
    def _raise(name):
        def _fn(*args, **kwargs):
            raise RuntimeError(
                f"torch.npu.{name} was called during tracing; Dynamo inlined it "
                "instead of dispatching to the handler table"
            )

        return _fn

    module = types.ModuleType("torch.npu")
    module.current_stream = _raise("current_stream")
    module.synchronize = _raise("synchronize")
    # exchange_device_helper calls the exchange fn at trace time and installs a
    # guard on current_device plus a set_device cleanup hook.
    module._exchange_device = lambda i: 1
    module._maybe_exchange_device = lambda i: 1
    module.current_device = lambda: 0
    module.set_device = lambda i: None
    return module


class PrivateUse1RoutingTests(torch._dynamo.test_case.TestCase):
    def test_privateuse1_routing(self):
        # The backend getter is patched rather than really renamed: a rename is
        # once-per-process and would leak into every later test.
        module = _make_privateuse1_module()
        fns = (
            module.current_stream,
            module.synchronize,
            module._exchange_device,
            module._maybe_exchange_device,
        )
        with patch.object(torch._C, "_get_privateuse1_backend_name", return_value="npu"):
            with patch.object(torch, "npu", module, create=True):
                TorchInGraphFunctionVariable._get_handlers.cache_clear()
                try:
                    handlers = TorchInGraphFunctionVariable._get_handlers()
                    for fn in fns:
                        self.assertIn(fn, handlers)
                        self.assertIs(
                            trace_rules.lookup_callable(fn),
                            TorchInGraphFunctionVariable,
                        )

                    # allow_in_graph registrations survive trace_rules cache
                    # clears, e.g. from torch.distributed.init_process_group.
                    trace_rules.clear_lru_cache()
                    for fn in fns:
                        self.assertIs(
                            trace_rules.lookup_callable(fn),
                            TorchInGraphFunctionVariable,
                        )

                    cnt = _NoRunBackend()

                    @torch.compile(backend=cnt, fullgraph=True)
                    def compiled_fn():
                        torch.npu.synchronize()
                        torch.npu.current_stream()
                        torch.npu._exchange_device(0)

                    compiled_fn()
                    self.assertEqual(cnt.frame_count, 1)
                finally:
                    TorchInGraphFunctionVariable._get_handlers.cache_clear()


if __name__ == "__main__":
    run_tests()
