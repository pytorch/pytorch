# Owner(s): ["module: functorch"]

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import torch
from torch._functorch._aot_autograd import utils as aot_utils
from torch._functorch._aot_autograd.collect_metadata_analysis import (
    run_functionalized_fw_and_collect_metadata,
)
from torch._functorch._aot_autograd.descriptors import PlainAOTInput
from torch._functorch._aot_autograd.utils import _get_autocast_states
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _collect_metadata(f: Callable[..., list[torch.Tensor]]):
    collect = run_functionalized_fw_and_collect_metadata(
        f,
        flat_args_descs=[PlainAOTInput(0)],
        keep_input_mutations=False,
    )
    fake_mode = FakeTensorMode()
    x = fake_mode.from_tensor(torch.randn(2))
    return collect, x


class TestAOTAutocastStates(TestCase):
    def test_get_autocast_states_includes_supported_device(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "privateuseone"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ) as cache_enabled,
        ):
            enabled.side_effect = lambda device: device == "privateuseone"
            dtype.side_effect = lambda device: {
                "cpu": torch.bfloat16,
                "privateuseone": torch.float16,
            }[device]

            states = _get_autocast_states()

            self.assertEqual(
                states,
                [
                    False,
                    torch.bfloat16,
                    True,
                    torch.float16,
                    cache_enabled.return_value,
                ],
            )

    def test_get_autocast_states_reads_without_top_level_torch_module(self) -> None:
        device_type = "autocast_only_device"
        if hasattr(torch, device_type):
            self.skipTest(f"torch.{device_type} unexpectedly exists")
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", device_type],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ),
        ):
            enabled.side_effect = lambda device: device == device_type
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                device_type: torch.float16,
            }[device]

            states = _get_autocast_states()

            enabled.assert_any_call(device_type)
            dtype.assert_any_call(device_type)
            self.assertEqual(
                states,
                [False, torch.float32, True, torch.float16, False],
            )

    def test_get_autocast_states_appends_cache_enabled_once(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "cuda"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled", return_value=False),
            patch.object(
                aot_utils.torch, "get_autocast_dtype", return_value=torch.float32
            ),
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=True
            ),
        ):
            states = _get_autocast_states()
            self.assertEqual(states.count(True), 1)
            self.assertEqual(states[-1], True)

    def test_get_autocast_states_preserves_supported_device_order(self) -> None:
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "cuda", "xpu"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ),
        ):
            enabled.side_effect = lambda device: device != "cpu"
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                "cuda": torch.bfloat16,
                "xpu": torch.float16,
            }[device]

            self.assertEqual(
                _get_autocast_states(),
                [
                    False,
                    torch.float32,
                    True,
                    torch.bfloat16,
                    True,
                    torch.float16,
                    False,
                ],
            )

    def test_get_autocast_states_includes_renamed_privateuse1_backend(self) -> None:
        # DeviceTypeName(PrivateUse1) already returns the renamed backend name in
        # _autocast_supported_devices(); the snapshot must accept that string.
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "renamed_backend"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=True
            ),
        ):
            enabled.side_effect = lambda device: device == "renamed_backend"
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                "renamed_backend": torch.float16,
            }[device]

            self.assertEqual(
                _get_autocast_states(),
                [False, torch.float32, True, torch.float16, True],
            )

    def test_get_autocast_states_sees_third_party_enabled_cuda_cpu_legacy_missed(
        self,
    ) -> None:
        # Old hardcoded [cuda_en, cpu_en, cuda_dtype, cpu_dtype, cache] never read
        # privateuseone. The new snapshot must include that device's enabled bit.
        with (
            patch.object(
                aot_utils.torch._C,
                "_autocast_supported_devices",
                return_value=["cpu", "privateuseone"],
            ),
            patch.object(aot_utils.torch, "is_autocast_enabled") as enabled,
            patch.object(aot_utils.torch, "get_autocast_dtype") as dtype,
            patch.object(
                aot_utils.torch, "is_autocast_cache_enabled", return_value=False
            ),
        ):
            enabled.side_effect = lambda device: device == "privateuseone"
            dtype.side_effect = lambda device: {
                "cpu": torch.float32,
                "privateuseone": torch.float16,
            }[device]

            states = _get_autocast_states()
            self.assertEqual(
                states, [False, torch.float32, True, torch.float16, False]
            )
            # Index 2 is privateuseone enabled in the interleaved layout.
            self.assertTrue(states[2])

    def test_collect_metadata_preserves_stable_autocast_state(self) -> None:
        def f(x: torch.Tensor) -> list[torch.Tensor]:
            return [x + 1]

        collect, x = _collect_metadata(f)
        before = _get_autocast_states()
        collect(x)
        self.assertEqual(before, _get_autocast_states())

    def test_collect_metadata_detects_autocast_state_mutation(self) -> None:
        # mtia is always present in _AUTOCAST_SUPPORTED_DEVICES.
        extra = "mtia"
        prior_enabled = torch.is_autocast_enabled(extra)
        prior_dtype = torch.get_autocast_dtype(extra)

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            torch.set_autocast_enabled(extra, not prior_enabled)
            return [x + 1]

        collect, x = _collect_metadata(f)
        try:
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                collect(x)
            self.assertEqual(torch.is_autocast_enabled(extra), prior_enabled)
            self.assertEqual(torch.get_autocast_dtype(extra), prior_dtype)
        finally:
            torch.set_autocast_enabled(extra, prior_enabled)
            torch.set_autocast_dtype(extra, prior_dtype)

    def test_collect_metadata_detects_cpu_autocast_enabled_mutation(self) -> None:
        prior_enabled = torch.is_autocast_enabled("cpu")
        prior_dtype = torch.get_autocast_dtype("cpu")

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            torch.set_autocast_enabled("cpu", not prior_enabled)
            return [x + 1]

        collect, x = _collect_metadata(f)
        try:
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                collect(x)
            self.assertEqual(torch.is_autocast_enabled("cpu"), prior_enabled)
        finally:
            torch.set_autocast_enabled("cpu", prior_enabled)
            torch.set_autocast_dtype("cpu", prior_dtype)

    def test_collect_metadata_detects_cpu_autocast_dtype_mutation(self) -> None:
        prior_dtype = torch.get_autocast_dtype("cpu")
        new_dtype = torch.bfloat16 if prior_dtype != torch.bfloat16 else torch.float16

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            torch.set_autocast_dtype("cpu", new_dtype)
            return [x + 1]

        collect, x = _collect_metadata(f)
        try:
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                collect(x)
            self.assertEqual(torch.get_autocast_dtype("cpu"), prior_dtype)
        finally:
            torch.set_autocast_dtype("cpu", prior_dtype)

    def test_collect_metadata_restores_dtype_when_forward_raises(self) -> None:
        prior_dtype = torch.get_autocast_dtype("cpu")
        new_dtype = torch.bfloat16 if prior_dtype != torch.bfloat16 else torch.float16

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            torch.set_autocast_dtype("cpu", new_dtype)
            raise RuntimeError("forward failed after dtype mutation")

        collect, x = _collect_metadata(f)
        try:
            with self.assertRaisesRegex(RuntimeError, "forward failed after dtype mutation"):
                collect(x)
            self.assertEqual(torch.get_autocast_dtype("cpu"), prior_dtype)
        finally:
            torch.set_autocast_dtype("cpu", prior_dtype)

    def test_collect_metadata_allows_balanced_cpu_autocast_context(self) -> None:
        def f(x: torch.Tensor) -> list[torch.Tensor]:
            with torch.autocast("cpu"):
                return [x + 1]

        collect, x = _collect_metadata(f)
        collect(x)

    @parametrize("cache_enabled", [True, False])
    def test_collect_metadata_ignores_autocast_cache_enabled(
        self, cache_enabled: bool
    ) -> None:
        def f(x: torch.Tensor) -> list[torch.Tensor]:
            torch.set_autocast_cache_enabled(not cache_enabled)
            return [x + 1]

        collect, x = _collect_metadata(f)
        prior_cache_enabled = torch.is_autocast_cache_enabled()
        try:
            torch.set_autocast_cache_enabled(cache_enabled)
            collect(x)
            self.assertEqual(torch.is_autocast_cache_enabled(), cache_enabled)
        finally:
            torch.set_autocast_cache_enabled(prior_cache_enabled)

    def test_collect_metadata_restores_autocast_dtype_after_mutation(self) -> None:
        extra = "mtia"
        prior_dtype = torch.get_autocast_dtype(extra)
        new_dtype = torch.bfloat16 if prior_dtype != torch.bfloat16 else torch.float16

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            torch.set_autocast_dtype(extra, new_dtype)
            return [x + 1]

        collect, x = _collect_metadata(f)
        try:
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                collect(x)
            self.assertEqual(torch.get_autocast_dtype(extra), prior_dtype)
        finally:
            torch.set_autocast_dtype(extra, prior_dtype)

    def test_collect_metadata_detects_unmatched_enter_autocast(self) -> None:
        # Graph-break style residue: enter without exit leaves enabled set.
        prior_enabled = torch.is_autocast_enabled("cpu")
        prior_dtype = torch.get_autocast_dtype("cpu")
        prior_cache = torch.is_autocast_cache_enabled()

        def f(x: torch.Tensor) -> list[torch.Tensor]:
            torch.amp.autocast_mode._enter_autocast("cpu")
            return [x + 1]

        collect, x = _collect_metadata(f)
        try:
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                collect(x)
            self.assertEqual(torch.is_autocast_enabled("cpu"), prior_enabled)
            self.assertEqual(torch.get_autocast_dtype("cpu"), prior_dtype)
        finally:
            # _enter_autocast bumped nesting; nothing in the error path undoes that.
            torch.autocast_decrement_nesting()
            torch.clear_autocast_cache()
            torch.set_autocast_enabled("cpu", prior_enabled)
            torch.set_autocast_dtype("cpu", prior_dtype)
            torch.set_autocast_cache_enabled(prior_cache)

    def test_aot_function_detects_cpu_autocast_enabled_mutation(self) -> None:
        from functorch.compile import aot_function, nop

        prior_enabled = torch.is_autocast_enabled("cpu")
        prior_dtype = torch.get_autocast_dtype("cpu")

        def f(x: torch.Tensor) -> torch.Tensor:
            torch.set_autocast_enabled("cpu", not prior_enabled)
            return x + 1

        try:
            compiled = aot_function(f, nop)
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                compiled(torch.randn(2))
            self.assertEqual(torch.is_autocast_enabled("cpu"), prior_enabled)
            self.assertEqual(torch.get_autocast_dtype("cpu"), prior_dtype)
        finally:
            torch.set_autocast_enabled("cpu", prior_enabled)
            torch.set_autocast_dtype("cpu", prior_dtype)

    def test_aot_eager_detects_cpu_autocast_enabled_mutation(self) -> None:
        prior_enabled = torch.is_autocast_enabled("cpu")
        prior_dtype = torch.get_autocast_dtype("cpu")

        def f(x: torch.Tensor) -> torch.Tensor:
            torch.set_autocast_enabled("cpu", not prior_enabled)
            return x + 1

        try:
            compiled = torch.compile(f, backend="aot_eager", fullgraph=True)
            with self.assertRaisesRegex(RuntimeError, "mutate the autocast state"):
                compiled(torch.randn(2))
            self.assertEqual(torch.is_autocast_enabled("cpu"), prior_enabled)
            self.assertEqual(torch.get_autocast_dtype("cpu"), prior_dtype)
        finally:
            torch.set_autocast_enabled("cpu", prior_enabled)
            torch.set_autocast_dtype("cpu", prior_dtype)

    def test_aot_function_allows_balanced_cpu_autocast(self) -> None:
        from functorch.compile import aot_function, nop

        def f(x: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cpu"):
                return x + 1

        compiled = aot_function(f, nop)
        out = compiled(torch.randn(2))
        self.assertEqual(out.shape, (2,))


instantiate_parametrized_tests(TestAOTAutocastStates)


if __name__ == "__main__":
    run_tests()
