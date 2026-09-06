# Owner(s): ["module: dsl-native-ops"]

from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCuTeDSLTopKEligibility(TestCase):
    def test_pre_sm100_devices_are_ineligible(self) -> None:
        from torch._native.ops.topk import cutedsl_impl

        x = mock.Mock(
            is_cuda=True,
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            shape=(256, 256),
            ndim=2,
        )
        with (
            mock.patch.object(cutedsl_impl, "any_cow", return_value=False),
            mock.patch.object(cutedsl_impl, "last_dim_row_major_ok", return_value=True),
            mock.patch.object(cutedsl_impl, "_min_rows_for_full_wave", return_value=1),
            mock.patch.object(torch.cuda, "get_device_capability") as get_capability,
        ):
            for capability, expected in (
                ((8, 0), False),
                ((9, 0), False),
                ((10, 0), True),
            ):
                with self.subTest(capability=capability):
                    cutedsl_impl._sm100_or_above.cache_clear()
                    get_capability.reset_mock()
                    get_capability.return_value = capability
                    for _ in range(2):
                        self.assertEqual(
                            cutedsl_impl._eligible(x, 32, -1, True, True), expected
                        )
                    get_capability.assert_called_once_with(0)


if __name__ == "__main__":
    run_tests()
