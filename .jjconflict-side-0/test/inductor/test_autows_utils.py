# Owner(s): ["module: inductor"]
from torch._inductor.autows_utils import has_meta_ws, meta_ws_enabled
from torch._inductor.test_case import run_tests, TestCase


class AutoWSUtilsTests(TestCase):
    def test_meta_ws_enabled_gating(self) -> None:
        if not has_meta_ws():
            # Without fb-Triton the gate is off regardless of the knob.
            self.assertFalse(meta_ws_enabled())
            return
        from triton import knobs

        with knobs.nvidia.scope():
            knobs.nvidia.use_meta_ws = True
            self.assertTrue(meta_ws_enabled())
            knobs.nvidia.use_meta_ws = False
            self.assertFalse(meta_ws_enabled())


if __name__ == "__main__":
    run_tests()
