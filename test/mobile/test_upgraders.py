# Owner(s): ["oncall: mobile"]

import io
from itertools import product
from pathlib import Path

import torch
import torch.utils.bundled_inputs
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_utils import run_tests, TestCase


pytorch_test_dir = Path(__file__).resolve().parents[1]


class TestLiteScriptModule(TestCase):
    def _save_load_mobile_module(self, script_module: torch.jit.ScriptModule):
        buffer = io.BytesIO(
            script_module._save_to_buffer_for_lite_interpreter(
                _save_mobile_debug_info=True
            )
        )
        buffer.seek(0)
        mobile_module = _load_for_lite_interpreter(buffer)
        return mobile_module

    def _try_fn(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

    def test_versioned_div_tensor(self):
        # noqa: F841
        def div_tensor_0_3(self, other):  # noqa: F841
            if self.is_floating_point() or other.is_floating_point():
                return self.true_divide(other)
            return self.divide(other, rounding_mode="trunc")

        model_path = (
            pytorch_test_dir
            / "cpp"
            / "jit"
            / "upgrader_models"
            / "test_versioned_div_tensor_v2.ptl"
        )
        _load_for_lite_interpreter(str(model_path))
        jit_module_v2 = torch.jit.load(str(model_path))
        self._save_load_mobile_module(jit_module_v2)
        vals = (2.0, 3.0, 2, 3)
        for val_a, val_b in product(vals, vals):
            a = torch.tensor((val_a,))
            b = torch.tensor((val_b,))

            def _helper(m, fn):
                m_results = self._try_fn(m, a, b)
                fn_result = self._try_fn(fn, a, b)

                if isinstance(m_results, Exception):
                    self.assertTrue(isinstance(fn_result, Exception))
                else:
                    for result in m_results:
                        print("result: ", result)
                        print("fn_result: ", fn_result)
                        print(result == fn_result)
                        self.assertTrue(result.eq(fn_result))
                        # self.assertEqual(result, fn_result)

            # old operator should produce the same result as applying upgrader of torch.div op
            # _helper(mobile_module_v2, div_tensor_0_3)
            # latest operator should produce the same result as applying torch.div op
            # _helper(current_mobile_module, torch.div)


if __name__ == "__main__":
    run_tests()
