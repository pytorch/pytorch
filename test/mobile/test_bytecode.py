from torch.jit.mobile import _get_model_bytecode_version, _get_model_ops_and_info
from torch.testing._internal.common_utils import TestCase, run_tests
from pathlib import Path

pytorch_test_dri = Path(__file__).resolve().parents[1]

# script_module_v4.ptl and script_module_v5.ptl source code
# class TestModule(torch.nn.Module):
#     def __init__(self, v):
#         super().__init__()
#         self.x = v

#     def forward(self, y: int):
#         increment = torch.ones([2, 4], dtype=torch.float64)
#         return self.x + y + increment

# output_model_path = pathlib.Path(tmpdirname, "script_module_v5.ptl")
# script_module = torch.jit.script(TestModule(1))
# optimized_scripted_module = optimize_for_mobile(script_module)
# exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter(
#   str(output_model_path))

class testVariousModelVersions(TestCase):
    def test_get_model_bytecode_version(self):
        script_module_v4 = pytorch_test_dri / "cpp" / "jit" / "script_module_v4.ptl"
        version_v4 = _get_model_bytecode_version(script_module_v4)
        assert(version_v4 == 4)

    def test_get_model_ops_and_info(self):
        script_module_v6 = pytorch_test_dri / "cpp" / "jit" / "script_module_v6.ptl"
        ops_v6 = _get_model_ops_and_info(script_module_v6)
        assert(ops_v6["aten::linear"].num_schema_args == 3)

if __name__ == '__main__':
    run_tests()
