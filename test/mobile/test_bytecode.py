import torch
import io

from torch.jit.mobile import _load_for_lite_interpreter, _get_bytecode_version, _backport_for_mobile_to_buffer, _backport_for_mobile
from torch.testing._internal.common_utils import TestCase, run_tests
import pathlib
import tempfile
import torch.utils.show_pickle
import shutil
import fnmatch
from pathlib import Path

pytorch_test_dri = Path(__file__).resolve().parents[1]

# Script Module python source code
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

SCRIPT_MODULE_V4_BYTECODE_PKL = '''
(4,
 ('__torch__.*.TestModule.forward',
  (('instructions',
    (('STOREN', 1, 2),
     ('DROPR', 1, 0),
     ('LOADC', 0, 0),
     ('LOADC', 1, 0),
     ('MOVE', 2, 0),
     ('OP', 0, 0),
     ('LOADC', 1, 0),
     ('OP', 1, 0),
     ('RET', 0, 0))),
   ('operators', (('aten::add', 'int'), ('aten::add', 'Scalar'))),
   ('constants',
    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, '0', 'cpu', 8),),
       0,
       (2, 4),
       (4, 1),
       False,
       collections.OrderedDict()),
     1)),
   ('types', ()),
   ('register_size', 2)),
  (('arguments',
    ((('name', 'self'),
      ('type', '__torch__.*.TestModule'),
      ('default_value', None)),
     (('name', 'y'), ('type', 'int'), ('default_value', None)))),
   ('returns',
    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))
        '''

SCRIPT_MODULE_V5_BYTECODE_PKL = '''
(5,
 ('__torch__.*.TestModule.forward',
  (('instructions',
    (('STOREN', 1, 2),
     ('DROPR', 1, 0),
     ('LOADC', 0, 0),
     ('LOADC', 1, 0),
     ('MOVE', 2, 0),
     ('OP', 0, 0),
     ('LOADC', 1, 0),
     ('OP', 1, 0),
     ('RET', 0, 0))),
   ('operators', (('aten::add', 'int'), ('aten::add', 'Scalar'))),
   ('constants',
    (torch._utils._rebuild_tensor_v2(pers.obj(('storage', torch.DoubleStorage, 'constants/0', 'cpu', 8),),
       0,
       (2, 4),
       (4, 1),
       False,
       collections.OrderedDict()),
     1)),
   ('types', ()),
   ('register_size', 2)),
  (('arguments',
    ((('name', 'self'),
      ('type', '__torch__.*.TestModule'),
      ('default_value', None)),
     (('name', 'y'), ('type', 'int'), ('default_value', None)))),
   ('returns',
    ((('name', ''), ('type', 'Tensor'), ('default_value', None)),)))))
        '''

class testVariousModelVersions(TestCase):
    def test_get_bytecode_version(self):
        script_module_v4 = pytorch_test_dri / "cpp" / "jit" / "script_module_v4.ptl"
        script_module_v5 = pytorch_test_dri / "cpp" / "jit" / "script_module_v5.ptl"

        version_v4 = _get_bytecode_version(str(script_module_v4))
        version_v5 = _get_bytecode_version(str(script_module_v5))

        assert(version_v4 == 4)
        assert(version_v5 == 5)

    def test_backport_bytecode_from_file_to_file(self):
        script_module_v5 = pytorch_test_dri / "cpp" / "jit" / "script_module_v5.ptl"

        with tempfile.TemporaryDirectory() as tmpdirname:
            backport_model_path = pathlib.Path(tmpdirname, "backport_script_module_v5.ptl")
            _backport_for_mobile(str(script_module_v5), str(backport_model_path))
            buf = io.StringIO()
            torch.utils.show_pickle.main(["", tmpdirname + "/" + backport_model_path.name + "@*/bytecode.pkl"], output_stream=buf)
            output = buf.getvalue()

            expected_result = SCRIPT_MODULE_V4_BYTECODE_PKL
            acutal_result_clean = "".join(output.split())
            expect_result_clean = "".join(expected_result.split())
            isMatch = fnmatch.fnmatch(acutal_result_clean, expect_result_clean)
            assert(isMatch)

            # Load model v5 and run forward method
            mobile_module = _load_for_lite_interpreter(str(backport_model_path))
            module_input = 1
            mobile_module_result = mobile_module(module_input)
            expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
            torch.testing.assert_allclose(mobile_module_result, expected_mobile_module_result)
            shutil.rmtree(tmpdirname)

    def test_backport_bytecode_from_file_to_buffer(self):
        script_module_v5 = pytorch_test_dri / "cpp" / "jit" / "script_module_v5.ptl"

        # Backport model to v4
        buffer = _backport_for_mobile_to_buffer(str(script_module_v5))
        buf = io.StringIO()
        bytesio = io.BytesIO(buffer)
        backport_version = _get_bytecode_version(bytesio)
        assert(backport_version == 4)


    def test_run_backport_bytecode_from_file_to_buffer(self):
        script_module_v5 = pytorch_test_dri / "cpp" / "jit" / "script_module_v5.ptl"

        # Backport model to v4
        buffer = _backport_for_mobile_to_buffer(str(script_module_v5))
        bytesio = io.BytesIO(buffer)

        # Load model v4 from backport and run forward method
        mobile_module = _load_for_lite_interpreter(bytesio)
        module_input = 1
        mobile_module_result = mobile_module(module_input)
        expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
        torch.testing.assert_allclose(mobile_module_result, expected_mobile_module_result)

    def test_load_and_run_model(self):
        script_module_v4 = pytorch_test_dri / "cpp" / "jit" / "script_module_v4.ptl"
        script_module_v5 = pytorch_test_dri / "cpp" / "jit" / "script_module_v5.ptl"

        # Load model v5 and run forward method
        jit_module_v4 = torch.jit.load(str(script_module_v4))
        jit_module_v5 = torch.jit.load(str(script_module_v5))
        mobile_module_v4 = _load_for_lite_interpreter(str(script_module_v4))
        mobile_module_v5 = _load_for_lite_interpreter(str(script_module_v5))

        # mobile_module = _load_for_lite_interpreter(str(model_5))
        module_input = 1
        jit_module_v4_result = jit_module_v4(module_input)
        jit_module_v5_result = jit_module_v5(module_input)
        mobile_module_v4_result = mobile_module_v4(module_input)
        mobile_module_v5_result = mobile_module_v5(module_input)

        expected_result = 3 * torch.ones([2, 4], dtype=torch.float64)

        torch.testing.assert_allclose(jit_module_v4_result, expected_result)
        torch.testing.assert_allclose(jit_module_v5_result, expected_result)
        torch.testing.assert_allclose(mobile_module_v4_result, expected_result)
        torch.testing.assert_allclose(mobile_module_v5_result, expected_result)


    def test_save_load_model_v5(self):
        script_module_v5 = pytorch_test_dri / "cpp" / "jit" / "script_module_v5.ptl"

        # Load model v5 and run forward method
        jit_module_v5 = torch.jit.load(str(script_module_v5))

        with tempfile.TemporaryDirectory() as tmpdirname:
            output_model_path = pathlib.Path(tmpdirname, "resave_script_module_v5.ptl")
            exported_optimized_scripted_module = jit_module_v5._save_for_lite_interpreter(str(output_model_path))
            buf = io.StringIO()
            torch.utils.show_pickle.main(["", tmpdirname + "/" + output_model_path.name + "@*/bytecode.pkl"], output_stream=buf)
            output = buf.getvalue()

            expected_result = SCRIPT_MODULE_V5_BYTECODE_PKL
            acutal_result_clean = "".join(output.split())
            expected_result_clean = "".join(expected_result.split())
            isMatch = fnmatch.fnmatch(acutal_result_clean, expected_result_clean)
            assert(isMatch)

            # Load model v5 and run forward method
            mobile_module = _load_for_lite_interpreter(str(output_model_path))
            module_input = 1
            mobile_module_result = mobile_module(module_input)
            expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
            torch.testing.assert_allclose(mobile_module_result, expected_mobile_module_result)
            shutil.rmtree(tmpdirname)

if __name__ == '__main__':
    run_tests()
