import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import io

from torch.jit.mobile import _load_for_lite_interpreter, _backport_for_lite_interpreter
from torch.testing._internal.common_utils import TestCase, run_tests
import pathlib
import tempfile
import torch.utils.show_pickle
import shutil
import fnmatch

class testVariousModelVersions(TestCase):

    def test_save_load_model_v5(self):
        class TestModule(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.x = v

            def forward(self, y: int):
                increment = torch.ones([2, 4], dtype=torch.float64)
                return self.x + y + increment

        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save model v5
            # output_model_path = tmp
            # print("output_model_path: ", output_model_path)
            output_model_path = pathlib.Path(tmpdirname, "script_module_v5.ptl")
            script_module = torch.jit.script(TestModule(1))
            optimized_scripted_module = optimize_for_mobile(script_module)
            exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter(str(output_model_path))
            buf = io.StringIO()
            torch.utils.show_pickle.main(["", tmpdirname + "/" + output_model_path.name + "@*/bytecode.pkl"], output_stream=buf)
            output = buf.getvalue()

            expected_result = '''
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

    def test_backport_to_v4(self):
        class TestModule(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.x = v

            def forward(self, y: int):
                increment = torch.ones([2, 4], dtype=torch.float64)
                return self.x + y + increment

        # Save model v4
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save model v5
            output_v5_model_path = pathlib.Path(tmpdirname, "script_module_v5.ptl")
            output_v4_model_path = pathlib.Path(tmpdirname, "script_module_v4.ptl")
            script_module = torch.jit.script(TestModule(1))
            optimized_scripted_module = optimize_for_mobile(script_module)
            exported_optimized_scripted_module = \
                optimized_scripted_module._save_for_lite_interpreter(str(output_v5_model_path))

            # Backport bytecode v5 to v4
            _backport_for_lite_interpreter(str(output_v5_model_path), str(output_v4_model_path))
            buf = io.StringIO()
            torch.utils.show_pickle.main(["", tmpdirname + "/" + output_v4_model_path.name + "@*/bytecode.pkl"], output_stream=buf)
            output = buf.getvalue()

            expected_result_v4 = '''
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
            acutal_result_clean = "".join(output.split())
            expected_result_clean = "".join(expected_result_v4.split())
            isMatch = fnmatch.fnmatch(acutal_result_clean, expected_result_clean)

            assert(isMatch)

            # Load model v4 and run forward method
            mobile_module = _load_for_lite_interpreter(str(output_v4_model_path))
            module_input = 1
            mobile_module_result = mobile_module(module_input)
            expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
            torch.testing.assert_allclose(mobile_module_result, expected_mobile_module_result)
            shutil.rmtree(tmpdirname)

if __name__ == '__main__':
    run_tests()
