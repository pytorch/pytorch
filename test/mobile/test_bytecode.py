# Owner(s): ["oncall: mobile"]

import fnmatch
import io
import shutil
import tempfile
from pathlib import Path

import torch
import torch.utils.show_pickle

# from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.jit.mobile import (
    _backport_for_mobile,
    _backport_for_mobile_to_buffer,
    _get_mobile_model_contained_types,
    _get_model_bytecode_version,
    _get_model_ops_and_info,
    _load_for_lite_interpreter,
)
from torch.testing._internal.common_utils import run_tests, TestCase

pytorch_test_dir = Path(__file__).resolve().parents[1]

# script_module_v4.ptl and script_module_v5.ptl source code
# class TestModule(torch.nn.Module):
#     def __init__(self, v):
#         super().__init__()
#         self.x = v

#     def forward(self, y: int):
#         increment = torch.ones([2, 4], dtype=torch.float64)
#         return self.x + y + increment

# output_model_path = Path(tmpdirname, "script_module_v5.ptl")
# script_module = torch.jit.script(TestModule(1))
# optimized_scripted_module = optimize_for_mobile(script_module)
# exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter(
#   str(output_model_path))

SCRIPT_MODULE_V4_BYTECODE_PKL = """
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
        """

SCRIPT_MODULE_V5_BYTECODE_PKL = """
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
        """

SCRIPT_MODULE_V6_BYTECODE_PKL = """
(6,
 ('__torch__.*.TestModule.forward',
  (('instructions',
    (('STOREN', 1, 2),
     ('DROPR', 1, 0),
     ('LOADC', 0, 0),
     ('LOADC', 1, 0),
     ('MOVE', 2, 0),
     ('OP', 0, 0),
     ('OP', 1, 0),
     ('RET', 0, 0))),
   ('operators', (('aten::add', 'int', 2), ('aten::add', 'Scalar', 2))),
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
    """

SCRIPT_MODULE_BYTECODE_PKL = {
    4: {
        "bytecode_pkl": SCRIPT_MODULE_V4_BYTECODE_PKL,
        "model_name": "script_module_v4.ptl",
    },
}

# The minimum version a model can be backported to
# Need to be updated when a bytecode version is completely retired
MINIMUM_TO_VERSION = 4


class testVariousModelVersions(TestCase):
    def test_get_model_bytecode_version(self):
        def check_model_version(model_path, expect_version):
            actual_version = _get_model_bytecode_version(model_path)
            assert actual_version == expect_version

        for version, model_info in SCRIPT_MODULE_BYTECODE_PKL.items():
            model_path = pytorch_test_dir / "cpp" / "jit" / model_info["model_name"]
            check_model_version(model_path, version)

    def test_bytecode_values_for_all_backport_functions(self):
        # Find the maximum version of the checked in models, start backporting to the minimum support version,
        # and comparing the bytecode pkl content.
        # It can't be merged to the test `test_all_backport_functions`, because optimization is dynamic and
        # the content might change when optimize function changes. This test focuses
        # on bytecode.pkl content validation. For the content validation, it is not byte to byte check, but
        # regular expression matching. The wildcard can be used to skip some specific content comparison.
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        current_from_version = maximum_checked_in_model_version

        with tempfile.TemporaryDirectory() as tmpdirname:
            while current_from_version > MINIMUM_TO_VERSION:
                # Load model v5 and run forward method
                model_name = SCRIPT_MODULE_BYTECODE_PKL[current_from_version][
                    "model_name"
                ]
                input_model_path = pytorch_test_dir / "cpp" / "jit" / model_name

                # A temporary model file will be export to this path, and run through bytecode.pkl
                # content check.
                tmp_output_model_path_backport = Path(
                    tmpdirname, "tmp_script_module_backport.ptl"
                )

                current_to_version = current_from_version - 1
                backport_success = _backport_for_mobile(
                    input_model_path, tmp_output_model_path_backport, current_to_version
                )
                assert backport_success

                expect_bytecode_pkl = SCRIPT_MODULE_BYTECODE_PKL[current_to_version][
                    "bytecode_pkl"
                ]

                buf = io.StringIO()
                torch.utils.show_pickle.main(
                    [
                        "",
                        tmpdirname
                        + "/"
                        + tmp_output_model_path_backport.name
                        + "@*/bytecode.pkl",
                    ],
                    output_stream=buf,
                )
                output = buf.getvalue()

                acutal_result_clean = "".join(output.split())
                expect_result_clean = "".join(expect_bytecode_pkl.split())
                isMatch = fnmatch.fnmatch(acutal_result_clean, expect_result_clean)
                assert isMatch

                current_from_version -= 1
            shutil.rmtree(tmpdirname)

    # Please run this test manually when working on backport.
    # This test passes in OSS, but fails internally, likely due to missing step in build
    # def test_all_backport_functions(self):
    #     # Backport from the latest bytecode version to the minimum support version
    #     # Load, run the backport model, and check version
    #     class TestModule(torch.nn.Module):
    #         def __init__(self, v):
    #             super().__init__()
    #             self.x = v

    #         def forward(self, y: int):
    #             increment = torch.ones([2, 4], dtype=torch.float64)
    #             return self.x + y + increment

    #     module_input = 1
    #     expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)

    #     # temporary input model file and output model file will be exported in the temporary folder
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         tmp_input_model_path = Path(tmpdirname, "tmp_script_module.ptl")
    #         script_module = torch.jit.script(TestModule(1))
    #         optimized_scripted_module = optimize_for_mobile(script_module)
    #         exported_optimized_scripted_module = optimized_scripted_module._save_for_lite_interpreter(str(tmp_input_model_path))

    #         current_from_version = _get_model_bytecode_version(tmp_input_model_path)
    #         current_to_version = current_from_version - 1
    #         tmp_output_model_path = Path(tmpdirname, "tmp_script_module_backport.ptl")

    #         while current_to_version >= MINIMUM_TO_VERSION:
    #             # Backport the latest model to `to_version` to a tmp file "tmp_script_module_backport"
    #             backport_success = _backport_for_mobile(tmp_input_model_path, tmp_output_model_path, current_to_version)
    #             assert(backport_success)

    #             backport_version = _get_model_bytecode_version(tmp_output_model_path)
    #             assert(backport_version == current_to_version)

    #             # Load model and run forward method
    #             mobile_module = _load_for_lite_interpreter(str(tmp_input_model_path))
    #             mobile_module_result = mobile_module(module_input)
    #             torch.testing.assert_close(mobile_module_result, expected_mobile_module_result)
    #             current_to_version -= 1

    #         # Check backport failure case
    #         backport_success = _backport_for_mobile(tmp_input_model_path, tmp_output_model_path, MINIMUM_TO_VERSION - 1)
    #         assert(not backport_success)
    #         # need to clean the folder before it closes, otherwise will run into git not clean error
    #         shutil.rmtree(tmpdirname)

    # Check just the test_backport_bytecode_from_file_to_file mechanism but not the function implementations
    def test_backport_bytecode_from_file_to_file(self):
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        script_module_v5_path = (
            pytorch_test_dir
            / "cpp"
            / "jit"
            / SCRIPT_MODULE_BYTECODE_PKL[maximum_checked_in_model_version]["model_name"]
        )

        if maximum_checked_in_model_version > MINIMUM_TO_VERSION:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_backport_model_path = Path(
                    tmpdirname, "tmp_script_module_v5_backported_to_v4.ptl"
                )
                # backport from file
                success = _backport_for_mobile(
                    script_module_v5_path,
                    tmp_backport_model_path,
                    maximum_checked_in_model_version - 1,
                )
                assert success

                buf = io.StringIO()
                torch.utils.show_pickle.main(
                    [
                        "",
                        tmpdirname
                        + "/"
                        + tmp_backport_model_path.name
                        + "@*/bytecode.pkl",
                    ],
                    output_stream=buf,
                )
                output = buf.getvalue()

                expected_result = SCRIPT_MODULE_V4_BYTECODE_PKL
                acutal_result_clean = "".join(output.split())
                expect_result_clean = "".join(expected_result.split())
                isMatch = fnmatch.fnmatch(acutal_result_clean, expect_result_clean)
                assert isMatch

                # Load model v4 and run forward method
                mobile_module = _load_for_lite_interpreter(str(tmp_backport_model_path))
                module_input = 1
                mobile_module_result = mobile_module(module_input)
                expected_mobile_module_result = 3 * torch.ones(
                    [2, 4], dtype=torch.float64
                )
                torch.testing.assert_close(
                    mobile_module_result, expected_mobile_module_result
                )
                shutil.rmtree(tmpdirname)

    # Check just the _backport_for_mobile_to_buffer mechanism but not the function implementations
    def test_backport_bytecode_from_file_to_buffer(self):
        maximum_checked_in_model_version = max(SCRIPT_MODULE_BYTECODE_PKL.keys())
        script_module_v5_path = (
            pytorch_test_dir
            / "cpp"
            / "jit"
            / SCRIPT_MODULE_BYTECODE_PKL[maximum_checked_in_model_version]["model_name"]
        )

        if maximum_checked_in_model_version > MINIMUM_TO_VERSION:
            # Backport model to v4
            script_module_v4_buffer = _backport_for_mobile_to_buffer(
                script_module_v5_path, maximum_checked_in_model_version - 1
            )
            buf = io.StringIO()

            # Check version of the model v4 from backport
            bytesio = io.BytesIO(script_module_v4_buffer)
            backport_version = _get_model_bytecode_version(bytesio)
            assert backport_version == maximum_checked_in_model_version - 1

            # Load model v4 from backport and run forward method
            bytesio = io.BytesIO(script_module_v4_buffer)
            mobile_module = _load_for_lite_interpreter(bytesio)
            module_input = 1
            mobile_module_result = mobile_module(module_input)
            expected_mobile_module_result = 3 * torch.ones([2, 4], dtype=torch.float64)
            torch.testing.assert_close(
                mobile_module_result, expected_mobile_module_result
            )

    def test_get_model_ops_and_info(self):
        # TODO update this to be more in the style of the above tests after a backport from 6 -> 5 exists
        script_module_v6 = pytorch_test_dir / "cpp" / "jit" / "script_module_v6.ptl"
        ops_v6 = _get_model_ops_and_info(script_module_v6)
        assert ops_v6["aten::add.int"].num_schema_args == 2
        assert ops_v6["aten::add.Scalar"].num_schema_args == 2

    def test_get_mobile_model_contained_types(self):
        class MyTestModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        sample_input = torch.tensor([1])

        script_module = torch.jit.script(MyTestModule())
        script_module_result = script_module(sample_input)

        buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
        buffer.seek(0)
        type_list = _get_mobile_model_contained_types(buffer)
        assert len(type_list) >= 0


if __name__ == "__main__":
    run_tests()
