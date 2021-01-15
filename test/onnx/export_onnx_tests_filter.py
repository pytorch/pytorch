import argparse
import glob
import onnx.backend.test
import os
import shutil
from test_caffe2_common import run_generated_test
import google.protobuf.text_format
import test_onnx_common
import traceback

_fail_test_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "fail", "generated")


_expect_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "expect")


def collect_generated_testcases(root_dir=test_onnx_common.pytorch_converted_dir,
                                verbose=False, fail_dir=None, expect=True):
    total_pass = 0
    total_fail = 0
    for d in os.listdir(root_dir):
        dir_name = os.path.join(root_dir, d)
        if os.path.isdir(dir_name):
            failed = False
            try:
                model_file = os.path.join(dir_name, "model.onnx")
                data_dir_pattern = os.path.join(dir_name, "test_data_set_*")
                for data_dir in glob.glob(data_dir_pattern):
                    for device in torch.testing.get_all_device_types():
                        run_generated_test(model_file, data_dir, device)
                if expect:
                    expect_file = os.path.join(_expect_dir,
                                               "PyTorch-generated-{}.expect".format(d))
                    with open(expect_file, "w") as text_file:
                        model = onnx.load(model_file)
                        onnx.checker.check_model(model)
                        onnx.helper.strip_doc_string(model)
                        text_file.write(google.protobuf.text_format.MessageToString(model))
                total_pass += 1
            except Exception as e:
                if verbose:
                    print("The test case in {} failed!".format(dir_name))
                    traceback.print_exc()
                if fail_dir is None:
                    shutil.rmtree(dir_name)
                else:
                    target_dir = os.path.join(fail_dir, d)
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    shutil.move(dir_name, target_dir)
                total_fail += 1
    print("Successfully generated/updated {} test cases from PyTorch.".format(total_pass))
    if expect:
        print("Expected pbtxt files are generated in {}.".format(_expect_dir))
    print("Failed {} testcases are moved to {}.".format(total_fail, _fail_test_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check and filter the failed test cases.')
    parser.add_argument('-v', action="store_true", default=False, help="verbose")
    parser.add_argument('--delete', action="store_true", default=False, help="delete failed test cases")
    parser.add_argument('--no-expect', action="store_true", default=False, help="generate expect txt files")
    args = parser.parse_args()
    verbose = args.v
    delete = args.delete
    expect = not args.no_expect
    fail_dir = _fail_test_dir
    if delete:
        fail_dir = None
    if fail_dir:
        if not os.path.exists(fail_dir):
            os.makedirs(fail_dir)

    collect_generated_testcases(verbose=verbose, fail_dir=fail_dir, expect=expect)
    # We already generate the expect files for test_operators.py.
    collect_generated_testcases(root_dir=test_onnx_common.pytorch_operator_dir,
                                verbose=verbose, fail_dir=fail_dir, expect=False)
