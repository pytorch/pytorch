import io
import os
import shutil
import traceback

import onnx
import onnx_test_common
from onnx import numpy_helper
from test_nn import new_module_tests

import torch
from torch.autograd import Variable
from torch.testing._internal.common_nn import module_tests


# Take a test case (a dict) as input, return the test name.
def get_test_name(testcase):
    if "fullname" in testcase:
        return "test_" + testcase["fullname"]

    test_name = "test_" + testcase["constructor"].__name__
    if "desc" in testcase:
        test_name += "_" + testcase["desc"]
    return test_name


# Take a test case (a dict) as input, return the input for the module.
def gen_input(testcase):
    if "input_size" in testcase:
        if (
            testcase["input_size"] == ()
            and "desc" in testcase
            and testcase["desc"][-6:] == "scalar"
        ):
            testcase["input_size"] = (1,)
        return Variable(torch.randn(*testcase["input_size"]))
    elif "input_fn" in testcase:
        input = testcase["input_fn"]()
        if isinstance(input, Variable):
            return input
        return Variable(testcase["input_fn"]())


def gen_module(testcase):
    if "constructor_args" in testcase:
        args = testcase["constructor_args"]
        module = testcase["constructor"](*args)
        module.train(False)
        return module
    module = testcase["constructor"]()
    module.train(False)
    return module


def print_stats(FunctionalModule_nums, nn_module):
    print(f"{FunctionalModule_nums} functional modules detected.")
    supported = []
    unsupported = []
    not_fully_supported = []
    for key, value in nn_module.items():
        if value == 1:
            supported.append(key)
        elif value == 2:
            unsupported.append(key)
        elif value == 3:
            not_fully_supported.append(key)

    def fun(info, l):
        print(info)
        for v in l:
            print(v)

    # Fully Supported Ops: All related test cases of these ops have been exported
    # Semi-Supported Ops: Part of related test cases of these ops have been exported
    # Unsupported Ops: None of related test cases of these ops have been exported
    for info, l in [
        [f"{len(supported)} Fully Supported Operators:", supported],
        [
            f"{len(not_fully_supported)} Semi-Supported Operators:",
            not_fully_supported,
        ],
        [f"{len(unsupported)} Unsupported Operators:", unsupported],
    ]:
        fun(info, l)


def convert_tests(testcases, sets=1):
    print(f"Collect {len(testcases)} test cases from PyTorch.")
    failed = 0
    FunctionalModule_nums = 0
    nn_module = {}
    for t in testcases:
        test_name = get_test_name(t)
        module = gen_module(t)
        module_name = str(module).split("(")[0]
        if module_name == "FunctionalModule":
            FunctionalModule_nums += 1
        else:
            if module_name not in nn_module:
                nn_module[module_name] = 0
        try:
            input = gen_input(t)
            f = io.BytesIO()
            torch.onnx._export(
                module,
                input,
                f,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )
            onnx_model = onnx.load_from_string(f.getvalue())
            onnx.checker.check_model(onnx_model)
            onnx.helper.strip_doc_string(onnx_model)
            output_dir = os.path.join(onnx_test_common.pytorch_converted_dir, test_name)

            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
            with open(os.path.join(output_dir, "model.onnx"), "wb") as file:
                file.write(onnx_model.SerializeToString())

            for i in range(sets):
                output = module(input)
                data_dir = os.path.join(output_dir, f"test_data_set_{i}")
                os.makedirs(data_dir)

                for index, var in enumerate([input]):
                    tensor = numpy_helper.from_array(var.data.numpy())
                    with open(
                        os.path.join(data_dir, f"input_{index}.pb"), "wb"
                    ) as file:
                        file.write(tensor.SerializeToString())
                for index, var in enumerate([output]):
                    tensor = numpy_helper.from_array(var.data.numpy())
                    with open(
                        os.path.join(data_dir, f"output_{index}.pb"), "wb"
                    ) as file:
                        file.write(tensor.SerializeToString())
                input = gen_input(t)
                if module_name != "FunctionalModule":
                    nn_module[module_name] |= 1
        except:  # noqa: E722,B001
            traceback.print_exc()
            if module_name != "FunctionalModule":
                nn_module[module_name] |= 2
            failed += 1

    print(
        f"Collect {len(testcases)} test cases from PyTorch repo, failed to export {failed} cases."
    )
    print(
        f"PyTorch converted cases are stored in {onnx_test_common.pytorch_converted_dir}."
    )
    print_stats(FunctionalModule_nums, nn_module)


if __name__ == "__main__":
    testcases = module_tests + new_module_tests
    convert_tests(testcases)
