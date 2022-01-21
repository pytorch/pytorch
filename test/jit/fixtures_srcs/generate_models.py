import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import Set

import torch
from test.jit.fixtures_srcs.fixtures_src import TestVersionedDivTensorExampleV4
from torch.jit.mobile import _load_for_lite_interpreter, _export_operator_list

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""
This file is used to generate model for test operator change. Please refer to
https://github.com/pytorch/rfcs/blob/master/RFC-0017-PyTorch-Operator-Versioning.md for more details.

A systematic workflow to change operator is needed to ensure
Backwards Compatibility (BC) / Forwards Compatibility (FC) for operator changes. For BC-breaking operator change,
an upgrader is needed. Here is the flow to properly land a BC-breaking operator change.

1. Write an upgrader in caffe2/torch/csrc/jit/operator_upgraders/upgraders_entry.cpp file. The softly enforced
naming format is <operator_name>_<operator_overload>_<start>_<end>. For example, the below example means that
div.Tensor at version from 0 to 3 needs to be replaced by this upgrader.

```
/*
div_Tensor_0_3 is added for a change of operator div in pr xxxxxxx.
Create date: 12/02/2021
Expire date: 06/02/2022
*/
     {"div_Tensor_0_3", R"SCRIPT(
def div_Tensor_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT"},
```

2. In caffe2/torch/csrc/jit/operator_upgraders/version_map.h, add changes like below.
You will need to make sure that the entry is SORTED according to the version bump number.
```
    {"div.Tensor",
      {{4,
        "div_Tensor_0_3",
        "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},
```

3. After rebuild PyTorch, run the following command and it will auto generate a change to
fbcode/caffe2/torch/csrc/jit/mobile/upgrader_mobile.cpp

```
python pytorch/tools/codegen/operator_versions/gen_mobile_upgraders.py
```

4. Generate the test to cover upgrader.

4.1 Switch the commit before the operator change, and add a module in
`test/jit/fixtures_srcs/fixtures_src.py`. The reason why switching to commit is that,
an old model with the old operator before the change is needed to ensure the upgrader
is working as expected. In `test/jit/fixtures_srcs/generate_models.py`, add the module and
it's corresponding changed operator like following
```
ALL_MODULES = {
    TestVersionedDivTensorExampleV4(): "aten::div.Tensor",
}
```
This module should includes the changed operator. If the operator isn't covered in the model,
the model export process in step 4.2 will fail.

4.2 Export the model to `test/jit/fixtures` by running
```
python /Users/chenlai/pytorch/test/jit/fixtures_src/generate_models.py
```

4.3 In `test/jit/test_save_load_for_op_version.py`, add a test to cover the old models and
ensure the result is equivalent between current module and old module + upgrader.

4.4 Save all change in 4.1, 4.2 and 4.3, as well as previous changes made in step 1, 2, 3.
Submit a pr

"""

"""
A map of test modules and it's according changed operator
key: test module
value: changed operator
"""
ALL_MODULES = {
    TestVersionedDivTensorExampleV4(): "aten::div.Tensor",
}

"""
Get the path to `test/jit/fixtures`, where all test models for operator changes
(upgrader/downgrader) are stored
"""
def get_fixtures_path() -> Path:
    pytorch_dir = Path(__file__).resolve().parents[3]
    fixtures_path = pytorch_dir / "test" / "jit" / "fixtures"
    return fixtures_path

"""
Get all models' name in `test/jit/fixtures`
"""
def get_all_models() -> Set[str]:
    files_in_fixtures = get_fixtures_path().glob('**/*')
    all_models_from_fixtures = [fixture.stem for fixture in files_in_fixtures if fixture.is_file()]
    return set(all_models_from_fixtures)

"""
Check if a given model already exist in `test/jit/fixtures`
"""
def model_exist(model_file_name: str, all_models: Set[str]) -> bool:
    return model_file_name in all_models

"""
Get the operator list given a module
"""
def get_operator_list(script_module: torch) -> Set[str]:
    buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
    buffer.seek(0)
    mobile_module = _load_for_lite_interpreter(buffer)
    operator_list = _export_operator_list(mobile_module)
    return operator_list

"""
Get the output model operator version, given a module
"""
def get_output_model_version(script_module: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.jit.save(script_module, buffer)
    buffer.seek(0)
    zipped_model = zipfile.ZipFile(buffer)
    version = int(zipped_model.read('archive/version').decode("utf-8"))
    return version

"""
Loop through all test modules. If the corresponding model doesn't exist in
`test/jit/fixtures`, generate one. For the following reason, a model won't be exported:

1. The test module doens't cover the changed operator. For example, test_versioned_div_tensor_example_v4
is supposed to test the operator aten::div.Tensor. If the model doesn't include this operator, it will fail.
The error message includes the actual operator list from the model.

2. The output model version is not the same as expected version. For example, test_versioned_div_tensor_example_v4
is used to test an operator change aten::div.Tensor, and the operator version will be bumped to v5. This script is
supposed to run before the operator change (before the commit to make the change). If the actual model version is v5,
likely this script is running with the commit to make the change.

3. The model already exists in `test/jit/fixtures`.

"""
def generate_models():
    all_models = get_all_models()
    for a_module, expect_operator in ALL_MODULES.items():
        print(a_module, expect_operator)
        script_module = torch.jit.script(a_module)
        model_version = get_output_model_version(script_module)

        # For example: TestVersionedDivTensorExampleV4
        torch_module_name = type(a_module).__name__

        # The corresponding model name is: test_versioned_div_tensor_example_v4
        model_name = ''.join([
            '_' + char.lower() if char.isupper() else char for char in torch_module_name
        ]).lstrip('_')

        logger.info(f"Processing {torch_module_name}")
        if model_exist(model_name, all_models):
            logger.info(f"Model {model_name} already exists, skipping")
            continue

        actual_model_version = "v" + str(model_version)
        expect_model_version = model_name.split("_")[-1]
        if actual_model_version != expect_model_version:
            logger.error(
                f"Actual model version {actual_model_version} "
                f"doesn't match the expect model version {expect_model_version}. "
                f"Please run the script before the commit to change operator.")

        actual_operator_list = get_operator_list(script_module)
        if expect_operator not in actual_operator_list:
            logger.error(
                f"The model includes operator: {actual_operator_list}, "
                f"however it doesn't cover the operator {expect_operator}."
                f"Please ensure the output model includes the tested operator.")

        export_model_path = str(get_fixtures_path() / (str(model_name) + ".ptl"))
        script_module._save_for_lite_interpreter(export_model_path)
        logger.info(f"Generating model {model_name} and it's save to {export_model_path}")

def main() -> None:
    generate_models()

if __name__ == '__main__':
    main()
