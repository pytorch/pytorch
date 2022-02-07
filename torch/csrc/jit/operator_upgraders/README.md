# Guidance for Operator Developer

After add change for the operator, if it causes BC breakage, an upgrader is needed and following is the process to add an upgrader:


1. Prepare a model before making changes to the operator, following the process:
     1. add a module in `test/jit/fixtures_srcs/fixtures_src.py`. The reason why switching to commit is that, an old model with the old operator before the change is needed to ensure the upgrader is working as expected. In `test/jit/fixtures_srcs/generate_models.py`, add the module and its corresponding changed operator like following

        ```
        ALL_MODULES = {
           TestVersionedDivTensorExampleV7(): "aten::div.Tensor",
        }
        ```

        This module should include the changed operator. If the operator isn't covered in the model, the model export process will fail.

       2. Export the model to `test/jit/fixtures` by running
            ```
            python test/jit/fixtures_src/generate_models.py
            ```

       3. Commit the change and submit a pr

2. Make changes to the operator and write an upgrader.
    1. Make the operator change.
    2. Write an upgrader in `caffe2/torch/csrc/jit/operator_upgraders/upgraders_entry.cpp` file inside a map `kUpgradersEntryMap`. The softly enforced naming format is `&lt;operator_name>_&lt;operator_overload>_&lt;start>_&lt;end>`. For example, the below example means that `linspace.out `and` linspace` at versions from 0 to 7 need to be replaced by these upgraders.

    ```
    static std::unordered_map<std::string, std::string> kUpgradersEntryMap(
        {
          {"linspace_0_7", R"SCRIPT(
    def linspace_0_7(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], *, dtype: Optional[int], layout: Optional[int],
                     device: Optional[Device], pin_memory: Optional[bool]):
      if (steps is None):
        return torch.linspace(start=start, end=end, steps=100, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
      return torch.linspace(start=start, end=end, steps=steps, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
    )SCRIPT"},
        }
    ```

    2. Bump the `kProducedFileFormatVersion` by 1 and provide the reasons under `caffe2/caffe2/versions.h `
    ```
    // We describe new operator version bump reasons here:
    // 1) [01/24/2022]
    //     We bump the version number to 8 to update aten::linspace
    //     and aten::linspace.out to error out when steps is not
    //     provided. (see: https://github.com/pytorch/pytorch/issues/55951)
    // 2) [01/30/2022]
    //     Bump the version number to 9 to update aten::logspace and
    //     and aten::logspace.out to error out when steps is not
    //     provided. (see: https://github.com/pytorch/pytorch/issues/55951)
    constexpr uint64_t kProducedFileFormatVersion = 0x9L;
    ```

    3. In `caffe2/torch/csrc/jit/operator_upgraders/version_map.cpp`, add changes like below. You will need to make sure that the entry is **SORTED** according to the version bump number.
    ```
    {{"aten::linspace",
      {{8,
        "linspace_0_7",
        "aten::linspace(Scalar start, Scalar end, int? steps=None, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor"}}},
    ```

    4. After rebuild PyTorch, run the following command and it will auto generate a change to `caffe2/torch/csrc/jit/mobile/upgrader_mobile.cpp`. After rebuild PyTorch from source (`python setup.py`), run

    ```
    python pytorch/tools/codegen/operator_versions/gen_mobile_upgraders.py
    ```

    5. Add test:
        1. Using the model generated from step 1, you will need to add tests in following places:
            1. For mobile tests: `test/test_save_load_for_op_versions.py`
            2. For server tests: `test/jit/test_upgraders.py`
    6. Commit  changes made in all changes in step 2 in a single PR
