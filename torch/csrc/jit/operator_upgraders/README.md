# Guidance for Operator Developer

After you change to operator either the operator schema is BC-breaking way or the semantics of the operator, you will need to write an “upgrader” to make the change non-BC breaking iff they are used in TorchScript or mobile. In general, you can know your operator is BC breaking, if it fails `test/forward_backward_compatibility/check_forward_backward_compatibility.py `

The steps to write upgrader:

1. Prepare a model before making changes to the operator, following the process:
     1. add a module in `test/jit/fixtures_srcs/fixtures_src.py`. The reason why switching to commit is that, an old model with the old operator before the change is needed to ensure the upgrader is working as expected. In `test/jit/fixtures_srcs/generate_models.py`, add the module and its corresponding changed operator like following

        ```
        ALL_MODULES = {
           TestVersionedLinspaceV7(): "aten::linspace",
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
    2. Write an upgrader in `caffe2/torch/csrc/jit/operator_upgraders/upgraders_entry.cpp` file inside a map `kUpgradersEntryMap`. The softly enforced naming format is `<operator_name>_<operator_overload>_<start>_<end>`. The start and end means the upgrader can be applied to the operator exported during when [the global operator version](https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/versions.h#L82) within the range `[start, end]`. Let's take an operator operator `linspace` with the overloaded name `out` as an example. The first thing is to check if the upgrader exists in in [upgraders_entry.cpp](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/operator_upgraders/upgraders_entry.cpp).
        1. If the upgrader doesn't exist in `upgraders_entry.cpp`, the upgrader name can be `linspace_out_0_{kProducedFileFormatVersion}`, where [`kProducedFileFormatVersion`](https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/versions.h#L82) can found in [versions.h](https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/versions.h).
        2. If the upgrader exist in `upgraders_entry.cpp`, for example `linspace_out_0_7` (means `linspace.out` operator is changed when operator version is bumped from 7 to 8),
            1. If it's possible to write an upgrader valid for `linspace` before versioning bumping to 8, after versioning bumping to 8, write an upgrader `linspace_out_0_{kProducedFileFormatVersion}`
            2. If it's imporssible to write an upgrader valid for `linspace` before versiioning bumping to 8, check the date when the version is bumped to 8  at [`versions.h`](https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/versions.h#L82). If it has been 180 days, write an upgrader `linspace_out_8_{kProducedFileFormatVersion}` for `linspace.out` after bumping to 8, and deprecate the old upgrader. If it hasn't been 180 days, wait until 180 days and do the same changes as above.

    To write an upgrader, you would need to know how the new runtime with the new `linspace` operator can handle an old model with the old `linspace` operator. When `linspace` is bumped to 8, the change is to make `step` a required argument, instead of an optional argument. The old schema is:
    ```
    linspace(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], dtype: Optional[int], layout: Optional[int],
                     device: Optional[Device], pin_memory: Optional[bool]):
    ```
    And the new schema is:
    ```
    linspace(start: Union[int, float, complex], end: Union[int, float, complex], steps: int, dtype: Optional[int], layout: Optional[int],
                     device: Optional[Device], pin_memory: Optional[bool]):
    ```
    upgrader will only be applied to old model and it won't be applied to new model. The upgrader can be written with the following logic:
    ```
    def linspace_0_7(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], *, dtype: Optional[int], layout: Optional[int],
                     device: Optional[Device], pin_memory: Optional[bool]):
      if (steps is None):
        return torch.linspace(start=start, end=end, steps=100, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
      return torch.linspace(start=start, end=end, steps=steps, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
    ```

    The actual upgrader need to be written as TorchScript, and the below example is the actual upgrader of the operator `linspace.out `and the operator ` linspace` exported at version from 0 to 7.
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
    With the upgrader, when new runtime loads an old model, it will first check the operator version of the old model. If it's older than the current runtime, it will replace the operator from the old model with the upgrader above.

    2. Bump [`kMaxSupportedFileFormatVersion`](https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/versions.h#L15) the [`kProducedFileFormatVersion`](https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/versions.h#L82) by 1 and provide the reasons under [`versions.h`](https://github.com/pytorch/pytorch/blob/master/caffe2/serialize/versions.h#L73-L81)
    ```

    constexpr uint64_t kMaxSupportedFileFormatVersion = 0x9L;

    ...
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

You can look at following PRs to get the rough idea of what needs to be done:
1. [PR that adds `logspace` test modules](https://github.com/pytorch/pytorch/pull/72052)
2. [PR that updates `logspace`](https://github.com/pytorch/pytorch/pull/72051)

---
**NOTE**

Adding default arguments is not BC breaking and it doesn't requirement. For example, the following change is not BC breaking, and doesn't require upgrader:
```
# before
def foo(x, y):
    return x, y
```
```
# after
def foo(x, y, z=100):
    return x, y, z
```

---
