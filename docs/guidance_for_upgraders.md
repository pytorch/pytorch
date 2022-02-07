# Guidance for Operator Developer

After you change either the operator signature in a BC breaking way or the semantics of the operator, you will need to write an “upgrader” to make the change non-BC breaking iff they are used in TorchScript or mobile. In general, you can know your operator is BC breaking, if it fails `test/forward_backward_compatibility/check_forward_backward_compatibility.py `

The steps to write upgrader:

1. Prepare a model before making changes to the operator, following the process:
    1. Add a module in `test/jit/fixtures_srcs/fixtures_src.py`. The reason why we generate the model file before updating the operator is that, an old model with the old operator before the change is needed to ensure the upgrader is working as expected. In `test/jit/fixtures_srcs/generate_models.py`, add the module and its corresponding changed operator name like following:

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
    3. Commit the change and submit a PR

2. Write an upgrader in `torch/csrc/jit/operator_upgraders/upgraders_entry.cpp` file inside a map `kUpgradersEntryMap`. The softly enforced naming format is `<operator_name>_<operator_overload>_<start>_<end>`. For example, the below example means that `linspace.out `and` linspace` at versions from 0 to 7 need to be replaced by these upgraders. Note that the upgrader is essentially a python source code that is TorchScriptable.

```
     {"linspace_0_7", R"SCRIPT(
        def linspace_0_7(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], *, dtype: Optional[int], layout: Optional[int],
                        device: Optional[Device], pin_memory: Optional[bool]):
        if (steps is None):
            return torch.linspace(start=start, end=end, steps=100, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
        return torch.linspace(start=start, end=end, steps=steps, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
        )SCRIPT"},
     {"linspace_out_0_7", R"SCRIPT(
        def linspace_out_0_7(start: Union[int, float, complex], end: Union[int, float, complex], steps: Optional[int], *, out: Tensor):
        if (steps is None):
            return torch.linspace(start=start, end=end, steps=100, out=out)
        return torch.linspace(start=start, end=end, steps=steps, out=out)
        )SCRIPT"},
```

3. Bump the `kProducedFileFormatVersion` and `kMaxProducedFileFormatVersion` by 1 and provide the reasons under `caffe2/versions.h`.

4. In `torch/csrc/jit/operator_upgraders/version_map.cpp`, add changes like below. You will need to make sure that the entry is **SORTED** according to the version bump number. Otherwise, the sanity check will fail.

```
{"div.Tensor",
     {{4,
       "div_Tensor_0_3",
       "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"}}},
```

This can be interpreted as: "We made a change to `div.Tensor` which caused a global operator version bump to 4
. For outdated operators (aka 0 - 3) which have a signature `aten::div.Tensor(Tensor self, Tensor other) -> Tensor`, they need to use `div_Tensor_0_3` to be runnable in new runtime.

5. After rebuild PyTorch, run the following command and it will auto generate a change to `torch/csrc/jit/mobile/upgrader_mobile.cpp`. This file contains the bytecode entries of the those upgraders which is used in mobile runtime.

```
python pytorch/tools/codegen/operator_versions/gen_mobile_upgraders.py
```

6. Using the model generated from step 1, you will need to add tests in following places:
    1. For mobile tests: test/test_save_load_for_op_versions.py
    2. For server tests: test/jit/test_upgraders.py

7. Commit changes made in step 2 to step 5 in a single PR.

You can look at following PRs to get the rough idea of what needs to be done:
1. PR that adds `logspace` test modules: https://github.com/pytorch/pytorch/pull/72052
2. PR that updates `logspace`: https://github.com/pytorch/pytorch/pull/72051

# How to deprecate an upgrader?

If an upgrader passes the expiration date and it’s a blocker for continuous development, it should be ok to remove it.
