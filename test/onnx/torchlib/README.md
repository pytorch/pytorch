# Test op correctness by comparing with PyTorch results using OpInfo

`OpInfo` is PyTorch's standard mechanism for composing test data for operators.
Read more about them on https://github.com/pytorch/pytorch/blob/ce4a097bf769d753712a1fd969b446c59e29d8b9/torch/testing/_internal/opinfo/core.py#L362.

## Usage

```bash
# All
python -m pytest test_ops.py

# To run tests on a specific operator (e.g. torch.ceil):
python -m pytest test_ops.py -k ceil

# To run tests on a nn operator (e.g. nn.functional.scaled_dot_product_attention):
python -m pytest test_ops.py -k nn_functional_scaled_dot_product_attention
```

### Environment variables

1. Set environment variable `CATCH_ORT_SEGFAULT=1` to catch segmentation faults
in onnxruntime by running the inference sessions in a separate process.
2. Set `CREATE_REPRODUCTION_REPORT=1` to create markdown files for reproduction of errors. E.g.

    ```bash
    CREATE_REPRODUCTION_REPORT=1 python -m pytest test/onnx/torchlib/test_ops.py -k div_mode_int
    ```

## How to add a new operator test

See _usage_ in [`ops_test_data.py`](./ops_test_data.py)

## How to add custom OpInfo tests

Sometimes, there is no existing OpInfo that fits our need to test an operator. You want to create a custom OpInfo for it.

Follow the steps below to create new OpInfo tests:

1. Use the implementation for `ops.aten.slice_scatter` as a reference (https://github.com/microsoft/onnxscript/blob/e67335101e4a06b8cc98cb4129935a9af5062c77/tests/function_libs/torch_lib/extra_opinfo.py#L2412-L2418) to declare an OpInfo in [`extra_opinfo.py`](./extra_opinfo.py)

   ```py
    opinfo_core.OpInfo(
        "ops.aten.slice_scatter",
        aten_name="slice_scatter",
        dtypes=common_dtype.all_types_and(torch.bfloat16, torch.half, torch.bool),
        sample_inputs_func=sample_inputs_slice_scatter,
        supports_out=False,
    ),
    ```

   - The first argument should be the operator name under the `torch.ops` namespace. For example, if you want to test the `prims.var` op, then put `"ops.prims.var"`. It should almost always start with `ops.`.
   - Follow existing examples to specify the `dtypes` you want to test the op on.
   - Specify `op=` if the target operator is not the same as the OpInfo name (first arg). For example https://github.com/microsoft/onnxscript/blob/e67335101e4a06b8cc98cb4129935a9af5062c77/tests/function_libs/torch_lib/extra_opinfo.py#L2065-L2068.

    ```py
        opinfo_core.OpInfo(
            "ops.aten.bernoulli.p_deterministic",
            op=torch.ops.aten.bernoulli.p,
    ```

    The op is `torch.ops.aten.bernoulli.p`, which is different from the name `ops.aten.bernoulli.p_deterministic`. OpInfo names need to be globally unique in a test suite. When `op` is not specified, it will look for the op in `torch.` using its name.

2. Implement the `sample_inputs_func`. (Ref: https://github.com/microsoft/onnxscript/blob/e67335101e4a06b8cc98cb4129935a9af5062c77/tests/function_libs/torch_lib/extra_opinfo.py#L1242-L1268)
   1. Copy the function and decide what the input shapes should be. Use `make_arg` to generate a torch.Tensor. Alternatively you could also use `torch.tensor` to generate the tensor yourself. Be sure to double check the dtype and device. Finally yield each test cases with

   ```py
   yield opinfo_core.SampleInput(input, args=(...), kwargs={...})
   ```

   `input` is the first arg. The rest of the args are in `args`.
3. Enable the test case in [`ops_test_data.py`](./ops_test_data.py)
    1. Add a `TorchLibOpInfo` entry to the `TESTED_TORCHLIB_OPS` list. (For example https://github.com/microsoft/onnxscript/blob/e67335101e4a06b8cc98cb4129935a9af5062c77/tests/function_libs/torch_lib/ops_test_data.py#L2116)

    ```py
    TorchLibOpInfo("ops.aten.slice_scatter", core_ops.aten_slice_scatter)
    ```

    You can additionally specify dtype tolerance (https://github.com/microsoft/onnxscript/blob/e67335101e4a06b8cc98cb4129935a9af5062c77/tests/function_libs/torch_lib/ops_test_data.py#L539) or conditional skips (https://github.com/microsoft/onnxscript/blob/e67335101e4a06b8cc98cb4129935a9af5062c77/tests/function_libs/torch_lib/ops_test_data.py#L586-L590).

Now that the test is added, you may run the test like mentioned above. Set `CREATE_REPRODUCTION_REPORT=1` to get markdown reports and view failing input combinations should any test case fails.
