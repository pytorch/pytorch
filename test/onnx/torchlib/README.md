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

PyTorch-owned implementations are ordinary Python functions accepting `onnx_ir.Value`
inputs. Use `current_builder().call("Operator", ...)` from
`torch.onnx._internal.exporter._ir_builder` to construct nodes; unpack its tuple of
outputs explicitly. Tensor casts, shape construction, and control flow are explicit.
The builder converts Python constants according to the ONNX schema but does not
promote tensor inputs. Operator annotations document inputs and outputs, not dispatch
overloads: registration selects the ATen target, real/complex branch, and opset.

The test harness activates a native builder and supplies plain IR inputs for these
functions. The isolated ONNX Script evaluator path remains for external torch_lib
implementations until they are migrated. Tests requiring newer operators also require
an ONNX Runtime release supporting their opset.

Set `TorchLibOpInfo.supported_dtypes` to preserve a migrated implementation's
source-supported numerical coverage without depending on its old annotations.
This metadata only filters tests; it does not participate in exporter dispatch.
Keep source tolerances and sample-specific skips with the migrated mapping.

Structural core translations (views, axis transforms, indexing, scattering, masking,
and reductions) use native IR registrations. Additional source samples live in
`extra_opinfo.py`; they cover broadcast masks, optional slice bounds, scalar
scatter inputs, and advanced index placement. Native regressions in
`../exporter/test_native_structural_ops.py` also exercise symbolic dimensions,
complex component axes, duplicate-index reductions, and FX export with optimization
both enabled and disabled. Those tests prohibit legacy graph construction.
Use ONNX's reference evaluator when the installed runtime does not support a
tested opset; do not skip a migrated translation merely because its opset is newer.

Factories, random sampling, windows, matrix products, and Tensor[] operations
also use native IR. `factories.py`, `linalg.py`, and `sequences.py` preserve source
implementations and comments without ONNX Script evaluation. Dynamic split sizes
produce typed ONNX sequences, explicitly unpacked at FX multioutput boundaries.
Sequence maps use native IR subgraphs and `If` nodes. Factory sizes may be an
INT64 tensor or a list of static/symbolic dimensions. Device, generator, and
layout limitations remain those of the source translations; empty allocation
is represented by zeros. Random tests compare shapes, dtypes, and valid ranges,
not identical random streams. Window tolerances retain ONNX's Hamming coefficient
difference. `test_native_factory_sequence_ops.py` covers dynamic sizes, empty
and scalar outputs, repeats, complex stacking, and optimizer-off/on FX exports.
The OpInfo harness uses the reference evaluator for boolean `SplitToSequence`,
which is missing from older ONNX Runtime kernel tables.

Recurrent, embedding, and residual core operators live in `recurrent.py`,
`embedding.py`, and `residual.py`. GRU/LSTM preserve gate ordering, direction
layout, optional biases, initial states, and interlayer dropout. Sequence length
and batch size may be dynamic; the recurrent hidden size must be known.
Embedding bags use explicitly typed `Loop`/`If` subgraphs with lexical captures,
including empty bags, padding, per-sample weights, and auxiliary outputs.
`test_native_recurrent_residual_ops.py` compares these graphs with PyTorch and
exports models with optimization both enabled and disabled without legacy
translations.

Convolution and normalization now live in `normalization.py`, including complex
convolutions and functional batch-normalization outputs. Training is retained
rather than forced to inference for older ORT kernels. Saved inference statistics
keep source torchlib's CUDA convention, which differs from CPU's empty outputs.
`neural.py` contains source NN translations, including opset18 GELU, group norm,
and attention alternatives to the newer implementations in `nn.py`. Grouped-query
attention requires static head counts; batch and sequence sizes may be dynamic.
`test_native_neural_ops.py` checks native-only dynamic FX exports with optimization
off/on. Source OpInfo mappings retain numerical tolerances and known limitations.
The experimental flash/efficient-attention wrappers are native as well. Like
the source translations, their auxiliary outputs are zero placeholders, not
usable log-sum-exp statistics, sequence offsets, debug masks, or RNG state.

FFT, prims, special, determinant/vector norm, optional vision, and decomposed
dequantization translations complete the native registry. Built-in population
does not consult ONNX Script's torch_lib. Construction uses plain `onnx_ir.Value`
objects and `GraphBuilder`; ONNX Script remains available for optimization and
the isolated legacy `custom_translation_table` adapter. Global native
registrations are snapshotted per export; local translations override globals,
which override built-ins. Built-ins select the highest eligible opset separately
for real and complex inputs without dtype/signature overload searches.

`test_native_final_ops.py` checks real/complex FFT, prims, special functions,
quantization boundaries, attention auxiliary shapes, and direct vision graphs.
Vision registration uses qualified names without importing torchvision; optional
operators become executable FX targets only when their namespace is loaded.
Unavailable/version-specific source aliases remain qualified-name registrations.
DeformConv requires opset19; DFT uses the axis input starting at opset20.
The source's 622 unsupported placeholder bodies do not add operator support.

See _usage_ in [`ops_test_data.py`](./ops_test_data.py)

## How to add custom OpInfo tests

Sometimes, there is no existing OpInfo that fits our need to test an operator. You want to create a custom OpInfo for it.

Follow the steps below to create new OpInfo tests:

1. Use the implementation for `ops.aten.slice_scatter` as a reference (https://github.com/microsoft/onnxscript/blob/e67335101e4a06b8cc98cb4129935a9af5062c77/tests/function_libs/torch_lib/extra_opinfo.py#L2412-L2418) to declare an `OpInfo` in `extra_opinfo.py`.

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
