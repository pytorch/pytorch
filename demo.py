import torch
import torch.onnx._internal.ops._impl


input_data = torch.rand(2, 3, 4, 8)
position_ids_data = torch.randint(0, 50, (2, 3)).long()
sin_cache_data = torch.rand(50, 4)
cos_cache_data = torch.rand(50, 4)

# Eager mode is supported. Autograd is also supported so users can choose to use the op
# in development and production
result = torch.ops.onnx.RotaryEmbedding(
    input_data, cos_cache_data, sin_cache_data, position_ids_data
)
print(result.shape)


class Model(torch.nn.Module):
    def forward(self, input_data, cos_cache_data, sin_cache_data, position_ids_data):
        # Users can choose to use the ONNX op directly
        # Opset version is automatically selected by the PyTorch dispatcher
        # It is also possible to specify the overload:
        # torch.ops.onnx.RotaryEmbedding.opset23
        return torch.ops.onnx.RotaryEmbedding(
            input_data, cos_cache_data, sin_cache_data, position_ids_data
        )


model = Model()

# Dynamic shapes are supported
dynamic_shapes = {
    "input_data": {0: torch.export.Dim.DYNAMIC},
    "cos_cache_data": None,
    "sin_cache_data": None,
    "position_ids_data": {0: torch.export.Dim.DYNAMIC},
}

# The program can be exported with the onnx op preserved
ep = torch.export.export(
    model,
    (input_data, cos_cache_data, sin_cache_data, position_ids_data),
    dynamic_shapes=dynamic_shapes,
)
print("ep", ep)

# The program can be decomposed into aten ops so it is fully compatible with the PyTorch ecosystem
aten_decomped = ep.run_decompositions(torch.onnx._internal.ops._impl._ONNX_DECOMP_TABLE)
print("aten_decomped", aten_decomped)

# It can be exported to ONNX without any additional registration
onnx_program = torch.onnx.export(
    model,
    (input_data, cos_cache_data, sin_cache_data, position_ids_data),
    dynamo=True,
    dynamic_shapes=dynamic_shapes,
)
print("onnx", onnx_program.model)

"""
output:

torch.Size([2, 3, 4, 8])
/home/justinchu/dev/pytorch/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /home/justinchu/dev/pytorch/aten/src/ATen/Context.cpp:148.)
  torch._C._set_onednn_allow_tf32(_allow_tf32)
/home/justinchu/dev/pytorch/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /home/justinchu/dev/pytorch/aten/src/ATen/Context.cpp:148.)
  torch._C._set_onednn_allow_tf32(_allow_tf32)
ep ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, input_data: "f32[s0, 3, 4, 8]", cos_cache_data: "f32[50, 4]", sin_cache_data: "f32[50, 4]", position_ids_data: "i64[s0, 3]"):
             # File: /home/justinchu/dev/pytorch/demo.py:24 in forward, code: return torch.ops.onnx.RotaryEmbedding(
            rotary_embedding: "f32[s0, 3, 4, 8]" = torch.ops.onnx.RotaryEmbedding.opset23(input_data, cos_cache_data, sin_cache_data, position_ids_data);  input_data = cos_cache_data = sin_cache_data = position_ids_data = None
            return (rotary_embedding,)

Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='input_data'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='cos_cache_data'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='sin_cache_data'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='position_ids_data'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='rotary_embedding'), target=None)])
Range constraints: {s0: VR[2, int_oo]}

/home/justinchu/dev/pytorch/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /home/justinchu/dev/pytorch/aten/src/ATen/Context.cpp:148.)
  torch._C._set_onednn_allow_tf32(_allow_tf32)
aten_decomped ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, input_data: "f32[s0, 3, 4, 8]", cos_cache_data: "f32[50, 4]", sin_cache_data: "f32[50, 4]", position_ids_data: "i64[s0, 3]"):
             # File: /home/justinchu/dev/pytorch/demo.py:24 in forward, code: return torch.ops.onnx.RotaryEmbedding(
            slice_1: "f32[s0, 3, 4, 8]" = torch.ops.aten.slice.Tensor(input_data, 0, 0, 9223372036854775807)
            slice_2: "f32[s0, 3, 4, 8]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
            slice_3: "f32[s0, 3, 4, 8]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 9223372036854775807);  slice_2 = None
            slice_4: "f32[s0, 3, 4, 8]" = torch.ops.aten.slice.Tensor(input_data, 0, 0, 9223372036854775807);  input_data = None
            slice_5: "f32[s0, 3, 4, 8]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 9223372036854775807);  slice_4 = None
            slice_6: "f32[s0, 3, 4, 8]" = torch.ops.aten.slice.Tensor(slice_5, 2, 0, 9223372036854775807);  slice_5 = None
            slice_7: "f32[s0, 3, 4, 0]" = torch.ops.aten.slice.Tensor(slice_6, 3, 8, 9223372036854775807);  slice_6 = None
            index: "f32[s0, 3, 4]" = torch.ops.aten.index.Tensor(cos_cache_data, [position_ids_data]);  cos_cache_data = None
            index_1: "f32[s0, 3, 4]" = torch.ops.aten.index.Tensor(sin_cache_data, [position_ids_data]);  sin_cache_data = position_ids_data = None
            slice_8: "f32[s0, 3, 4]" = torch.ops.aten.slice.Tensor(index, 0, 0, 9223372036854775807);  index = None
            slice_9: "f32[s0, 3, 4]" = torch.ops.aten.slice.Tensor(slice_8, 1, 0, 9223372036854775807);  slice_8 = None
            slice_10: "f32[s0, 3, 4]" = torch.ops.aten.slice.Tensor(index_1, 0, 0, 9223372036854775807);  index_1 = None
            slice_11: "f32[s0, 3, 4]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 9223372036854775807);  slice_10 = None
            unsqueeze: "f32[s0, 3, 1, 4]" = torch.ops.aten.unsqueeze.default(slice_9, 2);  slice_9 = None
            unsqueeze_1: "f32[s0, 3, 1, 4]" = torch.ops.aten.unsqueeze.default(slice_11, 2);  slice_11 = None
            split = torch.ops.aten.split.Tensor(slice_3, 4, -1);  slice_3 = None
            getitem: "f32[s0, 3, 4, 4]" = split[0]
            getitem_1: "f32[s0, 3, 4, 4]" = split[1];  split = None
            mul: "f32[s0, 3, 4, 4]" = torch.ops.aten.mul.Tensor(unsqueeze, getitem)
            mul_1: "f32[s0, 3, 4, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_1, getitem_1)
            sub: "f32[s0, 3, 4, 4]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
            mul_2: "f32[s0, 3, 4, 4]" = torch.ops.aten.mul.Tensor(unsqueeze_1, getitem);  unsqueeze_1 = getitem = None
            mul_3: "f32[s0, 3, 4, 4]" = torch.ops.aten.mul.Tensor(unsqueeze, getitem_1);  unsqueeze = getitem_1 = None
            add: "f32[s0, 3, 4, 4]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
            cat: "f32[s0, 3, 4, 8]" = torch.ops.aten.cat.default([sub, add], -1);  sub = add = None
            cat_1: "f32[s0, 3, 4, 8]" = torch.ops.aten.cat.default([cat, slice_7], -1);  cat = slice_7 = None
            return (cat_1,)

Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='input_data'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='cos_cache_data'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='sin_cache_data'), target=None, persistent=None), InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='position_ids_data'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='cat_1'), target=None)])
Range constraints: {s0: VR[2, int_oo]}

/home/justinchu/anaconda3/envs/pytorch/lib/python3.12/site-packages/onnxscript/converter.py:820: FutureWarning: 'onnxscript.values.Op.param_schemas' is deprecated in version 0.1 and will be removed in the future. Please use '.op_signature' instead.
  param_schemas = callee.param_schemas()
/home/justinchu/anaconda3/envs/pytorch/lib/python3.12/site-packages/onnxscript/converter.py:820: FutureWarning: 'onnxscript.values.OnnxFunction.param_schemas' is deprecated in version 0.1 and will be removed in the future. Please use '.op_signature' instead.
  param_schemas = callee.param_schemas()
[torch.onnx] Obtain model graph for `Model()` with `torch.export.export(..., strict=False)`...
/home/justinchu/dev/pytorch/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /home/justinchu/dev/pytorch/aten/src/ATen/Context.cpp:148.)
  torch._C._set_onednn_allow_tf32(_allow_tf32)
[torch.onnx] Obtain model graph for `Model()` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
/home/justinchu/dev/pytorch/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /home/justinchu/dev/pytorch/aten/src/ATen/Context.cpp:148.)
  torch._C._set_onednn_allow_tf32(_allow_tf32)
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
onnx <
    ir_version=10,
    opset_imports={'pkg.onnxscript.torch_lib.common': 1, '': 18},
    producer_name='pytorch',
    producer_version='2.7.0a0+git5da00d7',
    domain=None,
    model_version=None,
>
graph(
    name=main_graph,
    inputs=(
        %"input_data"<FLOAT,[s0,3,4,8]>,
        %"cos_cache_data"<FLOAT,[50,4]>,
        %"sin_cache_data"<FLOAT,[50,4]>,
        %"position_ids_data"<INT64,[s0,3]>
    ),
    outputs=(
        %"rotary_embedding"<FLOAT,[s0,3,4,8]>
    ),
) {
    0 |  # rotary_embedding
         %"rotary_embedding"<FLOAT,[s0,3,4,8]> ⬅️ ::RotaryEmbedding(%"input_data", %"cos_cache_data", %"sin_cache_data", %"position_ids_data")
    return %"rotary_embedding"<FLOAT,[s0,3,4,8]>
}
"""
