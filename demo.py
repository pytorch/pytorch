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

# The program can be exported with the onnx op preserved
ep = torch.export.export(
    model, (input_data, cos_cache_data, sin_cache_data, position_ids_data)
)
print("ep", ep)

# The program can be decomposed into aten ops so it is fully compatible with the PyTorch ecosystem
aten_decomped = ep.run_decompositions(torch.onnx._internal.ops._impl._ONNX_DECOMP_TABLE)
print("aten_decomped", aten_decomped)

# It can be exported to ONNX without any additional registration
onnx_program = torch.onnx.export(
    model, (input_data, cos_cache_data, sin_cache_data, position_ids_data), dynamo=True
)
print("onnx", onnx_program.model)
