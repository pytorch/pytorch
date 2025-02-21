import torch
import torch.onnx._internal.ops._impl

input_data = torch.rand(2, 3, 4, 8)
position_ids_data = torch.randint(0, 50, (2, 3)).long()
sin_cache_data = torch.rand(50, 4)
cos_cache_data = torch.rand(50, 4)

result = torch.ops.onnx.RotaryEmbedding(
    input_data, cos_cache_data, sin_cache_data, position_ids_data
)
print(result.shape)


class Model(torch.nn.Module):
    def forward(self, input_data, cos_cache_data, sin_cache_data, position_ids_data):
        return torch.ops.onnx.RotaryEmbedding(
            input_data, cos_cache_data, sin_cache_data, position_ids_data
        )


model = Model()
ep = torch.export.export(
    model, (input_data, cos_cache_data, sin_cache_data, position_ids_data)
)
print("ep", ep)

aten_decomped = ep.run_decompositions(torch.onnx._internal.ops._impl._ONNX_DECOMP_TABLE)
print("aten_decomped", aten_decomped)

onnx_program = torch.onnx.export(
    model, (input_data, cos_cache_data, sin_cache_data, position_ids_data), dynamo=True
)
print("onnx", onnx_program.model)
