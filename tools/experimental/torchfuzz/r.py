import torch
torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1412198902)

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    var_node_3 = torch.full((16, 11), 0.197265625, dtype=torch.bfloat16) # size=(16, 11), stride=(11, 1), dtype=bfloat16, device=cuda
    var_node_4 = arg_0 # size=(11, 16), stride=(16, 1), dtype=bfloat16, device=cuda
    var_node_2 = torch.matmul(var_node_3.to(torch.bfloat16), var_node_4.to(torch.bfloat16)) # size=(16, 16), stride=(16, 1), dtype=bfloat16, device=cuda
    var_node_6 = arg_1 # size=(16, 2), stride=(2, 1), dtype=float32, device=cuda
    var_node_7 = torch.full((2, 16), 0.8236594200134277, dtype=torch.float32) # size=(2, 16), stride=(16, 1), dtype=float32, device=cuda
    var_node_5 = torch.matmul(var_node_6.to(torch.float32), var_node_7.to(torch.float32)) # size=(16, 16), stride=(16, 1), dtype=float32, device=cuda
    var_node_1 = torch.add(var_node_2, var_node_5) # size=(16, 16), stride=(16, 1), dtype=bfloat16, device=cuda
    var_node_10 = torch.full((16, 12), 0.0693359375, dtype=torch.bfloat16) # size=(16, 12), stride=(12, 1), dtype=bfloat16, device=cuda
    var_node_11 = arg_2 # size=(12, 14), stride=(14, 1), dtype=bfloat16, device=cuda
    var_node_9 = torch.matmul(var_node_10.to(torch.bfloat16), var_node_11.to(torch.bfloat16)) # size=(16, 14), stride=(14, 1), dtype=bfloat16, device=cuda
    var_node_12 = torch.full((14, 28), 0.2353515625, dtype=torch.bfloat16) # size=(14, 28), stride=(28, 1), dtype=bfloat16, device=cuda
    var_node_8 = torch.matmul(var_node_9.to(torch.bfloat16), var_node_12.to(torch.bfloat16)) # size=(16, 28), stride=(28, 1), dtype=bfloat16, device=cuda
    var_node_0 = torch.matmul(var_node_1.to(torch.bfloat16), var_node_8.to(torch.bfloat16)) # size=(16, 28), stride=(1, 16), dtype=bfloat16, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randn(176).to(torch.bfloat16), (11, 16), (16, 1))
arg_1 = torch.as_strided(torch.randn(32).to(torch.float32), (16, 2), (2, 1))
arg_2 = torch.as_strided(torch.randn(168).to(torch.bfloat16), (12, 14), (14, 1))

args = (arg_0, arg_1, arg_2) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')
