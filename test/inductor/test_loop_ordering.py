import torch
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(114755)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, sentinel):
    var_node_4 = torch.full((2, 3), 3, dtype=torch.int16) # size=(2, 3), stride=(3, 1), dtype=int16, device=cuda
    var_node_3 = torch.unique(var_node_4) # size=(5,), stride=(1,), dtype=int16, device=cuda
    var_node_6 = arg_0 # size=(5,), stride=(1,), dtype=int16, device=cuda
    var_node_5 = var_node_6.contiguous().view([5]) # size=(5,), stride=(1,), dtype=int16, device=cuda
    var_node_2 = torch.mul(var_node_3, var_node_5) # size=(5,), stride=(1,), dtype=int16, device=cuda
    var_node_1 = torch.unsqueeze(var_node_2, dim=1) # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_10 = arg_1 # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_11 = arg_2 # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_9 = torch.sub(var_node_10, var_node_11) # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_13 = arg_3 # size=(5,), stride=(76,), dtype=int16, device=cuda
    var_node_12 = torch.reshape(var_node_13, [5, 1]) # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_8 = torch.div(var_node_9, var_node_12) # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_14 = arg_4 # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_7 = torch.sub(var_node_8, var_node_14) # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    var_node_0 = torch.mul(var_node_1, var_node_7) # size=(5, 1), stride=(1, 1), dtype=int16, device=cuda
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randint(5, 30, (5,)).to(torch.int16), (5,), (1,))
arg_1 = torch.as_strided(torch.randint(5, 30, (5,)).to(torch.int16), (5, 1), (1, 1))
arg_2 = torch.as_strided(torch.randint(5, 30, (5,)).to(torch.int16), (5, 1), (1, 1))
arg_3 = torch.as_strided(torch.randint(5, 30, (305,)).to(torch.int16), (5,), (76,))
arg_4 = torch.as_strided(torch.randint(5, 30, (5,)).to(torch.int16), (5, 1), (1, 1))

args = (arg_0, arg_1, arg_2, arg_3, arg_4) + (sentinel,)
result_original = fuzzed_program(*args)
print('✅ eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print('✅ compile success')