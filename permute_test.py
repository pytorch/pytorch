import torch
from torch.profiler import profile, record_function, ProfilerActivity

shape_list = [(4096, 1219, 160),
(4096, 160, 64),
(4096, 88, 160),
(4096, 160, 32)]

shape_list_unique = list(dict.fromkeys(shape_list))

tensors = [torch.randn(shape).cuda() for shape in shape_list_unique]
for i, tensor in enumerate(tensors):
# b = torch.randn(4096, 160, 1219).cuda()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    #     with_stack=True, with_modules=True, record_shapes = True) as prof:
    permuted_b = tensor.permute(0, 2, 1)
    permuted_c = permuted_b.contiguous()
# prof.export_chrome_trace("permute_ori.json")
# prof.export_chrome_trace("permute_tile.json")
# prof.export_chrome_trace("permute_256.json")
# 
# def custom_cuda_transpose_last_two(tensor):
#     batch_size, dim1, dim2 = tensor.shape
#     new_tensor = torch.empty(batch_size, dim2, dim1, device=tensor.device)
#     for i in range(batch_size):
#         for j in range(dim2):
#             for k in range(dim1):
#                 new_tensor[i][j][k] = tensor[i][k][j]
#     return new_tensor

# 创建在 CUDA 上的张量 b，形状不同于 a

# # 对张量 b 进行 permute 操作
# print("tensor b:", b)
# print("after permute b:", permuted_b)



# print("transpose:")
# transpose_b = b.transpose(2, 1)
# print("after transpose b:", transpose_b)

# result = custom_cuda_transpose_last_two(b)

# print(torch.equal(permuted_b.cpu(), transpose_b.cpu()))
# print(torch.equal(permuted_c.cpu(), result.cpu()))
# print(torch.equal(transpose_b.cpu(), result.cpu()))