# import torch

# fixed_values = torch.tensor([[    0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000],
#                              [    0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000],
#                              [    0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000],
#                              [-1068.6385,     0.0000,    65.0000,     0.0000, -torch.inf,     0.0000]],
#                             dtype=torch.float64)

# free_vars_mask = torch.tensor([[ True,  True, False,  True,  True,  True],
#                                [ True, False,  True,  True,  True,  True],
#                                [ True,  True,  True,  True,  True,  True],
#                                [False, False, False, False, False,  True]])

# free_vars_linear_indices = torch.where(free_vars_mask.ravel())[0]
# free_vars_indices = tuple(map(lambda x: x.detach().clone(),\
#                               torch.unravel_index(free_vars_linear_indices,\
#                                                   free_vars_mask.shape)))

# test_input = torch.tensor([-218.5399,    3.1056,   21.8333,    4.1535,    0.2   ,  144.8986,
#                              49.6429,   60.1429,    3.9028,    0.59  ,  126.218 ,   -6.0392,
#                              98.5   ,   35.5714,    4.8792,    0.2   ,    0.01  ],
#                           dtype=torch.float64)

# def to_constrained_params(vars):
#     processed_vars = torch.zeros_like(vars)
#     processed_vars[:, 0] += torch.abs(vars[:, 2])
#     processed_vars[:, 0] += torch.abs(vars[:, 2])
#     return processed_vars

# def test_func(free_vars):
#     free_vars_same_shape = torch.zeros_like(fixed_values, dtype=free_vars.dtype)
#     free_vars_same_shape[free_vars_indices] += free_vars
#     processed_vars = to_constrained_params(fixed_values + free_vars_same_shape)
#     return processed_vars.sum()

# print(torch.func.grad(test_func, argnums=0)(test_input))
# print(torch.compile(torch.func.grad(test_func, argnums=0), fullgraph=True, backend="inductor")(test_input))


import torch
from torch import device

def g(x):
    y = torch.zeros_like(x)
    y[:, 0] += x[:, 2].abs()
    y[:, 0] += x[:, 2].abs()
    return y.sum()


def f(x):
    return torch.func.grad(g)(x)


@torch.compile(backend='inductor', fullgraph=True)
def h(x):
    return f(x)


x = torch.randn(2, 3)
print(f(x))
# tensor([[ 0.,  0., -2.],
#         [ 0.,  0.,  2.]])
# print(h(x))
# tensor([[ 0.,  0., -1.],
#         [ 0.,  0.,  1.]])


import torch
from torch import device


def forward(arg0_1: "f32[2, 3][3, 1]cpu"):
    slice_2: "f32[2, 3][3, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807)
    select_1: "f32[2][3]cpu" = torch.ops.aten.select.int(slice_2, 1, 2);  slice_2 = None

    # File: /home/guilhermeleobas/git/pytorch/c.py:45 in g, code: y[:, 0] += x[:, 2].abs()
    slice_10: "f32[2, 3][3, 1]cpu" = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807);  arg0_1 = None
    select_7: "f32[2][3]cpu" = torch.ops.aten.select.int(slice_10, 1, 2);  slice_10 = None

    # File: /home/guilhermeleobas/git/pytorch/torch/_functorch/eager_transforms.py:1433 in grad_and_value_impl, code: flat_grad_input = _autograd_grad(
    full_1: "f32[][]cpu" = torch.ops.aten.full.default([], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    expand: "f32[2, 3][0, 0]cpu" = torch.ops.aten.expand.default(full_1, [2, 3]);  full_1 = None
    new_empty_strided: "f32[2, 3][3, 1]cpu" = torch.ops.aten.new_empty_strided.default(expand, [2, 3], [3, 1])
    copy_2: "f32[2, 3][3, 1]cpu" = torch.ops.aten.copy.default(new_empty_strided, expand);  new_empty_strided = expand = None
    as_strided_1: "f32[2][3]cpu" = torch.ops.aten.as_strided.default(copy_2, [2], [3], 0)
    clone: "f32[2][1]cpu" = torch.ops.aten.clone.default(as_strided_1, memory_format = torch.contiguous_format)
    full_2: "f32[2][1]cpu" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    copy_3: "f32[2][3]cpu" = torch.ops.aten.copy.default(as_strided_1, full_2);  as_strided_1 = full_2 = None
    as_strided_scatter: "f32[2, 3][3, 1]cpu" = torch.ops.aten.as_strided_scatter.default(copy_2, copy_3, [2], [3], 0);  copy_2 = copy_3 = None
    full_3: "f32[6][1]cpu" = torch.ops.aten.full.default([6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_3: "f32[2][3]cpu" = torch.ops.aten.as_strided.default(full_3, [2], [3], 0)
    copy_4: "f32[2][3]cpu" = torch.ops.aten.copy.default(as_strided_3, clone);  as_strided_3 = clone = None
    as_strided_scatter_1: "f32[6][1]cpu" = torch.ops.aten.as_strided_scatter.default(full_3, copy_4, [2], [3], 0);  full_3 = copy_4 = None
    as_strided_6: "f32[2, 3][3, 1]cpu" = torch.ops.aten.as_strided.default(as_strided_scatter_1, [2, 3], [3, 1], 0);  as_strided_scatter_1 = None
    add_2: "f32[2, 3][3, 1]cpu" = torch.ops.aten.add.Tensor(as_strided_scatter, as_strided_6);  as_strided_scatter = as_strided_6 = None
    new_empty_strided_1: "f32[2, 3][3, 1]cpu" = torch.ops.aten.new_empty_strided.default(add_2, [2, 3], [3, 1])
    copy_5: "f32[2, 3][3, 1]cpu" = torch.ops.aten.copy.default(new_empty_strided_1, add_2);  new_empty_strided_1 = add_2 = None
    as_strided_8: "f32[2][3]cpu" = torch.ops.aten.as_strided.default(copy_5, [2], [3], 0)
    clone_1: "f32[2][1]cpu" = torch.ops.aten.clone.default(as_strided_8, memory_format = torch.contiguous_format)
    copy_6: "f32[2][3]cpu" = torch.ops.aten.copy.default(as_strided_8, clone_1);  as_strided_8 = None
    as_strided_scatter_2: "f32[2, 3][3, 1]cpu" = torch.ops.aten.as_strided_scatter.default(copy_5, copy_6, [2], [3], 0);  copy_5 = copy_6 = None
    sign: "f32[2][1]cpu" = torch.ops.aten.sign.default(select_7);  select_7 = None
    mul: "f32[2][1]cpu" = torch.ops.aten.mul.Tensor(clone_1, sign);  clone_1 = sign = None
    full_4: "f32[2, 3][3, 1]cpu" = torch.ops.aten.full.default([2, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_4: "f32[2, 3][3, 1]cpu" = torch.ops.aten.select_scatter.default(full_4, mul, 1, 2);  full_4 = mul = None
    full_5: "f32[2, 3][3, 1]cpu" = torch.ops.aten.full.default([2, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_4: "f32[2, 3][3, 1]cpu" = torch.ops.aten.slice_scatter.default(full_5, select_scatter_4, 0, 0, 9223372036854775807);  full_5 = select_scatter_4 = None
    new_empty_strided_2: "f32[2, 3][3, 1]cpu" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_2, [2, 3], [3, 1])
    copy_7: "f32[2, 3][3, 1]cpu" = torch.ops.aten.copy.default(new_empty_strided_2, as_strided_scatter_2);  new_empty_strided_2 = as_strided_scatter_2 = None
    as_strided_11: "f32[2][3]cpu" = torch.ops.aten.as_strided.default(copy_7, [2], [3], 0)
    clone_2: "f32[2][1]cpu" = torch.ops.aten.clone.default(as_strided_11, memory_format = torch.contiguous_format)
    full_6: "f32[2][1]cpu" = torch.ops.aten.full.default([2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    copy_8: "f32[2][3]cpu" = torch.ops.aten.copy.default(as_strided_11, full_6);  as_strided_11 = full_6 = None
    as_strided_scatter_3: "f32[2, 3][3, 1]cpu" = torch.ops.aten.as_strided_scatter.default(copy_7, copy_8, [2], [3], 0);  copy_7 = copy_8 = None
    full_7: "f32[6][1]cpu" = torch.ops.aten.full.default([6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_13: "f32[2][3]cpu" = torch.ops.aten.as_strided.default(full_7, [2], [3], 0)
    copy_9: "f32[2][3]cpu" = torch.ops.aten.copy.default(as_strided_13, clone_2);  as_strided_13 = clone_2 = None
    as_strided_scatter_4: "f32[6][1]cpu" = torch.ops.aten.as_strided_scatter.default(full_7, copy_9, [2], [3], 0);  full_7 = copy_9 = None
    as_strided_16: "f32[2, 3][3, 1]cpu" = torch.ops.aten.as_strided.default(as_strided_scatter_4, [2, 3], [3, 1], 0);  as_strided_scatter_4 = None
    add_3: "f32[2, 3][3, 1]cpu" = torch.ops.aten.add.Tensor(as_strided_scatter_3, as_strided_16);  as_strided_scatter_3 = as_strided_16 = None
    new_empty_strided_3: "f32[2, 3][3, 1]cpu" = torch.ops.aten.new_empty_strided.default(add_3, [2, 3], [3, 1])
    copy_10: "f32[2, 3][3, 1]cpu" = torch.ops.aten.copy.default(new_empty_strided_3, add_3);  new_empty_strided_3 = add_3 = None
    as_strided_18: "f32[2][3]cpu" = torch.ops.aten.as_strided.default(copy_10, [2], [3], 0);  copy_10 = None
    clone_3: "f32[2][1]cpu" = torch.ops.aten.clone.default(as_strided_18, memory_format = torch.contiguous_format);  as_strided_18 = None
    sign_1: "f32[2][1]cpu" = torch.ops.aten.sign.default(select_1);  select_1 = None
    mul_1: "f32[2][1]cpu" = torch.ops.aten.mul.Tensor(clone_3, sign_1);  clone_3 = sign_1 = None
    full_8: "f32[2, 3][3, 1]cpu" = torch.ops.aten.full.default([2, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_5: "f32[2, 3][3, 1]cpu" = torch.ops.aten.select_scatter.default(full_8, mul_1, 1, 2);  full_8 = mul_1 = None
    full_9: "f32[2, 3][3, 1]cpu" = torch.ops.aten.full.default([2, 3], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_5: "f32[2, 3][3, 1]cpu" = torch.ops.aten.slice_scatter.default(full_9, select_scatter_5, 0, 0, 9223372036854775807);  full_9 = select_scatter_5 = None
    add_4: "f32[2, 3][3, 1]cpu" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None
    return (add_4,)


@torch.compile(backend='inductor', fullgraph=True)
def j(x):
    return forward(x)


x = torch.randn(2, 3)
print(j(x))