import torch
from torch.jit import BatchTensor


@torch.jit.script
def batch_tanh(data, mask, dims):
    data = torch.tanh(data)
    return data, mask, dims


@torch.jit.script
def batch_sigmoid(data, mask, dims):
    data = torch.sigmoid(data)
    return data, mask, dims


@torch.jit.script
def batch_add(data1, mask1, dims1, data2, mask2, dims2):
    data = torch.add(data1, data2)
    mask = mask1 * mask2
    dims = dims1 or dims2
    return data, mask, dims


@torch.jit.script
def batch_sub(data1, mask1, dims1, data2, mask2, dims2):
    data = torch.sub(data1, data2)
    mask = mask1 * mask2
    dims = dims1 or dims2
    return data, mask, dims


@torch.jit.script
def batch_mul(data1, mask1, dims1, data2, mask2, dims2):
    data = torch.mul(data1, data2)
    mask = mask1 * mask2
    dims = dims1 or dims2
    return data, mask, dims


@torch.jit.script
def batch_mm(data1, mask1, dims1, data2, mask2, dims2):
    data1 = data1 * mask1.type_as(data1)
    data2 = data2 * mask2.type_as(data2)
    data = torch.bmm(data1, data2)
    mask = torch.bmm(mask1.narrow(2, 0, 1), mask2.narrow(1, 0, 1))
    dims = torch.cat((dims1[:1], dims2[1:dims2.size(0)]))
    return data, mask, dims


@torch.jit.script
def batch_matmul(data1, mask1, dims1, data2, mask2, dims2):
    d1 = data1.dim() - 1
    d2 = data2.dim() - 1
    data1 = data1 * mask1.type_as(data1)
    data2 = data2 * mask2.type_as(data2)
    if d1 == 1:
        data1 = data1.unsqueeze(-2)
    if d2 == 1:
        data2 = data2.unsqueeze(-1)
    data = torch.bmm(data1, data2)
    mask = mask1
    dims = dims1
    if d1 == 1 and d2 == 1:
        # if (batch1.dims[0] or batch2.dims[0]) and not batch1.mask.eq(batch2.mask).all():
        #    raise ValueError("cannot contract non-matching dimensions")
        data = data.squeeze(-1).squeeze(-1)
        mask = mask1.narrow(1, 0, 1).squeeze(-1)
        dims = dims1[:0]  # empty tensor
    if d1 == 2 and d2 == 1:
        # if (batch1.dims[1] or batch2.dims[0]) and not batch1.mask[:, 0].eq(batch2.mask).all():
        #    raise ValueError("cannot contract non-matching dimensions")
        data = data.squeeze(-1)
        mask = torch.bmm(mask1.narrow(2, 0, 1), mask2.narrow(1, 0, 1).unsqueeze(-1)).squeeze(-1)
        dims = dims1[:1]
    elif d1 == 1 and d2 == 2:
        # if (batch1.dims[0] or batch2.dims[0]) and not batch1.mask.eq(batch2.mask[:, :, 0]).all():
        #    raise ValueError("cannot contract non-matching dimensions")
        data = data.squeeze(-2)
        mask = torch.bmm(mask1.narrow(1, 0, 1).unsqueeze(-2), mask2.narrow(1, 0, 1)).squeeze(-2)
        dims = dims2[1:dims2.size(0)]
    elif d1 == 2 and d2 == 2:
        # if (batch1.dims[1] or batch2.dims[0]) and not batch1.mask[:, 0].eq(batch2.mask[:, :, 0]).all():
        #    raise ValueError("cannot contract non-matching dimensions")
        mask = torch.bmm(mask1.narrow(2, 0, 1), mask2.narrow(1, 0, 1))
        dims = torch.cat((dims1[:1], dims2[1:dims2.size(0)]))
    # else:
    #     raise NotImplementedError("matmul not implemented with batches of 3+D tensors")
    return data, mask, dims


@torch.jit.script
def batch_select(data, mask, dims, dim, index):
    # if dim == 0:
    #     raise ValueError("Cannot select 0 dim in BatchTensor")
    data = data.select(dim, index)
    if dims[dim - 1]:
        mask = mask.select(dim, 0)
    else:
        mask = mask.select(dim, index)
    dims = torch.cat((dims[:dim - 1], dims[dim:dims.size(0)]))
    return data, mask, dims


# assume data, data1, data2 have same size
@torch.jit.script
def batch_where(data, mask, dims, data1, mask1, dims1, data2, mask2, dims2):
    res_data = torch.where(data, data1, data2)
    res_mask = torch.where(data, mask1, mask2)
    res_dims = dims1 or dims2
    return res_data, res_mask, res_dims


@torch.jit.script
def batch_update(batch_data, batch_mask, batch_dims, new_data, new_mask, new_dims):
    data = torch.where(new_mask, new_data, batch_data)
    return data, new_mask, new_dims  # TODO: consider whether return new_mask and new_dims


@torch.jit.script
def batch_any(data, mask, dims):
    return torch.gt(torch.sum(data * mask), 0)


@torch.jit.script
def batch_any_false(data, mask, dims):
    data = torch.eq(data, torch.zeros_like(data))
    return batch_any(data, mask, dims)


@torch.jit.script
def batch_type_as(data, mask, dims, data1, mask1, dims1):
    return data.type_as(data1), mask, dims


@torch.jit.script
def batch_gt(data, mask, dims, data1, mask1, dims1):
    return torch.gt(data, data1), mask * mask1, dims or dims1


@torch.jit.script
def batch_from_scalar_tensor(data):
    data = data.unsqueeze(0)
    mask = torch.ones([1], dtype=torch.uint8)
    dims = torch.zeros([0], dtype=torch.uint8)
    return data, mask, dims

torch.register_batch_operator("tanh", batch_tanh.graph)
torch.register_batch_operator("sigmoid", batch_sigmoid.graph)
torch.register_batch_operator("add", batch_add.graph)
torch.register_batch_operator("sub", batch_sub.graph)
torch.register_batch_operator("mul", batch_mul.graph)
torch.register_batch_operator("matmul", batch_matmul.graph)
torch.register_batch_operator("mm", batch_mm.graph)
torch.register_batch_operator("select", batch_select.graph)
torch.register_batch_operator("where", batch_where.graph)
torch.register_batch_operator("update", batch_update.graph)
torch.register_batch_operator("any", batch_any.graph)
torch.register_batch_operator("any_false", batch_any_false.graph)
torch.register_batch_operator("type_as", batch_type_as.graph)
torch.register_batch_operator("gt", batch_gt.graph)
torch.register_batch_operator("batch_from_scalar_tensor", batch_from_scalar_tensor.graph)
