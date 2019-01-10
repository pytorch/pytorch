import torch
from torch.jit import BatchTensor


# TODO: there are some commented raise statements
# when we support rasie exception in script, we want to check them
@torch.jit.script
def batch_tanh(data, mask, dims):
    data = torch.tanh(data)
    return data, mask, dims


@torch.jit.script
def batch_sigmoid(data, mask, dims):
    data = torch.sigmoid(data)
    return data, mask, dims


@torch.jit.script
def batch_relu(data, mask, dims):
    data = torch.relu(data)
    return data, mask, dims


@torch.jit.script
def batch_neg(data, mask, dims):
    data = torch.neg(data)
    return data, mask, dims


@torch.jit.script
def batch_neg_scalar(data):
    return torch.neg(data)


@torch.jit.script
def batch_add(data1, mask1, dims1, data2, mask2, dims2, alpha_):
    alpha = float(alpha_)
    data = torch.add(data1, data2, alpha=alpha)
    mask = mask1 * mask2
    dims = dims1.__or__(dims2)
    return data, mask, dims


@torch.jit.script
def batch_add_scalar(data, mask, dims, other, alpha_):
    alpha = float(alpha_)
    data = torch.add(data, other.type_as(data), alpha=alpha)
    return data, mask, dims


@torch.jit.script
def batch_sub(data1, mask1, dims1, data2, mask2, dims2, alpha_):
    alpha = float(alpha_)
    data = torch.sub(data1, data2, alpha=alpha)
    mask = mask1 * mask2
    dims = dims1.__or__(dims2)
    return data, mask, dims


@torch.jit.script
def batch_sub_scalar(data1, data2):
    return data1 - data2


@torch.jit.script
def batch_mul(data1, mask1, dims1, data2, mask2, dims2):
    data = torch.mul(data1, data2)
    mask = mask1 * mask2
    dims = dims1.__or__(dims2)
    return data, mask, dims


@torch.jit.script
def batch_mul_scalar(data1, data2):
    return data1 * data2


@torch.jit.script
def batch_div(data, mask, dims, other):  # div(batchtensor, scalar)
    data = torch.div(data, other)
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
def batch_select(data, mask, dims, dim_, index_):
    dim = int(dim_)
    index = int(index_)
    # if dim == 0:
    #     raise ValueError("Cannot select 0 dim in BatchTensor")
    data = data.select(dim, index)
    if bool(dims[dim - 1]):
        mask = mask.select(dim, index)
    else:
        mask = mask.select(dim, 0)
    dims = torch.cat((dims[:dim - 1], dims[dim:dims.size(0)]))
    return data, mask, dims


@torch.jit.script
def batch_fmod(data, mask, dims, other_):
    other = int(other_)
    data = torch.fmod(data, other)
    return data, mask, dims


@torch.jit.script
def batch_zeros_like(data, mask, dims):
    res_data = torch.zeros_like(data)
    return res_data, mask, dims


@torch.jit.script
def batch_index_select(data, mask, dims, dim_, index_data, index_mask, index_dims):
    dim = int(dim_)
    # if dim == 0:
    #     raise ValueError("Cannot index_select along 0 dim in BatchTensor")
    batch_size = data.size(0)  # TODO maybe index_mask will be used at some point
    res_data = torch.zeros([0])
    res_mask = torch.zeros([0])
    for i in range(batch_size):
        d = data[i].index_select(dim - 1, index_data[i]).unsqueeze(0)
        if bool(dims[dim - 1]):
            m = mask[i].index_select(dim - 1, index_data[i]).unsqueeze(0)
        else:
            m = mask[i].unsqueeze(0)
        if i == 0:
            res_data = d
            res_mask = m
        else:
            res_data = torch.cat((res_data, d), 0)
            res_mask = torch.cat((res_mask, m), 0)
    return res_data, res_mask, dims


@torch.jit.script
def batch_view_as(data, mask, dims, data1, mask1, dims1):
    # if data.size(0) != data1.size(0):
    #     raise ValueError("In view_as, tensor and target tensor should have the same batch_size")
    # if not torch.equal(dims, dims1):
    #     raise ValueError("In batched view_as, dims and target dims should be the same")
    data = data.view_as(data1)
    mask = mask.view_as(mask1)
    dims = dims1
    return data, mask, dims


# assume data, data1, data2 have same size
@torch.jit.script
def batch_where(data, mask, dims, data1, mask1, dims1, data2, mask2, dims2):
    data = data * mask.type_as(data)
    cond_data = data
    cond_mask = data
    if data.dim() == 1:
        for _ in range(data1.dim() - 1):
            data = data.unsqueeze(data.dim())
        cond_data = data.expand_as(data1)
        cond_mask = data.expand_as(mask1)
    res_data = torch.where(cond_data, data1, data2)
    res_mask = torch.where(cond_mask, mask1, mask2)
    res_dims = dims1.__or__(dims2)
    return res_data, res_mask, res_dims


@torch.jit.script
def batch_where_scalar(cond, data1, mask1, dims1, data2, mask2, dims2):
    cond = torch.zeros([1], dtype=torch.uint8)
    res_data = torch.where(cond, data1, data2)
    res_mask = torch.where(cond, mask1, mask2)
    res_dims = torch.where(cond, dims1, dims2)
    return res_data, res_mask, res_dims


@torch.jit.script
def batch_update(batch_data, batch_mask, batch_dims, new_data, new_mask, new_dims):
    data = torch.where(new_mask, new_data, batch_data)
    return data, new_mask, new_dims  # TODO: consider whether return new_mask and new_dims


@torch.jit.script
def batch_any(data, mask, dims):
    return torch.gt(torch.sum(data * mask), 0)


@torch.jit.script
def batch_type_as(data, mask, dims, data1, mask1, dims1):
    return data.type_as(data1), mask, dims


@torch.jit.script
def batch_gt(data, mask, dims, data1, mask1, dims1):
    return torch.gt(data, data1), mask * mask1, dims.__or__(dims1)


@torch.jit.script
def batch_gt_scalar(data1, data2):
    return torch.gt(data1, data2)


@torch.jit.script
def batch_gt_one_scalar(data, mask, dims, other_):
    other = float(other_)
    return torch.gt(data, other), mask, dims


@torch.jit.script
def batch_lt(data, mask, dims, data1, mask1, dims1):
    return torch.lt(data, data1), mask * mask1, dims.__or__(dims1)


@torch.jit.script
def batch_eq(data, mask, dims, data1, mask1, dims1):
    return torch.eq(data, data1), mask * mask1, dims.__or__(dims1)


@torch.jit.script
def batch_size(data, mask, dims, dim_):
    dim = int(dim_)
    return data.size(dim)


@torch.jit.script
def batch_dim(data, mask, dims):
    return data.dim()


@torch.jit.script
def batch_squeeze(data, mask, dims, dim_):
    if int(dim_) < 0:
        dim_ = dim_ + data.dim()
    dim = int(dim_)
    # if dim == 0:
    #     raise ValueError("cannot do squeeze along batch_dim")
    data = data.squeeze(dim)
    mask = mask.squeeze(dim)
    dims = torch.cat((dims[:dim - 1], dims[dim:dims.size(0)]))
    return data, mask, dims


@torch.jit.script
def batch_unsqueeze(data, mask, dims, dim_):
    if int(dim_) < 0:
        dim_ = dim_ + data.dim() + 1
    dim = int(dim_)
    # if dim == 0:
    #     raise ValueError("cannot do unsqueeze along batch_dim")
    data = data.unsqueeze(dim)
    mask = mask.unsqueeze(dim)
    dims = torch.cat((dims[:dim], torch.zeros([1], dtype=torch.uint8), dims[dim:dims.size(0)]))
    return data, mask, dims


@torch.jit.script
def batch_argmax(data, mask, dims, dim_, keepdim_):
    dim = int(dim_)
    keepdim = bool(keepdim_)
    # if dim == 0:
    #     raise ValueError("cannot do argmax along batch_dim")
    batch_size = data.size(0)
    res_data = torch.zeros([0])
    for i in range(batch_size):
        if bool(dims[dim - 1]):
            if dim - 1 != 0:
                m = mask[i].transpose(0, dim - 1)
            else:
                m = mask[i]
            valid_num = m.sum(0, keepdim=True)
            while(valid_num.dim() >= 1):
                valid_num = valid_num[0]
            d = data[i].unsqueeze(0).narrow(dim, 0, int(valid_num))
        else:
            d = data[i].unsqueeze(0)
        d = d.argmax(dim, keepdim)
        if i == 0:
            res_data = d
        else:
            res_data = torch.cat([res_data, d], 0)
    if keepdim:
        mask = mask
    else:
        mask = mask.select(dim, 0)
        dims = torch.cat((dims[:dim - 1], dims[dim:dims.size(0)]))
    return res_data, mask, dims


@torch.jit.script
def batch_topk(data, mask, dims, k_, dim_, largest_, sorted_):
    k = int(k_)
    dim = int(dim_)
    largest = bool(largest_)
    sorted = bool(sorted_)
    # if dim == 0:
    #     raise ValueError("cannot do topk along batch_dim")
    batch_size = data.size(0)
    res_data = torch.zeros([0])
    res_index = torch.zeros([0])
    for i in range(batch_size):
        if bool(dims[dim - 1]):
            if dim - 1 != 0:
                m = mask[i].transpose(0, dim - 1)
            else:
                m = mask[i]
            valid_num = m.sum(0, keepdim=True)
            while(valid_num.dim() >= 1):
                valid_num = valid_num[0]
            d = data[i].unsqueeze(0).narrow(dim, 0, int(valid_num))
        else:
            d = data[i].unsqueeze(0)
        d, idx = d.topk(k, dim, largest, sorted)
        if i == 0:
            res_data = d
            res_index = idx
        else:
            res_data = torch.cat([res_data, d], 0)
            res_index = torch.cat([res_index, idx], 0)
    if bool(dims[dim - 1]):
        mask = mask.narrow(dim, 0, k)
    return res_data, mask, dims, res_index, mask, dims


@torch.jit.script
def batch_softmax(data, mask, dims, dim_):
    dim = int(dim_)
    # if dim == 0:
    #     raise ValueError("cannot do softmax along batch_dim")
    batch_size = data.size(0)
    max_len = data.size(dim)
    res_data = torch.zeros([0])
    for i in range(batch_size):
        if bool(dims[dim - 1]):
            if dim - 1 != 0:
                m = mask[i].transpose(0, dim - 1)
            else:
                m = mask[i]
            valid_num = m.sum(0, keepdim=True)
            while(valid_num.dim() >= 1):
                valid_num = valid_num[0]
            valid_num = int(valid_num)
            d = data[i].unsqueeze(0).narrow(dim, 0, valid_num).softmax(dim)
            if valid_num < max_len:
                d = torch.cat([d, data[i].unsqueeze(0).narrow(dim, valid_num, max_len - valid_num)], dim)
        else:
            d = data[i].unsqueeze(0).softmax(dim)
        if i == 0:
            res_data = d
        else:
            res_data = torch.cat([res_data, d], 0)
    return res_data, mask, dims


# size argument in dynamic dimension has to be -1
# in static dimension, size has to be specified, -1 is not supported
@torch.jit.script
def batch_view(data, mask, dims, sizes):
    batch_size = data.size(0)
    # if(sizes[0] != batch_size and sizes[0] != -1 and sizes[0] != 1):
    #     raise "first dim in view must be 1, -1, or batch size"
    # for i in range(dims.size(0)):
    #     if dims[0] == 1 and sizes[i + 1] != -1:
    #         raise "size argument in dynamic dimension has to be -1"
    sizes = sizes.type_as(torch.ones([1], dtype=torch.int))
    data_sizes_ = torch.cat([torch.ones([1], dtype=torch.int) * batch_size, sizes.narrow(0, 1, sizes.size(0) - 1)], 0)
    data_sizes = data_sizes_._tensor_to_list()
    res_data = data.view(data_sizes)
    mask_sizes_ = data_sizes_.narrow(0, 0, 1)
    res_dims = data_sizes_.narrow(0, 0, 1)
    for i_ in range(sizes.size(0) - 1):
        i = i_ + 1
        if bool(sizes[i] == -1):
            cur_size_ = mask.size(i)
            cur_dim = 1
        else:
            cur_size_ = 1
            cur_dim = 0
        mask_sizes_ = torch.cat([mask_sizes_, torch.ones([1], dtype=torch.int) * cur_size_])
        res_dims = torch.cat([res_dims, torch.ones([1], dtype=torch.int) * cur_dim])
    mask_sizes = mask_sizes_._tensor_to_list()
    res_mask = mask.view(mask_sizes)
    return res_data, res_mask, res_dims.narrow(0, 1, res_dims.size(0) - 1).type_as(dims)


@torch.jit.script
def batch_cat2(data1, mask1, dims1, data2, mask2, dims2, dim_):
    dim = int(dim_)
    data = torch.cat([data1, data2], dim)
    if bool(dims1[dim - 1]):
        mask = torch.cat([mask1, mask2], dim)
    else:
        mask = mask1
    return data, mask, dims1


@torch.jit.script
def batch_cat3(data1, mask1, dims1, data2, mask2, dims2, data3, mask3, dims3, dim_):
    dim = int(dim_)
    data = torch.cat([data1, data2, data3], dim)
    if bool(dims1[dim - 1]):
        mask = torch.cat([mask1, mask2, mask3], dim)
    else:
        mask = mask1
    return data, mask, dims1


@torch.jit.script
def batch_narrow(data, mask, dims, dimension_, start_, length_):
    dimension = int(dimension_)
    start = int(start_)
    length = int(length_)
    # if dimension == 0:
    #     raise ValueError("cannot do narrow along batch_dim")
    data = data.narrow(dimension, start, length)
    if bool(dims[dimension - 1]):
        mask = mask.narrow(dimension, start, length)
    else:
        mask = mask.narrow(dimension, 0, 1)
    return data, mask, dims


@torch.jit.script
def batch_sum(data, mask, dims):
    data = data * mask.type_as(data)
    for _ in range(dims.size(0)):
        data = data.sum(1)
    mask = torch.ones([data.size(0)], dtype=torch.uint8)
    dims = dims[:0]  # empty tensor
    return data, mask, dims


@torch.jit.script
def batch_from_scalar_tensor(data):
    data = data.unsqueeze(0)
    mask = torch.ones([1], dtype=torch.uint8)
    dims = torch.zeros([0], dtype=torch.uint8)
    return data, mask, dims

torch.register_batch_operator("tanh", batch_tanh.graph)
torch.register_batch_operator("sigmoid", batch_sigmoid.graph)
torch.register_batch_operator("relu", batch_relu.graph)
torch.register_batch_operator("neg", batch_neg.graph)
torch.register_batch_operator("neg", batch_neg_scalar.graph)
torch.register_batch_operator("add", batch_add.graph)
torch.register_batch_operator("add", batch_add_scalar.graph)
torch.register_batch_operator("sub", batch_sub.graph)
torch.register_batch_operator("sub", batch_sub_scalar.graph)
torch.register_batch_operator("mul", batch_mul.graph)
torch.register_batch_operator("mul", batch_mul_scalar.graph)
torch.register_batch_operator("div", batch_div.graph)
torch.register_batch_operator("matmul", batch_matmul.graph)
torch.register_batch_operator("mm", batch_mm.graph)
torch.register_batch_operator("fmod", batch_fmod.graph)
torch.register_batch_operator("zeros_like", batch_zeros_like.graph)
torch.register_batch_operator("select", batch_select.graph)
torch.register_batch_operator("index_select", batch_index_select.graph)
torch.register_batch_operator("view_as", batch_view_as.graph)
torch.register_batch_operator("where", batch_where.graph)
torch.register_batch_operator("where", batch_where_scalar.graph)
torch.register_batch_operator("update", batch_update.graph)
torch.register_batch_operator("any", batch_any.graph)
torch.register_batch_operator("type_as", batch_type_as.graph)
torch.register_batch_operator("gt", batch_gt.graph)
torch.register_batch_operator("gt", batch_gt_scalar.graph)
torch.register_batch_operator("gt", batch_gt_one_scalar.graph)
torch.register_batch_operator("lt", batch_lt.graph)
torch.register_batch_operator("eq", batch_eq.graph)
torch.register_batch_operator("size", batch_size.graph)
torch.register_batch_operator("dim", batch_dim.graph)
torch.register_batch_operator("squeeze", batch_squeeze.graph)
torch.register_batch_operator("unsqueeze", batch_unsqueeze.graph)
torch.register_batch_operator("argmax", batch_argmax.graph)
torch.register_batch_operator("topk", batch_topk.graph)
torch.register_batch_operator("softmax", batch_softmax.graph)
torch.register_batch_operator("view", batch_view.graph)
torch.register_batch_operator("cat", batch_cat2.graph)
torch.register_batch_operator("cat", batch_cat3.graph)
torch.register_batch_operator("narrow", batch_narrow.graph)
torch.register_batch_operator("sum", batch_sum.graph)
torch.register_batch_operator("batch_from_scalar_tensor", batch_from_scalar_tensor.graph)
