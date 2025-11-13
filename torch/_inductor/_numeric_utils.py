from functools import reduce

import torch
from torch.utils._ordered_set import OrderedSet


@torch.compile(dynamic=True)
@torch._dynamo.dont_skip_tracing
def inductor_fma(a, b, c):
    torch._check(a.shape == b.shape == c.shape)
    return torch._inductor.inductor_prims.fma(b, c, a)
    return a + b * c


def fma(a, b, c):
    if type(a) is int and a == 0:
        return b * c
    if isinstance(b, torch._subclasses.fake_tensor.FakeTensor):
        return torch._inductor.inductor_prims.fma(b, c, a)
    else:
        a_shape = a.shape
        a, b, c = torch.broadcast_tensors(a, b, c)
        a = a.reshape(-1).contiguous()
        b = b.reshape(-1).contiguous()
        c = c.reshape(-1).contiguous()
        a = inductor_fma(a, b, c)
        return a.reshape(*a_shape)


def linear_fma_sum(a, b):
    a, b = torch.broadcast_tensors(a, b)
    a = a.unbind(-1)
    b = b.unbind(-1)
    return reduce(lambda a, bc: fma(a, bc[0], bc[1]), zip(a, b), 0)


def linear_sum(a):
    a = a.unbind(-1)
    return sum(a, 0)


def hierarchical_sum(a):
    return reduce(lambda a, _: linear_sum(a), range(a.ndim - 1), a).unsqueeze(-1)


def hierarchical_fma_sum(a, b):
    return reduce(
        lambda a, _: linear_fma_sum(*a) if type(a) is tuple else linear_sum(a),
        range(a.ndim - 1),
        (a, b),
    ).unsqueeze(-1)


def ordered_reshape(a, order):
    # make work when size is less than order!
    flat_order = [e for o in order for e in (o if type(o) is tuple else [o])]
    input_shape = [a.shape[0], -1] + [2] * len(flat_order)
    permute = [0, 1] + [
        len(input_shape) - i - 1
        for i in sorted(
            range(len(flat_order)), key=flat_order.__getitem__, reverse=True
        )
    ]
    # this is the tricky part...
    # ((1, 4, 2), 16, 8)
    # mode
    #   0  2  1   4   3
    # then reverse and flip and offset
    output_shape = [a.shape[0], -1] + list(
        reversed([2 if type(o) is int else 2 ** len(o) for o in order])
    )
    a = a.reshape(*input_shape)
    a = a.permute(*permute)
    a = a.reshape(*output_shape)
    return a


def ordered_sum_2d(a, order):
    return hierarchical_sum(ordered_reshape(a, order))


def ordered_fma_sum_2d(a, b, order):
    a, b = torch.broadcast_tensors(a, b)
    return hierarchical_fma_sum(ordered_reshape(a, order), ordered_reshape(b, order))


def ordered_sum(a, dim=None, keepdim=False, order=None):
    if order is None:
        return a.sum(dim=dim, keepdim=keepdim)
    if dim is None:
        dim = list(range(a.ndim))
    if type(dim) is int:
        dim = [dim]
    dim = [(d + a.ndim) % a.ndim for d in dim]
    other_dim = [d for d in range(a.ndim) if d not in dim]
    size = reduce(lambda a, b: a * b, [a.shape[d] for d in dim], 1)
    other_size = reduce(lambda a, b: a * b, [a.shape[d] for d in other_dim], 1)
    a_shape = a.shape
    a = a.permute(*other_dim, *dim)
    a = a.reshape(*other_size, *size)
    r = ordered_sum_2d(a, order)
    if keepdim:
        return r.reshape(*[1 if i in dim else a for i, a in enumerate(a_shape)])
    else:
        return r.reshape(*[a for i, a in enumerate(a_shape) if i not in dim])


def ordered_fma_sum(a, b, dim=None, keepdim=False, order=None):
    a, b = torch.broadcast_tensors(a, b)
    if order is None:
        return (a * b).sum(dim=dim, keepdim=keepdim)
    if dim is None:
        dim = list(range(a.ndim))
    if type(dim) is int:
        dim = [dim]
    dim = [(d + a.ndim) % a.ndim for d in dim]
    other_dim = [d for d in range(a.ndim) if d not in dim]
    size = reduce(lambda a, b: a * b, [a.shape[d] for d in dim], 1)
    other_size = reduce(lambda a, b: a * b, [a.shape[d] for d in other_dim], 1)
    a_shape = a.shape
    a = a.permute(*other_dim, *dim)
    a = a.reshape(other_size, size)
    b = b.permute(*other_dim, *dim)
    b = b.reshape(other_size, size)
    r = ordered_fma_sum_2d(a, b, order)
    if keepdim:
        return r.reshape(*[1 if i in dim else a for i, a in enumerate(a_shape)])
    else:
        return r.reshape(*[a for i, a in enumerate(a_shape) if i not in dim])


def flatten_to_tuple(order):
    if type(order) is int:
        return (order,)
    return tuple([e for o in order for e in flatten_to_tuple(o)])


def flatten_to_tuple_or_int(order):
    order = flatten_to_tuple(order)
    return order[0] if len(order) == 1 else order


def normalize_order(order):
    return [flatten_to_tuple_or_int(o) for o in order]


def extend_order(order, new_stride=None):  # presume currently that all sizes are two
    if new_stride is None:
        flat_order = [e for o in order for e in ([o] if type(o) is int else o)]
        new_stride = max(flat_order) * 2 if flat_order else 1
    if type(order) is int:
        return [(order, new_stride), (new_stride, order)]
    order = list(order)
    result = [order[:at] + [new_stride] + order[at:] for at in range(len(order) + 1)]
    result += [
        order[:at] + [new_order] + order[at + 1 :]
        for at in range(len(order))
        for new_order in extend_order(order[at], new_stride)
    ]
    result = list(OrderedSet([tuple(normalize_order(o)) for o in result]))
    return [list(e) for e in result]


def get_inputs_for_nnz(dtype, input_shapes, reduction_dims, nnz):
    torch.manual_seed(0x81600F)
    # args = [torch.randn(*shape, dtype=dtype, device='cuda') for shape in input_shapes]
    args = [
        torch.testing.make_tensor(*shape, dtype=dtype, device="cuda")
        for shape in input_shapes
    ]
    with torch.no_grad():
        assert len(input_shapes) == len(reduction_dims)
        for idx, rd in enumerate(reduction_dims):
            if rd is None:
                continue
            slicer = tuple(
                [
                    slice(None)
                    if i != (rd + args[idx].ndim) % args[idx].ndim
                    else slice(nnz, None)
                    for i in range(args[idx].ndim)
                ]
            )
            args[idx][slicer].zero_()
    return args


def find_failing_nnz(fn, model, dtype, input_shapes, reduction_dims):
    reduction_size = next(
        iter(
            [
                input_shapes[idx][reduction_dims[idx]]
                for idx in range(len(input_shapes))
                if reduction_dims[idx] is not None
            ]
        )
    )
    nnz = 1
    while nnz <= reduction_size:
        args = get_inputs_for_nnz(dtype, input_shapes, reduction_dims, nnz)
        ref = fn(*args)
        mod = model(*args)
        if not torch.equal(ref, mod):
            return nnz
        nnz *= 2
    return None


def find_order(fn, model, dtype, input_shapes, reduction_dims):
    reduction_size = next(
        iter(
            [
                input_shapes[idx][reduction_dims[idx]]
                for idx in range(len(input_shapes))
                if reduction_dims[idx] is not None
            ]
        )
    )
    nnz = 1
    order = []
    while nnz < reduction_size:
        nnz *= 2
        args = get_inputs_for_nnz(dtype, input_shapes, reduction_dims, nnz)
        ref = fn(*args)
        good_orders = []
        for new_order in extend_order(order):
            mod = model(*args, order=new_order)
            good = torch.equal(ref, mod)
            if good:
                good_orders.append(new_order)
        if len(good_orders) == 0:
            return nnz, order
        assert len(good_orders) == 1
        order = good_orders[0]
    return None, order
