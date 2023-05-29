## Part 2 batching rule for `simple_mul`

```py
def simple_mul(x: Tensor, y: Tensor):
    return torch.mul(x, y)
```

### Batching rule for `simple_mul`

```py
def mul_batched(x: torch.Tensor, y: torch.Tensor, in_dims = (None, None)):
    """Performs the main mul operation"""
    x, y = mul_meta(x, y, in_dims)
    return torch.mul(x, y)

def mul_meta(x: torch.Tensor, y: torch.Tensor, in_dims = (None, None)):
    """The meta implementation handles shape and dimension"""
    # I am not sure whether it is correct to add a psuedo-batch dimension
    # of 1 when there are no batch dimension
    if in_dims[0] == None:
        # add a psuedo batch dimension if no batch dimension
        x = x.unsqueeze(0)
    elif in_dims[0] != 0:
        # make batch dimension as first dimension
        x = torch.moveaxis(x, in_dims[0], 0)
        
    if in_dims[1] == None:
        y = y.unsqueeze(0)
    elif in_dims[1] != 0:
        y = torch.moveaxis(y, in_dims[1], 0)
    
    x_shape, y_shape = x.size(), y.size()
    x_batch_dim, x_data_dim = x_shape[0], x_shape[1:]
    y_batch_dim, y_data_dim = y_shape[0], y_shape[1:]

    x_new_data_dim, y_new_data_dim = broadcast_shapes(list(x_data_dim), list(y_data_dim))
    x_new_dim = [x_batch_dim] + x_new_data_dim
    y_new_dim = [y_batch_dim] + y_new_data_dim

    x = x.reshape(x_new_dim)
    y = y.reshape(y_new_dim)

    return x, y


def broadcast_shapes(a_dim, b_dim):
    """Broadcast and returns new shape

    Example
    -------
    >>> broadcast_shapes([3], [4, 3])
    ([1, 3], [4, 3])
    >>> broadcast_shapes([15, 5], [15, 3, 5])
    ([15, 1, 5], [15, 3, 5])
    >>> broadcast_shapes([5, 4], [1])
    ([5, 4], [1, 1])
    """
    ab_swapped = 0
    if len(b_dim) < len(a_dim):
        # make sure that a has the least dimension
        a_dim, b_dim = b_dim, a_dim
        ab_swapped = 1
    
    new_a_dim = []
    a_ptr = 0

    for dim in b_dim:
        if a_ptr >= len(a_dim):
            new_a_dim.append(1)
        elif dim == a_dim[a_ptr]:
            new_a_dim.append(dim)
            a_ptr += 1
        else:
            new_a_dim.append(1)
    
    if ab_swapped:
        a_dim, b_dim = b_dim, new_a_dim
    else:
        a_dim = new_a_dim
    return a_dim, b_dim
```

```py
# checking results of mul_batched(in_dims, x, y) and vmap(simple_mul, in_dims)(x, y)
x = torch.randn(B)
y = torch.randn(B)

torch.allclose(vmap(simple_mul)(x, y), mul_batched(x, y))  # True

x = torch.randn(3)
y = torch.randn(B)

torch.allclose(vmap(op, in_dims=(None, 0))(x, y), mul_batched(x, y, (None, 0)))  # True

x = torch.randn(4, 3)
y = torch.randn(3, B)

torch.allclose(vmap(op, in_dims=(None, 1))(x, y), mul_batched(x, y, (None, 1)))  # True
```

I am not sure how to implement nested vmap via `mul_batched`
