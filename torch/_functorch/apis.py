# NOTE: We allow Dynamo to see this file (via torch/_dynamo/skipfiles.py) so that it can
#       trace through `grad`.
#       Currently, we can't allow Dynamo to see `eager_transforms.py` as that break a lot of thing
#       and there isn't a mechanism to selectively expose only some functions (eg. grad) from a file
#       to Dynamo.
from torch._functorch.vmap import vmap_impl, _check_randomness_arg, exposed_in, Callable, in_dims_t, out_dims_t
import functools

# vmap(func)(inputs) wraps all Tensor inputs to be batched in BatchedTensors,
# sends those into func, and then unwraps the output BatchedTensors. Operations
# on BatchedTensors perform the batched operations that the user is asking for.
#
# vmap's randomness behavior differs from JAX's, which would require a PRNG key
# to be passed everywhere.

@exposed_in('torch.func')
def vmap(
        func: Callable,
        in_dims: in_dims_t = 0,
        out_dims: out_dims_t = 0,
        randomness: str = 'error',
        *,
        chunk_size=None) -> Callable:
    """
    vmap is the vectorizing map; ``vmap(func)`` returns a new function that
    maps ``func`` over some dimension of the inputs. Semantically, vmap
    pushes the map into PyTorch operations called by ``func``, effectively
    vectorizing those operations.

    vmap is useful for handling batch dimensions: one can write a function
    ``func`` that runs on examples and then lift it to a function that can
    take batches of examples with ``vmap(func)``. vmap can also be used to
    compute batched gradients when composed with autograd.

    .. note::
        :func:`torch.vmap` is aliased to :func:`torch.func.vmap` for
        convenience. Use whichever one you'd like.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. ``in_dims`` should have a
            structure like the inputs. If the ``in_dim`` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If ``out_dims`` is a Tuple, then
            it should have one element per output. Default: 0.
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.
        chunk_size (None or int): If None (default), apply a single vmap over inputs.
            If not None, then compute the vmap :attr:`chunk_size` samples at a time.
            Note that :attr:`chunk_size=1` is equivalent to computing the vmap with a for-loop.
            If you run into memory issues computing the vmap, please try a non-None chunk_size.

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        ``func``, except each input has an extra dimension at the index
        specified by ``in_dims``. It takes returns the same outputs as
        ``func``, except each output has an extra dimension at the index
        specified by ``out_dims``.

    .. warning:
        :func:`vmap` works best with functional-style code. Please do not
        perform any side-effects in ``func``, with the exception of
        in-place PyTorch operations. Examples of side-effects include mutating
        Python data structures and assigning values to variables not captured
        in ``func``.

    One example of using :func:`vmap` is to compute batched dot products. PyTorch
    doesn't provide a batched ``torch.dot`` API; instead of unsuccessfully
    rummaging through docs, use :func:`vmap` to construct a new function.

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.func.vmap(torch.dot)  # [N, D], [N, D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)

    :func:`vmap` can be helpful in hiding batch dimensions, leading to a simpler
    model authoring experience.

        >>> batch_size, feature_size = 3, 5
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>>
        >>> def model(feature_vec):
        >>>     # Very simple linear model with activation
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> examples = torch.randn(batch_size, feature_size)
        >>> result = torch.vmap(model)(examples)

    :func:`vmap` can also help vectorize computations that were previously difficult
    or impossible to batch. One example is higher-order gradient computation.
    The PyTorch autograd engine computes vjps (vector-Jacobian products).
    Computing a full Jacobian matrix for some function f: R^N -> R^N usually
    requires N calls to ``autograd.grad``, one per Jacobian row. Using :func:`vmap`,
    we can vectorize the whole computation, computing the Jacobian in a single
    call to ``autograd.grad``.

        >>> # Setup
        >>> N = 5
        >>> f = lambda x: x ** 2
        >>> x = torch.randn(N, requires_grad=True)
        >>> y = f(x)
        >>> I_N = torch.eye(N)
        >>>
        >>> # Sequential approach
        >>> jacobian_rows = [torch.autograd.grad(y, x, v, retain_graph=True)[0]
        >>>                  for v in I_N.unbind()]
        >>> jacobian = torch.stack(jacobian_rows)
        >>>
        >>> # vectorized gradient computation
        >>> def get_vjp(v):
        >>>     return torch.autograd.grad(y, x, v)
        >>> jacobian = torch.vmap(get_vjp)(I_N)

    :func:`vmap` can also be nested, producing an output with multiple batched dimensions

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.vmap(torch.vmap(torch.dot))  # [N1, N0, D], [N1, N0, D] -> [N1, N0]
        >>> x, y = torch.randn(2, 3, 5), torch.randn(2, 3, 5)
        >>> batched_dot(x, y) # tensor of size [2, 3]

    If the inputs are not batched along the first dimension, ``in_dims`` specifies
    the dimension that each inputs are batched along as

        >>> torch.dot                            # [N], [N] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)   # output is [5] instead of [2] if batched along the 0th dimension

    If there are multiple inputs each of which is batched along different dimensions,
    ``in_dims`` must be a tuple with the batch dimension for each input as

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=(0, None))  # [N, D], [D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> batched_dot(x, y) # second arg doesn't have a batch dim because in_dim[1] was None

    If the input is a Python struct, ``in_dims`` must be a tuple containing a struct
    matching the shape of the input:

        >>> f = lambda dict: torch.dot(dict['x'], dict['y'])
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> input = {'x': x, 'y': y}
        >>> batched_dot = torch.vmap(f, in_dims=({'x': 0, 'y': None},))
        >>> batched_dot(input)

    By default, the output is batched along the first dimension. However, it can be batched
    along any dimension by using ``out_dims``

        >>> f = lambda x: x ** 2
        >>> x = torch.randn(2, 5)
        >>> batched_pow = torch.vmap(f, out_dims=1)
        >>> batched_pow(x) # [5, 2]

    For any function that uses kwargs, the returned function will not batch the kwargs but will
    accept kwargs

        >>> x = torch.randn([2, 5])
        >>> def fn(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> batched_pow = torch.vmap(fn)
        >>> assert torch.allclose(batched_pow(x), x * 4)
        >>> batched_pow(x, scale=x) # scale is not batched, output has shape [2, 2, 5]

    .. note::
        vmap does not provide general autobatching or handle variable-length
        sequences out of the box.
    """
    _check_randomness_arg(randomness)
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError(f"vmap: chunk_size should be None or greater than 0. (got {chunk_size})")

    # @functools.wraps(func)
    def wrapped(*args, **kwargs):
        return vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs)

    return wrapped
