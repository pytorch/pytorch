from typing import Callable, Union, Tuple, List, Any, Optional
import torch._functorch.eager_transforms as _impl
import torch._functorch.vmap as _vmap_impl
from torch._functorch.vmap import in_dims_t, out_dims_t


# This file is for API declarations for the torch.func module.
# We need to re-declare the APIs (as opposed to importing everything from eager_transforms)
# due to PyTorch's public API rules: torch._functorch.eager_transforms.grad should
# not be public, but torch.func.grad is.
#
# If the implementation details of the API are more than a couple of lines long
# or rely on helper functions, please put the implementation in one of the
# imported files and call it from here.

__all__ = [
    'jvp',
    'grad',
    'grad_and_value',
    'vjp',
    'jacrev',
    'jvp',
    'jacfwd',
    'hessian',
    'functionalize',
]

argnums_t = Union[int, Tuple[int, ...]]


def vjp(func: Callable, *primals, has_aux: bool = False):
    """
    Standing for the vector-Jacobian product, returns a tuple containing the
    results of :attr:`func` applied to :attr:`primals` and a function that, when
    given ``cotangents``, computes the reverse-mode Jacobian of :attr:`func` with
    respect to :attr:`primals` times ``cotangents``.

    Args:
        func (Callable): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        primals (Tensors): Positional arguments to :attr:`func` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        has_aux (bool): Flag indicating that :attr:`func` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, vjp_fn)`` tuple containing the output of :attr:`func`
        applied to :attr:`primals` and a function that computes the vjp of
        :attr:`func` with respect to all :attr:`primals` using the cotangents passed
        to the returned function. If ``has_aux is True``, then instead returns a
        ``(output, vjp_fn, aux)`` tuple.
        The returned ``vjp_fn`` function will return a tuple of each VJP.

    When used in simple cases, :func:`vjp` behaves the same as :func:`grad`

        >>> x = torch.randn([5])
        >>> f = lambda x: x.sin().sum()
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> grad = vjpfunc(torch.tensor(1.))[0]
        >>> assert torch.allclose(grad, torch.func.grad(f)(x))

    However, :func:`vjp` can support functions with multiple outputs by
    passing in the cotangents for each of the outputs

        >>> x = torch.randn([5])
        >>> f = lambda x: (x.sin(), x.cos())
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc((torch.ones([5]), torch.ones([5])))
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    :func:`vjp` can even support outputs being Python structs

        >>> x = torch.randn([5])
        >>> f = lambda x: {'first': x.sin(), 'second': x.cos()}
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> cotangents = {'first': torch.ones([5]), 'second': torch.ones([5])}
        >>> vjps = vjpfunc(cotangents)
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    The function returned by :func:`vjp` will compute the partials with
    respect to each of the :attr:`primals`

        >>> x, y = torch.randn([5, 4]), torch.randn([4, 5])
        >>> (_, vjpfunc) = torch.func.vjp(torch.matmul, x, y)
        >>> cotangents = torch.randn([5, 5])
        >>> vjps = vjpfunc(cotangents)
        >>> assert len(vjps) == 2
        >>> assert torch.allclose(vjps[0], torch.matmul(cotangents, y.transpose(0, 1)))
        >>> assert torch.allclose(vjps[1], torch.matmul(x.transpose(0, 1), cotangents))

    :attr:`primals` are the positional arguments for :attr:`f`. All kwargs use their
    default value

        >>> x = torch.randn([5])
        >>> def f(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc(torch.ones_like(x))
        >>> assert torch.allclose(vjps[0], torch.full(x.shape, 4.))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``vjp``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``vjp(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``vjp`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     vjp(f)(x)

        In this case, ``vjp`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``vjp`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    return _impl.vjp(func, *primals, has_aux=has_aux)


def jacrev(func: Callable, argnums: Union[int, Tuple[int]] = 0, *, has_aux=False):
    """
    Computes the Jacobian of :attr:`func` with respect to the arg(s) at index
    :attr:`argnum` using reverse mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that :attr:`func` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a function that takes in the same inputs as :attr:`func` and
        returns the Jacobian of :attr:`func` with respect to the arg(s) at
        :attr:`argnums`. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by :attr:`func`.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacrev, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from torch.func import jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using :attr:`argnums`:

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to :attr:`argnums` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``jacrev``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     jacrev(f)(x)

        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``jacrev`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    return _impl.jacrev(func, argnums, has_aux=has_aux)


def jvp(func: Callable, primals: Any, tangents: Any, *, strict=False, has_aux: bool = False):
    """
    Standing for the Jacobian-vector product, returns a tuple containing
    the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at
    ``primals``" times ``tangents``. This is also known as forward-mode autodiff.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        primals (Tensors): Positional arguments to :attr:`func` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        tangents (Tensors): The "vector" for which Jacobian-vector-product is
            computed. Must be the same structure and sizes as the inputs to
            ``func``.
        has_aux (bool): Flag indicating that :attr:`func` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, jvp_out)`` tuple containing the output of ``func``
        evaluated at ``primals`` and the Jacobian-vector product.
        If ``has_aux is True``, then instead returns a ``(output, jvp_out, aux)`` tuple.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.

    jvp is useful when you wish to compute gradients of a function R^1 -> R^N

        >>> from torch.func import jvp
        >>> x = torch.randn([])
        >>> f = lambda x: x * torch.tensor([1., 2., 3])
        >>> value, grad = jvp(f, (x,), (torch.tensor(1.),))
        >>> assert torch.allclose(value, f(x))
        >>> assert torch.allclose(grad, torch.tensor([1., 2, 3]))

    :func:`jvp` can support functions with multiple inputs by passing in the
    tangents for each of the inputs

         >>> from torch.func import jvp
         >>> x = torch.randn(5)
         >>> y = torch.randn(5)
         >>> f = lambda x, y: (x * y)
         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
         >>> assert torch.allclose(output, x + y)

    """
    return _impl.jvp(func, primals, tangents, strict=strict, has_aux=has_aux)


def jacfwd(func: Callable, argnums: argnums_t = 0, has_aux: bool = False, *, randomness: str = "error"):
    """
    Computes the Jacobian of :attr:`func` with respect to the arg(s) at index
    :attr:`argnum` using forward-mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that :attr:`func` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        randomness(str): Flag indicating what type of randomness to use.
            See :func:`vmap` for more detail. Allowed: "different", "same", "error".
            Default: "error"

    Returns:
        Returns a function that takes in the same inputs as :attr:`func` and
        returns the Jacobian of :attr:`func` with respect to the arg(s) at
        :attr:`argnums`. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by :attr:`func`.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use :func:`jacrev`, which has better operator coverage.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>> jacobian = jacfwd(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacfwd` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacfwd, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacfwd(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacfwd(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`
    to produce Hessians

        >>> from torch.func import jacfwd, jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacfwd` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using :attr:`argnums`:

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to :attr:`argnums` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    """
    return _impl.jacfwd(func, argnums, has_aux=has_aux, randomness=randomness)


def hessian(func, argnums=0):
    """
    Computes the Hessian of :attr:`func` with respect to the arg(s) at index
    :attr:`argnum` via a forward-over-reverse strategy.

    The forward-over-reverse strategy (composing ``jacfwd(jacrev(func))``) is
    a good default for good performance. It is possible to compute Hessians
    through other compositions of :func:`jacfwd` and :func:`jacrev` like
    ``jacfwd(jacfwd(func))`` or ``jacrev(jacrev(func))``.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Hessian with respect to.
            Default: 0.

    Returns:
        Returns a function that takes in the same inputs as :attr:`func` and
        returns the Hessian of :attr:`func` with respect to the arg(s) at
        :attr:`argnums`.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use ``jacrev(jacrev(func))``, which has better
        operator coverage.

    A basic usage with a R^N -> R^1 function gives a N x N Hessian:

        >>> from torch.func import hessian
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hess, torch.diag(-x.sin()))

    """
    return jacfwd(jacrev(func, argnums), argnums)


def grad_and_value(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    """
    Returns a function to compute a tuple of the gradient and primal, or
    forward, computation.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified :attr:`has_aux`
            equals ``True``, function can return a tuple of single-element
            Tensor and other auxiliary objects: ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients
            with respect to. :attr:`argnums` can be single integer or tuple of
            integers. Default: 0.
        has_aux (bool): Flag indicating that :attr:`func` returns a tensor and
            other auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute a tuple of gradients with respect to its inputs
        and the forward computation. By default, the output of the function is
        a tuple of the gradient tensor(s) with respect to the first argument
        and the primal computation. If specified :attr:`has_aux` equals
        ``True``, tuple of gradients and tuple of the forward computation with
        output auxiliary objects is returned. If :attr:`argnums` is a tuple of
        integers, a tuple of a tuple of the output gradients with respect to
        each :attr:`argnums` value and the forward computation is returned.

    See :func:`grad` for examples
    """
    return _impl.grad_and_value(func, argnums, has_aux)


def grad(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    """``grad`` operator helps computing gradients of :attr:`func` with respect to the
    input(s) specified by :attr:`argnums`. This operator can be nested to
    compute higher-order gradients.

    Args:
        func (Callable): A Python function that takes one or more arguments.
            Must return a single-element Tensor. If specified :attr:`has_aux` equals ``True``,
            function can return a tuple of single-element Tensor and other auxiliary objects:
            ``(output, aux)``.
        argnums (int or Tuple[int]): Specifies arguments to compute gradients with respect to.
            :attr:`argnums` can be single integer or tuple of integers. Default: 0.
        has_aux (bool): Flag indicating that :attr:`func` returns a tensor and other
            auxiliary objects: ``(output, aux)``. Default: False.

    Returns:
        Function to compute gradients with respect to its inputs. By default, the output of
        the function is the gradient tensor(s) with respect to the first argument.
        If specified :attr:`has_aux` equals ``True``, tuple of gradients and output auxiliary objects
        is returned. If :attr:`argnums` is a tuple of integers, a tuple of output gradients with
        respect to each :attr:`argnums` value is returned.

    Example of using ``grad``:

        >>> from torch.func import grad
        >>> x = torch.randn([])
        >>> cos_x = grad(lambda x: torch.sin(x))(x)
        >>> assert torch.allclose(cos_x, x.cos())
        >>>
        >>> # Second-order gradients
        >>> neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
        >>> assert torch.allclose(neg_sin_x, -x.sin())

    When composed with ``vmap``, ``grad`` can be used to compute per-sample-gradients:

        >>> from torch.func import grad
        >>> from torch.func import vmap
        >>> batch_size, feature_size = 3, 5
        >>>
        >>> def model(weights, feature_vec):
        >>>     # Very simple linear model with activation
        >>>     assert feature_vec.dim() == 1
        >>>     return feature_vec.dot(weights).relu()
        >>>
        >>> def compute_loss(weights, example, target):
        >>>     y = model(weights, example)
        >>>     return ((y - target) ** 2).mean()  # MSELoss
        >>>
        >>> weights = torch.randn(feature_size, requires_grad=True)
        >>> examples = torch.randn(batch_size, feature_size)
        >>> targets = torch.randn(batch_size)
        >>> inputs = (weights, examples, targets)
        >>> grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)

    Example of using ``grad`` with :attr:`has_aux` and :attr:`argnums`:

        >>> from torch.func import grad
        >>> def my_loss_func(y, y_pred):
        >>>    loss_per_sample = (0.5 * y_pred - y) ** 2
        >>>    loss = loss_per_sample.mean()
        >>>    return loss, (y_pred, loss_per_sample)
        >>>
        >>> fn = grad(my_loss_func, argnums=(0, 1), has_aux=True)
        >>> y_true = torch.rand(4)
        >>> y_preds = torch.rand(4, requires_grad=True)
        >>> out = fn(y_true, y_preds)
        >>> > output is ((grads w.r.t y_true, grads w.r.t y_preds), (y_pred, loss_per_sample))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``grad``.

        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``grad(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``grad`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     grad(f)(x)

        In this case, ``grad`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``grad`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.

    """
    return _impl.grad(func, argnums, has_aux)


def functionalize(func: Callable, *, remove: str = 'mutations') -> Callable:
    """
    functionalize is a transform that can be used to remove (intermediate)
    mutations and aliasing from a function, while preserving the function's
    semantics.

    ``functionalize(func)`` returns a new function with the same semantics
    as ``func``, but with all intermediate mutations removed.
    Every inplace operation performed on an intermediate tensor:
    ``intermediate.foo_()``
    gets replaced by its out-of-place equivalent:
    ``intermediate_updated = intermediate.foo()``.

    functionalize is useful for shipping a pytorch program off to
    backends or compilers that aren't able to easily represent
    mutations or aliasing operators.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        remove (str): An optional string argument, that takes on either
            the value 'mutations' or 'mutations_and_views'.
            If 'mutations' is passed in then all mutating operators
            will be replaced with their non-mutating equivalents.
            If 'mutations_and_views' is passed in, then additionally, all aliasing
            operators will be replaced with their non-aliasing equivalents.
            Default: 'mutations'.

    Returns:
        Returns a new "functionalized" function. It takes the same inputs as
        :attr:`func`, and has the same behavior, but any mutations
        (and optionally aliasing) performed on intermeidate tensors
        in the function will be removed.

    functionalize will also remove mutations (and views) that were performed on function inputs.
    However to preserve semantics, functionalize will "fix up" the mutations after
    the transform has finished running, by detecting if any tensor inputs "should have"
    been mutated, and copying the new data back to the inputs if necessary.


    Example::

        >>> import torch
        >>> from torch.func import make_fx
        >>> from torch.func.experimental import functionalize
        >>>
        >>> A function that uses mutations and views, but only on intermediate tensors.
        >>> def f(a):
        ...     b = a + 1
        ...     c = b.view(-1)
        ...     c.add_(1)
        ...     return b
        ...
        >>> inpt = torch.randn(2)
        >>>
        >>> out1 = f(inpt)
        >>> out2 = functionalize(f)(inpt)
        >>>
        >>> # semantics are the same (outputs are equivalent)
        >>> print(torch.allclose(out1, out2))
        True
        >>>
        >>> f_traced = make_fx(f)(inpt)
        >>> f_no_mutations_traced = make_fx(functionalize(f))(inpt)
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>>
        >>> print(f_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1])
            add_ = torch.ops.aten.add_(view, 1);  view = None
            return add

        >>> print(f_no_mutations_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view, 1);  view = None
            view_1 = torch.ops.aten.view(add_1, [2]);  add_1 = None
            return view_1

        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view_copy = torch.ops.aten.view_copy(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add_1, [2]);  add_1 = None
            return view_copy_1


        >>> A function that mutates its input tensor
        >>> def f(a):
        ...     b = a.view(-1)
        ...     b.add_(1)
        ...     return a
        ...
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>>
        >>> All mutations and views have been removed,
        >>> but there is an extra copy_ in the graph to correctly apply the mutation to the input
        >>> after the function has completed.
        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            view_copy = torch.ops.aten.view_copy(a_1, [-1])
            add = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add, [2]);  add = None
            copy_ = torch.ops.aten.copy_(a_1, view_copy_1);  a_1 = None
            return view_copy_1


    There are a few "failure modes" for functionalize that are worth calling out:
      (1) Like other torch.func transforms, `functionalize()` doesn't work with functions
          that directly use `.backward()`. The same is true for torch.autograd.grad.
          If you want to use autograd, you can compute gradients directly
          with `functionalize(grad(f))`.
      (2) Like other torch.func transforms, `functionalize()` doesn't work with global state.
          If you call `functionalize(f)` on a function that takes views / mutations of
          non-local state, functionalization will simply no-op and pass the view/mutation
          calls directly to the backend.
          One way to work around this is is to ensure that any non-local state creation
          is wrapped into a larger function, which you then call functionalize on.
      (3) `resize_()` has some limitations: functionalize will only work on programs
          that use resize_()` as long as the tensor being resized is not a view.
      (4) `as_strided()` has some limitations: functionalize will not work on
          `as_strided()` calls that result in tensors with overlapping memory.


    Finally, a helpful mental model for understanding functionalization is that
    most user pytorch programs are writting with the public torch API.
    When executed, torch operators are generally decomposed into
    our internal C++ "ATen" API.
    The logic for functionalization happens entirely at the level of ATen.
    Functionalization knows how to take every aliasing operator in ATen,
    and map it to its non-aliasing equivalent
    (e.g. ``tensor.view({-1})`` -> ``at::view_copy(tensor, {-1})``),
    and how to take every mutating operator in ATen,
    and map it to its non-mutating equivalent
    (e.g. ``tensor.add_(1)`` -> ``at::add(tensor, -1)``),
    while tracking aliases and mutations out-of-line to know when to fix things up.
    Information about which ATen operators are aliasing or mutating all comes from
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml.
    """
    return _impl.functionalize(func, remove=remove)


def vmap(
        func: Callable,
        in_dims: in_dims_t = 0,
        out_dims: out_dims_t = 0,
        randomness: str = 'error') -> Callable:
    """
    vmap is the vectorizing map; ``vmap(func)`` returns a new function that
    maps :attr:`func` over some dimension of the inputs. Semantically, vmap
    pushes the map into PyTorch operations called by :attr:`func`, effectively
    vectorizing those operations.

    vmap is useful for handling batch dimensions: one can write a function
    :attr:`func` that runs on examples and then lift it to a function that can
    take batches of examples with ``vmap(func)``. vmap can also be used to
    compute batched gradients when composed with autograd.

    .. note::
        :func:`torch.vmap` and :func:`torch.func.vmap` are the same thing, use
        whichever one you'd like.

    Args:
        func (function): A Python function that takes one or more arguments.
            Must return one or more Tensors.
        in_dims (int or nested structure): Specifies which dimension of the
            inputs should be mapped over. :attr:`in_dims` should have a
            structure like the inputs. If the :attr:`in_dim` for a particular
            input is None, then that indicates there is no map dimension.
            Default: 0.
        out_dims (int or Tuple[int]): Specifies where the mapped dimension
            should appear in the outputs. If :attr:`out_dims` is a Tuple, then
            it should have one element per output. Default: 0.
        randomness (str): Specifies whether the randomness in this
            vmap should be the same or different across batches. If 'different',
            the randomness for each batch will be different. If 'same', the
            randomness will be the same across batches. If 'error', any calls to
            random functions will error. Default: 'error'. WARNING: this flag
            only applies to random PyTorch operations and does not apply to
            Python's random module or numpy randomness.

    Returns:
        Returns a new "batched" function. It takes the same inputs as
        :attr:`func`, except each input has an extra dimension at the index
        specified by :attr:`in_dims`. It takes returns the same outputs as
        :attr:`func`, except each output has an extra dimension at the index
        specified by :attr:`out_dims`.

    .. warning::
        :func:`vmap` works best with functional-style code. Please do not
        perform any side-effects in :attr:`func`, with the exception of
        in-place PyTorch operations. Examples of side-effects include mutating
        Python data structures and assigning values to variables not captured
        in :attr:`func`.

    One example of using :func:`vmap` is to compute batched dot products. PyTorch
    doesn't provide a batched ``torch.dot`` API; instead of unsuccessfully
    rummaging through docs, use :func:`vmap` to construct a new function.

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.vmap(torch.dot)  # [N, D], [N, D] -> [N]
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

    If the inputs are not batched along the first dimension, :attr:`in_dims` specifies
    the dimension that each inputs are batched along as

        >>> torch.dot                            # [N], [N] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=1)  # [N, D], [N, D] -> [D]
        >>> x, y = torch.randn(2, 5), torch.randn(2, 5)
        >>> batched_dot(x, y)   # output is [5] instead of [2] if batched along the 0th dimension

    If there are multiple inputs each of which is batched along different dimensions,
    :attr:`in_dims` must be a tuple with the batch dimension for each input as

        >>> torch.dot                            # [D], [D] -> []
        >>> batched_dot = torch.vmap(torch.dot, in_dims=(0, None))  # [N, D], [D] -> [N]
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> batched_dot(x, y) # second arg doesn't have a batch dim because in_dim[1] was None

    If the input is a Python struct, :attr:`in_dims` must be a tuple containing a struct
    matching the shape of the input:

        >>> f = lambda dict: torch.dot(dict['x'], dict['y'])
        >>> x, y = torch.randn(2, 5), torch.randn(5)
        >>> input = {'x': x, 'y': y}
        >>> batched_dot = torch.vmap(f, in_dims=({'x': 0, 'y': None},))
        >>> batched_dot(input)

    By default, the output is batched along the first dimension. However, it can be batched
    along any dimension by using :attr:`out_dims`

        >>> f = lambda x: x ** 2
        >>> x = torch.randn(2, 5)
        >>> batched_pow = torch.vmap(f, out_dims=1)
        >>> batched_pow(x) # [5, 2]

    For any function that uses kwargs, the returned function will not batch the kwargs but will
    accept kwargs

        >>> x = torch.randn([2, 5])
        >>> def f(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> batched_pow = torch.vmap(f)
        >>> assert torch.allclose(batched_pow(x), x * 4)
        >>> batched_pow(x, scale=x) # scale is not batched, output has shape [2, 2, 5]

    .. note::
        vmap does not provide general autobatching or handle variable-length
        sequences out of the box.
    """
    return _vmap_impl.vmap(func, in_dims, out_dims, randomness)
