import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._six import with_metaclass
import functools
import warnings
from collections import OrderedDict


class _ContextMethodMixin(object):

    def save_for_backward(self, *tensors):
        r"""Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors

    def mark_dirty(self, *args):
        r"""Marks given tensors as modified in an in-place operation.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be inputs.**

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.
        """
        self.dirty_tensors = args

    def mark_shared_storage(self, *pairs):
        warnings.warn(
            'mark_shared_storage is deprecated. '
            'Tensors with shared storages are automatically tracked. Note '
            'that calls to `set_()` are not tracked')

    def mark_non_differentiable(self, *args):
        r"""Marks outputs as non-differentiable.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be outputs.**

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.

        This is used e.g. for indices returned from a max :class:`Function`.
        """
        self.non_differentiable = args


class _HookMixin(object):

    @staticmethod
    def _register_hook(backward_hooks, hook):
        if backward_hooks is None:
            backward_hooks = OrderedDict()
        handle = hooks.RemovableHandle(backward_hooks)
        backward_hooks[handle.id] = hook
        return backward_hooks, handle


class BackwardCFunction(_C._FunctionBase, _ContextMethodMixin, _HookMixin):
    _is_legacy = False

    def apply(self, *args):
        return self._forward_cls.backward(self, *args)


class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _is_legacy: True if forward is not defined as a static method.
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """

    def __init__(cls, name, bases, attrs):
        for super_cls in cls.mro():
            forward = super_cls.__dict__.get('forward')
            if forward is not None:
                has_static_forward = isinstance(forward, staticmethod) or isinstance(forward, classmethod)
                break

        cls._is_legacy = not has_static_forward

        # old-style functions
        if not has_static_forward:
            return super(FunctionMeta, cls).__init__(name, bases, attrs)

        backward_fn = type(name + 'Backward', (BackwardCFunction,), {'_forward_cls': cls})
        cls._backward_cls = backward_fn

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class Function(with_metaclass(FunctionMeta, _C._FunctionBase, _ContextMethodMixin, _HookMixin)):
    r"""Records operation history and defines formulas for differentiating ops.

    Every operation performed on :class:`Tensor` s creates a new function
    object, that performs the computation, and records that it happened.
    The history is retained in the form of a DAG of functions, with edges
    denoting data dependencies (``input <- output``). Then, when backward is
    called, the graph is processed in the topological ordering, by calling
    :func:`backward` methods of each :class:`Function` object, and passing
    returned gradients on to next :class:`Function` s.

    Normally, the only way users interact with functions is by creating
    subclasses and defining new operations. This is a recommended way of
    extending torch.autograd.

    Each function object is meant to be used only once (in the forward pass).

    Examples::

        >>> class Exp(Function):
        >>>
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
    """

    # only for backward compatibility
    __call__ = _C._FunctionBase._do_forward

    # for the tracer
    is_traceable = False

    @staticmethod
    def forward(ctx, *args, **kwargs):
        r"""Performs the operation.

        This function is to be overridden by all subclasses.

        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).

        The context can be used to store tensors that can be then retrieved
        during the backward pass.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        r"""Defines a formula for differentiating the operation.

        This function is to be overridden by all subclasses.

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs did :func:`forward` return, and it should return as many
        tensors, as there were inputs to :func:`forward`. Each argument is the
        gradient w.r.t the given output, and each returned value should be the
        gradient w.r.t. the corresponding input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computated w.r.t. the
        output.
        """
        raise NotImplementedError


def once_differentiable(fn):

    @functools.wraps(fn)
    def wrapper(ctx, *args):
        with torch.no_grad():
            outputs = fn(ctx, *args)

        if not torch.is_grad_enabled():
            return outputs

        # If any of the inputs have requires_grad=True, we force the outputs
        # to have requires_grad=True but point to a grad_fn which throws an
        # error message during (double) back-propagation.
        # XXX: this is only an approximation of requires_grad - there's no way
        # to figure out if fn didn't use ctx.saved_tensors and as a result
        # some Tensors might require grad, even if no args do.
        # Unfortunately, this leads to unexpected error messages ("no nodes
        # require computing gradients"), but I don't have a better idea.
        # These functions would raise an error in backward anyway.
        requires_grad = any(isinstance(arg, torch.Tensor) and arg.requires_grad
                            for arg in args)
        if not requires_grad:
            return outputs

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        err_fn = torch._C._functions.DelayedError(
            b"trying to differentiate twice a function that was marked"
            b"with @once_differentiable", len(outputs))

        # Create aliases of each output that has requires_grad=True. We need
        # at least one of the inputs to err_fn to require grad so that the
        # output will have a grad_fn.
        def fake_requires_grad(var):
            if var is not None:
                var = var.detach()
                var.requires_grad = True
            return var

        return err_fn(*[fake_requires_grad(v) for v in outputs])
    return wrapper


def traceable(fn_cls):
    r"""Marks Function as traceable for the JIT.

    Traceable functions have additional restrictions - they can't pass any
    data-dependent values to backward (e.g. Prod passes the output, which makes
    it non-traceable), and their backward should be implemented entirely in terms
    of operations on autograd Tensors in all cases.

    DON'T USE THIS DECORATOR. IT IS FOR INTERNAL USE ONLY AND SHOULD BE HANDLED WITH
    CARE (or can give incorrect results otherwise).
    """
    fn_cls.is_traceable = True
    return fn_cls


class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace


def _nested_map(condition, fn, condition_msg=None):
    def _map(obj):
        if condition(obj):
            return fn(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            mapped = (_map(x) for x in obj)
            if hasattr(obj, '_fields'):
                # obj is namedtuple
                return type(obj)(*mapped)
            return type(obj)(mapped)
        elif isinstance(obj, dict):
            return {x : _map(obj[x]) for x in obj}
        else:
            raise ValueError("Auto nesting doesn't know how to process "
                             "an input object of type " + torch.typename(obj) +
                             (". Accepted types: " + condition_msg +
                              ", or lists/tuples of them"
                              if condition_msg else ""))

    return _map


def _jit_unwrap_structured(obj):
    if hasattr(obj, "_jit_unwrap"):
        return obj._jit_unwrap()
    return obj


def _iter_filter(condition, allow_unknown=False, condition_msg=None,
                 conversion=None):
    def _iter(obj):
        if conversion is not None:
            obj = conversion(obj)
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                for var in _iter(o):
                    yield var
        elif isinstance(obj, dict):
            # We only accept primitive key types, so we needn't inspect them
            for o in obj.values():
                for var in _iter(o):
                    yield var
        elif allow_unknown:
            yield obj
        else:
            raise ValueError("Auto nesting doesn't know how to process "
                             "an input object of type " + torch.typename(obj) +
                             (". Accepted types: " + condition_msg +
                              ", or lists/tuples of them"
                              if condition_msg else ""))

    return _iter


def _unflatten(input, proto):
    # unflatten a list or tuple input into a nested list/tuple structure
    # specified by proto
    def unflatten_helper(input, proto):
        res = []
        if hasattr(proto, "_jit_wrap"):
            return proto._jit_wrap(input)
        if not isinstance(proto, (list, tuple)):
            return input[0], input[1:]
        for e in proto:
            if e is None:
                res.append(e)
            else:
                res_e, input = unflatten_helper(input, e)
                res.append(res_e)
        return type(proto)(res), input

    return unflatten_helper(input, proto)[0]


_iter_jit_values = _iter_filter(lambda o: o is None or isinstance(o, torch._C.Value),
                                condition_msg="jit's Values or None")
_iter_tensors = _iter_filter(lambda x: isinstance(x, torch.Tensor), condition_msg="Tensors",
                             conversion=_jit_unwrap_structured)
_iter_tensors_permissive = _iter_filter(lambda x: isinstance(x, torch.Tensor),
                                        allow_unknown=True,
                                        condition_msg="Tensors (permissive)")
_iter_None_tensors = _iter_filter(lambda o: o is None or isinstance(o, torch.Tensor),
                                  condition_msg="Tensors or None")
_map_tensor_data = _nested_map(lambda x: isinstance(x, torch.Tensor), lambda o: o.data,
                               condition_msg="Tensors")


class NestedIOFunction(Function):

    def _do_forward(self, *input):
        self._nested_input = input
        flat_input = tuple(_iter_tensors(input))
        flat_output = super(NestedIOFunction, self)._do_forward(*flat_input)
        nested_output = self._nested_output
        nested_tensors = _unflatten(flat_output, self._nested_output)
        return nested_tensors

    def _do_backward(self, gradients, retain_variables):
        self.retain_variables = retain_variables
        result = super(NestedIOFunction, self)._do_backward(gradients, retain_variables)
        if not retain_variables:
            del self._nested_output
            del self._to_save_nested
        return result

    def backward(self, *gradients):
        nested_gradients = _unflatten(gradients, self._nested_output)
        result = self.backward_extended(*nested_gradients)
        return tuple(_iter_None_tensors(result))

    __call__ = _do_forward

    def forward(self, *args):
        nested_tensors = _map_tensor_data(self._nested_input)
        result = self.forward_extended(*nested_tensors)
        del self._nested_input
        self._nested_output = result
        return tuple(_iter_tensors(result))

    def save_for_backward(self, *args):
        self.to_save = tuple(_iter_tensors(args))
        self._to_save_nested = args

    @property
    def saved_tensors(self):
        flat_tensors = super(NestedIOFunction, self).saved_tensors
        return _unflatten(flat_tensors, self._to_save_nested)

    def mark_dirty(self, *args, **kwargs):
        self.dirty_tensors = tuple(_iter_tensors((args, kwargs)))

    def mark_non_differentiable(self, *args, **kwargs):
        self.non_differentiable = tuple(_iter_tensors((args, kwargs)))

    def forward_extended(self, *input):
        raise NotImplementedError

    def backward_extended(self, *grad_output):
        raise NotImplementedError
