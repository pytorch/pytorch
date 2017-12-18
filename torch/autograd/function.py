import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._six import with_metaclass
import functools
from collections import OrderedDict


class _ContextMethodMixin(object):

    def save_for_backward(self, *tensors):
        """Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute; or, if the corresponding Variable is needed (e.g. for double
        backwards), those can be accessed through the :attr:`saved_variables`
        attribute.  Before returning them to the user, a check is made, to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors

    def mark_dirty(self, *args):
        """Marks given tensors as modified in an in-place operation.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be inputs.**

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.
        """
        self.dirty_tensors = args

    def mark_shared_storage(self, *pairs):
        """Marks that given pairs of distinct tensors are sharing storage.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be pairs of
        (input, output).**

        If some of the outputs are going to be tensors sharing storage with
        some of the inputs, all pairs of (input_arg, output_arg) should be
        given to this function, to ensure correctness checking of in-place
        modification. The only exception is when an output is exactly the same
        tensor as input (e.g. in-place ops). In such case it's easy to conclude
        that they're sharing data, so we don't require specifying such
        dependencies.

        This function is not needed in most functions. It's primarily used in
        indexing and transpose ops.
        """
        self.shared_pairs = pairs

    def mark_non_differentiable(self, *args):
        """Marks outputs as non-differentiable.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be outputs.**

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be ``None``.

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

        setattr(cls, '_is_legacy', not has_static_forward)

        # old-style functions
        if not has_static_forward:
            return super(FunctionMeta, cls).__init__(name, bases, attrs)

        backward_fn = type(name + 'Backward', (BackwardCFunction,), {'_forward_cls': cls})
        setattr(cls, '_backward_cls', backward_fn)

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class Function(with_metaclass(FunctionMeta, _C._FunctionBase, _ContextMethodMixin, _HookMixin)):
    """Records operation history and defines formulas for differentiating ops.

    Every operation performed on :class:`Variable` s creates a new function
    object, that performs the computation, and records that it happened.
    The history is retained in the form of a DAG of functions, with edges
    denoting data dependencies (``input <- output``). Then, when backward is
    called, the graph is processed in the topological ordering, by calling
    :func:`backward` methods of each :class:`Function` object, and passing
    returned gradients on to next :class:`Function` s.

    Normally, the only way users interact with functions is by creating
    subclasses and defining new operations. This is a recommended way of
    extending torch.autograd.

    Each function is meant to be used only once (in the forward pass).

    Attributes:
        requires_grad: Boolean indicating whether the :func:`backward` will
            ever need to be called.

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
        >>>         result, = ctx.saved_variables
        >>>         return grad_output * result
    """

    # only for backward compatibility
    __call__ = _C._FunctionBase._do_forward

    # for the tracer
    is_traceable = False

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Performs the operation.

        This function is to be overriden by all subclasses.

        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).

        The context can be used to store variables that can be then retrieved
        during the backward pass.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Defines a formula for differentiating the operation.

        This function is to be overriden by all subclasses.

        It must accept a context ctx as the first argument, followed by as many
        outputs did :func:`forward` return, and it should return as many
        tensors, as there were inputs to :func:`forward`. Each argument is the
        gradient w.r.t the given output, and each returned value should be the
        gradient w.r.t. the corresponding input.

        The context can be used to retrieve variables saved during the forward
        pass.
        """
        raise NotImplementedError


def once_differentiable(fn):
    from .variable import Variable

    @functools.wraps(fn)
    def wrapper(ctx, *args):
        tensor_args = [arg.data if isinstance(arg, Variable) else arg
                       for arg in args]
        outputs = fn(ctx, *tensor_args)
        # XXX: this is only an approximation of these flags - there's no way
        # to figure out if fn didn't use ctx.saved_variables and as a result
        # some Variables might require grad, even if no args do.
        # Unfortunately, this leads to unexpected error messages ("no nodes
        # require computing gradients"), but I don't have a better idea.
        # These functions would raise an error in backward anyway.
        volatile = any(arg.volatile if isinstance(arg, Variable) else False
                       for arg in args)
        requires_grad = any(arg.requires_grad if isinstance(arg, Variable) else False
                            for arg in args)
        if volatile:
            def err_fn(*args):
                return args
            kwargs = {'volatile': True}
        else:
            err_fn = torch._C._functions.DelayedError(
                b"trying to differentiate twice a function that was marked"
                b"with @once_differentiable")
            kwargs = {'requires_grad': requires_grad}
        if not isinstance(outputs, tuple):
            var = Variable(outputs, **kwargs) if outputs is not None else None
            return err_fn(var)
        return err_fn(*[Variable(o, **kwargs) if o is not None else None
                      for o in outputs])
    return wrapper


def traceable(fn_cls):
    """Marks Function as traceable for the JIT.

    Traceable functions have additional restrictions - they can't pass any
    data-dependent values to backward (e.g. Prod passes the output, which makes
    it non-traceable), and their backward should be implemented entirely in terms
    of operations on autograd Variables in all cases (even when grads are volatile).

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
            return type(obj)(_map(x) for x in obj)
        else:
            raise ValueError("Auto nesting doesn't know how to process "
                             "an input object of type " + torch.typename(obj) +
                             (". Accepted types: " + condition_msg +
                              ", or lists/tuples of them"
                              if condition_msg else ""))

    return _map


def _iter_filter(condition, skip_unknown=False, condition_msg=None):
    def _iter(obj):
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                for var in _iter(o):
                    yield var
        elif not skip_unknown:
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
        if not isinstance(proto, (list, tuple)):
            return input[0], input[1:]
        for e in proto:
            res_e, input = unflatten_helper(input, e)
            res.append(res_e)
        return type(proto)(res), input

    return unflatten_helper(input, proto)[0]


_iter_variables = _iter_filter(lambda o: isinstance(o, torch.autograd.Variable), condition_msg="Variables")
_iter_variables_permissive = _iter_filter(lambda o: isinstance(o, torch.autograd.Variable), skip_unknown=True)
_iter_jit_values = _iter_filter(lambda o: isinstance(o, torch._C.Value), condition_msg="jit's Values")
_iter_tensors = _iter_filter(torch.is_tensor, condition_msg="Tensors")
_iter_None_tensors = _iter_filter(lambda o: o is None or torch.is_tensor(o), condition_msg="Tensors or None")
_map_variable_tensor = _nested_map(lambda o: isinstance(o, torch.autograd.Variable),
                                   lambda o: o.data, condition_msg="Variables")


class NestedIOFunction(Function):

    def _do_forward(self, *input):
        self._nested_input = input
        flat_input = tuple(_iter_variables(input))
        flat_output = super(NestedIOFunction, self)._do_forward(*flat_input)
        nested_output = self._nested_output
        nested_variables = _unflatten(flat_output, self._nested_output)
        return nested_variables

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
        nested_tensors = _map_variable_tensor(self._nested_input)
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
