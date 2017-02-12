import torch
import torch._C as _C
import torch.utils.hooks as hooks
from collections import OrderedDict
from itertools import chain


class Function(_C._FunctionBase):
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

    Since Function logic is a hotspot in most scripts, almost all of it
    was moved to our C backend, to ensure that the framework overhead is
    minimal.

    Each function is meant to be used only once (in the forward pass).

    Attributes:
        saved_tensors: Tuple of Tensors that were saved in the call to
            :func:`forward`.
        needs_input_grad: Tuple of booleans of length :attr:`num_inputs`,
            indicating whether a given input requires gradient. This can be
            used to optimize buffers saved for backward, and ignoring gradient
            computation in :func:`~Function.backward`.
        num_inputs: Number of inputs given to :func:`forward`.
        num_outputs: Number of tensors returned by :func:`forward`.
        requires_grad: Boolean indicating whether the :func:`backward` will
            ever need to be called.
        previous_functions: Tuple of (int, Function) pairs of length
            :attr:`num_inputs`. Each entry contains a reference to a
            :class:`Function` that created corresponding input, and an index
            of the previous function output that's been used.
    """
    __call__ = _C._FunctionBase._do_forward

    def save_for_backward(self, *tensors):
        """Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made, to
        ensure they weren't used in any in-place operation that modified
        their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors

    def mark_dirty(self, *args):
        """Marks given tensors as modified in an in-place operation.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be inputs.**

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correcness of our checks.
        It doesn't matter wheter the function is called before or after
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

        This will mark outputs as non requiring gradient, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for this output in :meth:`~Function.backward`, but it's always going to
        be ``None``.

        This is used e.g. for indices returned from a max :class:`Function`.
        """
        self.non_differentiable = args

    def register_hook(self, hook):
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[id(handle)] = hook
        return handle

    def forward(self, *input):
        """Performs the operation.

        This function is to be overriden by all subclasses.

        It can take and return an arbitrary number of tensors.
        """
        raise NotImplementedError

    def backward(self, *grad_output):
        """Defines a formula for differentiating the operation.

        This function is to be overriden by all subclasses.

        All arguments are tensors. It has to accept exactly as many arguments,
        as many outputs did :func:`forward` return, and it should return as
        many tensors, as there were inputs to :func:`forward`. Each argument
        is the gradient w.r.t the given output, and each returned value should
        be the gradient w.r.t. the corresponding input.
        """
        raise NotImplementedError


class InplaceFunction(Function):

    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace


def _nested_map(condition, fn):
    def _map(obj):
        if condition(obj):
            return fn(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_map(x) for x in obj)
        else:
            raise ValueError("NestedIOFunction doesn't know how to process "
                             "an input object of type " + torch.typename(obj))
    return _map


def _iter_filter(condition):
    def _iter(obj):
        if condition(obj):
            yield obj
        elif obj is None:
            return
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                for var in _iter(o):
                    yield var
        else:
            raise ValueError("NestedIOFunction doesn't know how to process "
                             "an input object of type " + torch.typename(obj))
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

_iter_variables = _iter_filter(lambda o: isinstance(o, torch.autograd.Variable))
_iter_tensors = _iter_filter(torch.is_tensor)
_iter_None_tensors = _iter_filter(lambda o: o is None or torch.is_tensor(o))
_map_variable_tensor = _nested_map(lambda o: isinstance(o, torch.autograd.Variable), lambda o: o.data)


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
