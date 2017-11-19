import sys
import torch
import torch._C as _C
from collections import OrderedDict
import torch.sparse as sparse
import torch.utils.hooks as hooks
import warnings
import weakref
from torch._six import imap
from torch._C import _add_docstr


class Variable(_C._VariableBase):
    """Wraps a tensor and records the operations applied to it.

    Variable is a thin wrapper around a Tensor object, that also holds
    the gradient w.r.t. to it, and a reference to a function that created it.
    This reference allows retracing the whole chain of operations that
    created the data. If the Variable has been created by the user, its grad_fn
    will be ``None`` and we call such objects *leaf* Variables.

    Since autograd only supports scalar valued function differentiation, grad
    size always matches the data size. Also, grad is normally only allocated
    for leaf variables, and will be always zero otherwise.

    Attributes:
        data: Wrapped tensor of any type.
        grad: Variable holding the gradient of type and location matching
            the ``.data``.  This attribute is lazily allocated and can't
            be reassigned.
        requires_grad: Boolean indicating whether the Variable has been
            created by a subgraph containing any Variable, that requires it.
            See :ref:`excluding-subgraphs` for more details.
            Can be changed only on leaf Variables.
        volatile: Boolean indicating that the Variable should be used in
            inference mode, i.e. don't save the history. See
            :ref:`excluding-subgraphs` for more details.
            Can be changed only on leaf Variables.
        is_leaf: Boolean indicating if the Variable is a graph leaf (i.e
            if it was created by the user).
        grad_fn: Gradient function graph trace.

    Parameters:
        data (any tensor class): Tensor to wrap.
        requires_grad (bool): Value of the requires_grad flag. **Keyword only.**
        volatile (bool): Value of the volatile flag. **Keyword only.**
    """

    def __getitem__(self, key):
        if torch.is_tensor(key):
            key = Variable(key)  # auto-wrap tensors
        if isinstance(key, Variable):
            if type(key.data).__name__ == 'ByteTensor':
                return MaskedSelect.apply(self, key)
            elif type(key.data).__name__ == 'LongTensor':
                return IndexSelect.apply(self, 0, key)
            # else fall through and raise an error in Index
        return Index.apply(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, Variable) and type(key.data).__name__ == 'ByteTensor':
            if isinstance(value, Variable):
                return MaskedScatter.apply(self, key, value, True)
            else:
                return MaskedFill.apply(self, key, value, True)
        else:
            return SetItem.apply(self, key, value)

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError("Only Variables created explicitly by the user "
                               "(graph leaves) support the deepcopy protocol at the moment")
        result = type(self)(self.data.clone())
        result.requires_grad = self.requires_grad
        result.volatile = self.volatile
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        state = (self.requires_grad, self.volatile, self._backward_hooks)
        if proto > 1:
            return type(self), (self.data,), state
        if sys.version_info[0] == 2:
            from copy_reg import __newobj__
        else:
            from copyreg import __newobj__
        return __newobj__, (type(self), self.data), state

    def __setstate__(self, state):
        if len(state) == 5:
            # legacy serialization of Variable
            self.data = state[0]
            state = (state[3], state[4], state[2])
        if not self.is_leaf:
            raise RuntimeError('__setstate__ can be only called on leaf variables')
        self.requires_grad, self.volatile, self._backward_hooks = state

    def __repr__(self):
        return 'Variable containing:' + self.data.__repr__()

    def __bool__(self):
        if self.data.numel() <= 1:
            return self.data.__bool__()
        raise RuntimeError("bool value of Variable containing " +
                           torch.typename(self.data) +
                           " with more than one value is ambiguous")

    __nonzero__ = __bool__

    def __int__(self):
        return int(self.data)

    def __long__(self):
        return long(self.data)

    def __float__(self):
        return float(self.data)

    def backward(self, gradient=None, retain_graph=None, create_graph=None, retain_variables=None):
        """Computes the gradient of current variable w.r.t. graph leaves.

        The graph is differentiated using the chain rule. If the variable is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying ``gradient``.
        It should be a tensor of matching type and location, that contains
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to
        zero them before calling it.

        Arguments:
            gradient (Tensor, Variable or None): Gradient w.r.t. the
                variable. If it is a tensor, it will be automatically converted
                to a Variable that is volatile unless ``create_graph`` is True.
                None values can be specified for scalar Variables or ones that
                don't require grad. If a None value would be acceptable then
                this argument is optional.
            retain_graph (bool, optional): If ``False``, the graph used to compute
                the grads will be freed. Note that in nearly all cases setting
                this option to True is not needed and often can be worked around
                in a much more efficient way. Defaults to the value of
                ``create_graph``.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``, unless ``gradient`` is a volatile
                Variable.
        """
        torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)

    def register_hook(self, hook):
        """Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        variable is computed. The hook should have the following signature::

            hook(grad) -> Variable or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        Example:
            >>> v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.Tensor([1, 1, 1]))
            >>> v.grad.data
             2
             2
             2
            [torch.FloatTensor of size 3]
            >>> h.remove()  # removes the hook
        """
        if self.volatile:
            raise RuntimeError("cannot register a hook on a volatile variable")
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a variable that "
                               "doesn't require gradient")
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        def trim(str):
            return '\n'.join([line.strip() for line in str.split('\n')])

        raise RuntimeError(trim(r"""reinforce() was removed.
            Use torch.distributions instead.
            See http://pytorch.org/docs/master/distributions.html

            Instead of:

            probs = policy_network(state)
            action = probs.multinomial()
            next_state, reward = env.step(action)
            action.reinforce(reward)
            action.backward()

            Use:

            probs = policy_network(state)
            # NOTE: categorical is equivalent to what used to be called multinomial
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            next_state, reward = env.step(action)
            loss = -m.log_prob(action) * reward
            loss.backward()
        """))

    detach = _add_docstr(_C._VariableBase.detach, r"""
    Returns a new Variable, detached from the current graph.

    Result will never require gradient. If the input is volatile, the output
    will be volatile too.

    .. note::

      Returned Variable uses the same data tensor, as the original one, and
      in-place modifications on either of them will be seen, and may trigger
      errors in correctness checks.
    """)

    detach_ = _add_docstr(_C._VariableBase.detach_, r"""
    Detaches the Variable from the graph that created it, making it a leaf.
    Views cannot be detached in-place.
    """)

    def retain_grad(self):
        """Enables .grad attribute for non-leaf Variables."""
        if self.grad_fn is None:  # no-op for leaves
            return
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Variable that has requires_grad=False")
        if hasattr(self, 'retains_grad'):
            return
        weak_self = weakref.ref(self)

        def retain_grad_hook(grad):
            var = weak_self()
            if var is None:
                return
            if var._grad is None:
                var._grad = grad.clone()
            else:
                var._grad = var._grad + grad

        self.register_hook(retain_grad_hook)
        self.retains_grad = True

    def type(self, t):
        if t != type(self.data):
            return Type.apply(self, t)
        return self

    def type_as(self, t):
        if isinstance(t, Variable):
            t = t.data
        return self.type(type(t))

    def _get_type(self, name):
        module = torch._import_dotted_name(self.data.__module__)
        return getattr(module, name)

    def cuda(self, device=None, async=False):
        return CudaTransfer.apply(self, device, async)

    def cpu(self):
        return self.type(getattr(torch, type(self.data).__name__))

    def double(self):
        return self.type(self._get_type('DoubleTensor'))

    def float(self):
        return self.type(self._get_type('FloatTensor'))

    def half(self):
        return self.type(self._get_type('HalfTensor'))

    def long(self):
        return self.type(self._get_type('LongTensor'))

    def int(self):
        return self.type(self._get_type('IntTensor'))

    def short(self):
        return self.type(self._get_type('ShortTensor'))

    def char(self):
        return self.type(self._get_type('CharTensor'))

    def byte(self):
        return self.type(self._get_type('ByteTensor'))

    def prod(self, dim=None, keepdim=None):
        return Prod.apply(self, dim, keepdim)

    def view_as(self, tensor):
        return self.view(tensor.size())

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], torch.Size):
            repeats = repeats[0]
        else:
            repeats = torch.Size(repeats)
        return Repeat.apply(self, repeats)

    def cumsum(self, dim):
        return Cumsum.apply(self, dim)

    def cumprod(self, dim):
        return Cumprod.apply(self, dim)

    def renorm(self, p, dim, maxnorm):
        t = self.transpose(dim, 0)
        flat = t.contiguous().view(self.size(0), -1)
        norms = flat.norm(p, 1, True)
        norms = norms.clamp(max=maxnorm).div(norms.add(1e-7))
        flat_out = flat.mul(norms.expand_as(flat))
        return flat_out.view(t.size()).transpose(dim, 0)

    def matmul(self, other):
        return torch.matmul(self, other)

    def resize(self, *sizes):
        return Resize.apply(self, sizes)

    def resize_as(self, variable):
        return Resize.apply(self, variable.size())

    def norm(self, p=2, dim=None, keepdim=False):
        if dim is None:
            return super(Variable, self).norm(p)
        else:
            return super(Variable, self).norm(p, dim, keepdim)

    def index_add(self, dim, index, tensor):
        return self.clone().index_add_(dim, index, tensor)

    def _advanced_index_add(self, index, tensor):
        return AdvancedIndexAdd.apply(self, index, tensor)

    def index_copy(self, dim, index, tensor):
        return self.clone().index_copy_(dim, index, tensor)

    def index_fill(self, dim, index, value):
        return self.clone().index_fill_(dim, index, value)

    def scatter(self, dim, index, source):
        return self.clone().scatter_(dim, index, source)

    def scatter_add(self, dim, index, source):
        return self.clone().scatter_add_(dim, index, source)

    def masked_copy(self, mask, variable):
        warnings.warn("masked_copy is deprecated and renamed to masked_scatter, and will be removed in v0.3")
        return self.masked_scatter(mask, variable)

    def masked_copy_(self, mask, variable):
        warnings.warn("masked_copy_ is deprecated and renamed to masked_scatter_, and will be removed in v0.3")
        return self.masked_scatter_(mask, variable)

    def masked_scatter(self, mask, variable):
        return self.clone().masked_scatter_(mask, variable)

    def masked_fill(self, mask, value):
        return self.clone().masked_fill_(mask, value)

    def expand_as(self, tensor):
        return self.expand(tensor.size())

    def multinomial(self, num_samples=1, replacement=False):
        return Variable(torch.multinomial(self.data, num_samples, replacement))

    def bernoulli(self):
        return Variable(torch.bernoulli(self.data))

    def __rsub__(self, other):
        return -self + other

    def __matmul__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return self.matmul(other)

    def __rdiv__(self, other):
        return self.reciprocal() * other
    __rtruediv__ = __rdiv__

    __pow__ = _C._VariableBase.pow

    def __ipow__(self, other):
        raise NotImplementedError("in-place pow not implemented")

    def __rpow__(self, other):
        return PowConstant.apply(other, self)

    __neg__ = _C._VariableBase.neg

    __eq__ = _C._VariableBase.eq
    __ne__ = _C._VariableBase.ne
    __lt__ = _C._VariableBase.lt
    __le__ = _C._VariableBase.le
    __gt__ = _C._VariableBase.gt
    __ge__ = _C._VariableBase.ge

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # NB: we use 'imap' and not 'map' here, so that in Python 2 we get a
        # generator and don't eagerly perform all the indexes.  This could
        # save us work, and also helps keep trace ordering deterministic
        # (e.g., if you zip(*hiddens), the eager map will force all the
        # indexes of hiddens[0] before hiddens[1], while the generator
        # map will interleave them.)
        return iter(imap(lambda i: self[i], range(self.size(0))))

    def __hash__(self):
        return id(self)

    def __dir__(self):
        variable_methods = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        keys = variable_methods + attrs
        return sorted(keys)

    class _torch(object):
        @staticmethod
        def normal(means, std=1):
            if isinstance(means, Variable):
                means = means.data
            if isinstance(std, Variable):
                std = std.data
            return Variable(torch.normal(means, std))


for method in dir(Variable):
    # This will also wrap some methods that normally aren't part of the
    # functional interface, but we don't care, as they won't ever be used
    if method.startswith('_') or method.endswith('_'):
        continue
    if hasattr(Variable._torch, method):
        continue
    as_static = staticmethod(getattr(Variable, method))
    setattr(Variable._torch, method, as_static)


from ._functions import *
from torch._C import _ImperativeEngine as ImperativeEngine
Variable._execution_engine = ImperativeEngine()
