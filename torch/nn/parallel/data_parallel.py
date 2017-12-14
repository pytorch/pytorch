import weakref
import torch
import torch.cuda.comm as comm
from collections import OrderedDict
from ..modules import Module
from .scatter_gather import scatter_kwargs, gather
from .replicate import replicate
from .parallel_apply import parallel_apply


# This class behaves exacly like OrderedDict, but lets you register callbacks
# that will be triggered at every modification. This is used to invalidate
# cached replicas in DataParallel in case the original module is modified.
# NOTE: these callbacks are one-off
# NOTE: they are deduplicated based on identity
class _CallbackOrderedDict(OrderedDict):
    @classmethod
    def _init_class(cls):
        mutating_methods = [
            '__setitem__',
            '__delitem__',
            'pop',
            'popitem',
            'update',
            'setdefault'
        ]

        def method_wrapper(name):
            def callback_method(self, *args, **kwargs):
                for callback in self.callbacks:
                    callback()
                del self.callbacks[:]
                return getattr(super(_CallbackOrderedDict, self), name)(*args, **kwargs)
            return callback_method
        for name in mutating_methods:
            setattr(cls, name, method_wrapper(name))

    @classmethod
    def from_ordered(cls, od):
        return cls(od.items())

    def __init__(self, *args, **kwargs):
        self.callbacks = []
        super(_CallbackOrderedDict, self).__init__(*args, **kwargs)

    def register_modification_callback(self, cb):
        if all(c is not cb for c in self.callbacks):
            self.callbacks.append(cb)

    # Don't serialize callbacks. If a module registers them on its submodules,
    # it will have to redo it after reloading.
    def __reduce__(self):
        state = super(_CallbackOrderedDict, self).__reduce__()
        # NB: 3nd element of reduce tuple is the __dict__.
        if state[2]:
            lstate = list(state)
            lstate[2] = state[2].copy().pop('callbacks')
            return tuple(lstate)
        else:
            return state


_CallbackOrderedDict._init_class()
torch._C._inherit_odict_getitem(_CallbackOrderedDict)


class DataParallel(Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self._init_cache()
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def _init_cache(self):
        self._replicas = []

        # Use a weakref to avoid reference cycles
        dp_weakref = weakref.ref(self)

        def invalidate_cache():
            dp = dp_weakref()
            if dp is None:
                return

            # Flush the cache
            dp._replicas = []
            del dp._orig_params, dp._orig_buffers
            del dp._repl_params, dp._repl_buffers

        self._invalidate_cache = invalidate_cache

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def __getstate__(self):
        state = super(DataParallel, self).__getstate__().copy()
        del state['_invalidate_cache'], state['_replicas'], state['_orig_params'], \
            state['_orig_buffers'], state['_repl_params'], state['_repl_buffers']
        return state

    def __setstate__(self, state):
        super(DataParallel, self).__setstate__(state)
        self._init_cache()

    def replicate(self, module, device_ids):
        if not self._replicas:
            self._replicas = replicate(module, device_ids)

            def register_callbacks(m, module_cb, param_cb, buffer_cb):
                for sm in m.modules():
                    sm._modules = _CallbackOrderedDict.from_ordered(sm._modules)
                    sm._modules.register_modification_callback(module_cb)
                    sm._parameters = _CallbackOrderedDict.from_ordered(sm._parameters)
                    sm._parameters.register_modification_callback(param_cb)
                    sm._buffers = _CallbackOrderedDict.from_ordered(sm._buffers)
                    sm._buffers.register_modification_callback(buffer_cb)

            # Cache has to be invalidated if any of modules, parameters or dicts
            # are modified in **any** of the submodules. Note, that we don't need to
            # ensure the same for __dict__ because it is shared with the original module.
            register_callbacks(module, self._invalidate_cache,
                               self._invalidate_cache, self._invalidate_cache)

            # Modification of these dicts in replicas is forbidden. They should always
            # mirror the original module exactly.
            def make_error(name):
                def callback():
                    raise RuntimeError("DataParallel replicas can't have their " + name + " modified")
                return callback
            for replica in self._replicas:
                register_callbacks(replica, make_error('modules'),
                                   make_error('parameters'), make_error('buffers'))

            # Cache information to speed up processing in the fast path
            self._orig_params = list(module.parameters())
            self._orig_buffers = list(module._all_buffers())
            self._repl_params = [list(r.parameters()) for r in self._replicas]
            self._repl_buffers = [list(r._all_buffers()) for r in self._replicas]
        else:
            # Param flags are the only things that we can't get callbacks for.
            # Everything is fine as long as it's never the case that a replica doesn't
            # require grad, while the original parameter does.
            requires_grad_valid = all(rp.requires_grad >= p.requires_grad
                                      for rp, p in zip(self._repl_params[0], self._orig_params))
            if not requires_grad_valid:
                self._replicas = []  # Flush the cache
                return self.replicate(module, device_ids)

            # NB: we "misuse" the fact that you can safely backprop through broadcasts
            # multiple times, because they hold no buffers
            param_copies = comm.broadcast_coalesced([p.data for p in self._orig_params], device_ids)
            for replica_params, copies in zip(self._repl_params, param_copies):
                for p, cp in zip(replica_params, copies):
                    p.data.set_(cp)

            buffer_copies = comm.broadcast_coalesced(self._orig_buffers, device_ids)
            for replica_buffers, copies in zip(self._repl_buffers, buffer_copies):
                for b, cb in zip(replica_buffers, copies):
                    b.set_(cb)

        return self._replicas

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
