import torch
from torch.autograd import Variable
from ..modules import Module
from .scatter_gather import scatter, gather
from .replicate import replicate
from .parallel_apply import parallel_apply


class DataParallel(Module):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices. In the forward pass, the
    module is replicated on each device, and each replica handles a portion of
    the input. During the backwards pass, gradients from each replica are
    summed into the original module.

    Arbitrary positional inputs are allowed to be passed into DataParallel.
    All variables will be scattered on dim specified (default 0). Tensors will
    be broadcasted across devices, however any modifications in model's forward
    will not be saved. Primitive types will be similarly broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass. Keyword arguments are not allowed as input.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])
    Example:
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input)
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs):
        #leave non-Variable/non-tuples in place
        def _to_cuda(obj):
            if isinstance(obj, Variable):
                return obj.cuda()
            if isinstance(obj, tuple):
                return tuple((map(_to_cuda, obj)))
            return obj

        if len(self.device_ids) == 1:
            with torch.cuda.device(self.device_ids[0]):
                inputs_cuda = _to_cuda(inputs)
            return self.module(*inputs_cuda)
        replicas = self.replicate(self.module, self.device_ids)
        scattered = self.scatter(inputs, self.device_ids)
        replicas = replicas[:len(scattered)]
        outputs = self.parallel_apply(replicas, scattered)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, input, device_ids):
        return scatter(input, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs):
        return parallel_apply(replicas, inputs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids, output_device=None, dim=0):
    """Evaluates module(input) in parallel across the GPUs given in device_ids.

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

    if not device_ids:
        return module(*inputs)

    if output_device is None:
        output_device = device_ids[0]

    replicas = replicate(module, device_ids)
    scattered = scatter(inputs, device_ids, dim)
    replicas = replicas[:len(scattered)]
    outputs = parallel_apply(replicas, scattered)
    return gather(outputs, output_device, dim)
