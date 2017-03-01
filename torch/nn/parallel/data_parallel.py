import torch
from torch.autograd import Variable
from ..modules import Module
from .scatter_gather import scatter, gather
from .replicate import replicate
from .parallel_apply import parallel_apply


class DataParallel(Module):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> device_ids=[0, 1, 2]
        >>> net = torch.nn.DataParallel(model, device_ids=device_ids)
        >>> input_var.size(0) % len(device_ids)
        0
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well
    def __init__(self, module, device_ids=None, output_device=None):
        super(DataParallel, self).__init__()
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs):
        def _to_cuda(obj):
            if isinstance(obj, Variable):
                return obj.cuda()
            return tuple((map(_to_cuda, obj)))

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
        return scatter(input, device_ids)

    def parallel_apply(self, replicas, inputs):
        return parallel_apply(replicas, inputs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device)


def data_parallel(module, inputs, device_ids, output_device=None):
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
    scattered = scatter(inputs, device_ids)
    replicas = replicas[:len(scattered)]
    outputs = parallel_apply(replicas, scattered)
    return gather(outputs, output_device)
