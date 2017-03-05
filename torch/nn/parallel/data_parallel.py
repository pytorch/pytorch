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

        >>> device_ids=[0, 1, 2]
        >>> net = torch.nn.DataParallel(model, device_ids=device_ids)
        >>> input_var.size(0) % len(device_ids)
        0
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

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

    def forward(self, *inputs, **kwargs):
        def _to_cuda(obj):
            if isinstance(obj, Variable):
                return obj.cuda()
            if isinstance(obj, tuple) or isinstance(obj, list):
                return type(obj)((map(_to_cuda, obj)))
            return obj

        if len(self.device_ids) == 1:
            with torch.cuda.device(self.device_ids[0]):
                inputs_cuda = _to_cuda(inputs)
                if kwargs:
                    gpu_dict = {}
                    for key in kwargs.keys():
                        gpu_dict[key] = _to_cuda(kwargs[key])
                    return self.module(*inputs_cuda, **gpu_dict)
                else:
                    return self.module(*inputs_cuda)

        replicas = self.replicate(self.module, self.device_ids)
        scattered = self.scatter(inputs, self.device_ids)
        used_gpus = len(scattered)  # The last GPU might not be used. For example, input of size 5, on 4 GPUs
        gpu_dicts = None
        if kwargs:
            scatter_kwargs = {}
            for key in kwargs.keys():
                scatter_kwargs[key] = self.scatter(
                    _to_cuda(kwargs[key]), self.device_ids)

            gpu_dicts = tuple()
            for i in range(used_gpus):
                gpu_dict = {}
                for key, values in scatter_kwargs.items():
                    assert len(values) == used_gpus
                    gpu_dict[key] = values[i]
                gpu_dicts += (gpu_dict,)
        replicas = replicas[:len(scattered)]
        outputs = self.parallel_apply(replicas, scattered, gpu_dicts)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, input, device_ids):
        return scatter(input, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def data_parallel(module, inputs, device_ids, output_device=None, dim=0, module_kwargs=None):
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
        if module_kwargs is None:
            return module(*inputs)
        else:
            return module(*inputs, **module_kwargs)

    if output_device is None:
        output_device = device_ids[0]

    replicas = replicate(module, device_ids)
    scattered = scatter(inputs, device_ids, dim)

    gpu_dicts = None
    if module_kwargs:
        scatter_kwargs = {}
        for key in module_kwargs.keys():
            scatter_kwargs[key] = scatter(module_kwargs[key], device_ids, dim)
        gpu_dicts = tuple(
            {key: values[i] for key, values in scatter_kwargs.items()}
            for i in device_ids
        )

    replicas = replicas[:len(scattered)]
    outputs = parallel_apply(replicas, scattered, gpu_dicts)
    return gather(outputs, output_device, dim)
