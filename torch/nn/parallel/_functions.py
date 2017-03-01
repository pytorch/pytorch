import torch.cuda.comm as comm
from torch.autograd import Function


class Broadcast(Function):

    def __init__(self, target_gpus):
        super(Broadcast, self).__init__()
        self.target_gpus = target_gpus

    def forward(self, *inputs):
        if not all(input.is_cuda for input in inputs):
            raise TypeError('Broadcast function not implemented for CPU tensors')
        if len(inputs) == 0:
            return tuple()
        self.input_device = inputs[0].get_device()
        outputs = comm.broadcast_coalesced(inputs, self.target_gpus)
        return tuple([t for tensors in outputs for t in tensors])

    def backward(self, *grad_outputs):
        grad_outputs = [grad_outputs[i:i + self.num_inputs]
                        for i in range(0, len(grad_outputs), self.num_inputs)]
        return comm.reduce_add_coalesced(grad_outputs, self.input_device)


class Gather(Function):

    def __init__(self, target_device, dim=0):
        super(Gather, self).__init__()
        self.target_device = target_device
        self.dim = dim

    def forward(self, *inputs):
        assert all(map(lambda i: i.is_cuda, inputs))
        self.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
        self.input_sizes = tuple(map(lambda i: i.size(self.dim), inputs))
        return comm.gather(inputs, self.dim, self.target_device)

    def backward(self, grad_output):
        return comm.scatter(grad_output, self.input_gpus, self.input_sizes,
                            self.dim)


class Scatter(Function):

    def __init__(self, target_gpus, chunk_sizes=None, dim=0):
        super(Scatter, self).__init__()
        self.target_gpus = target_gpus
        self.chunk_sizes = chunk_sizes
        self.dim = dim

    def forward(self, input):
        self.input_device = input.get_device() if input.is_cuda else -1
        return comm.scatter(input, self.target_gpus, self.chunk_sizes, self.dim)

    def backward(self, *grad_output):
        return comm.gather(grad_output, self.dim, self.input_device)
