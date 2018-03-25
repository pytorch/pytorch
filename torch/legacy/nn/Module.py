import torch
import torch._thnn
from .utils import clear, recursiveType


class Module(object):

    def __init__(self):
        self.gradInput = torch.Tensor()
        self.output = torch.Tensor()
        self._type = self.output.type()
        self._backend = torch._thnn.type2backend[self.output.type()]

    def __repr__(self):
        return 'nn.' + self.__class__.__name__

    def parameters(self):
        has_weight = hasattr(self, 'weight') and self.weight is not None
        has_bias = hasattr(self, 'bias') and self.bias is not None
        if has_weight and has_bias:
            return [self.weight, self.bias], [self.gradWeight, self.gradBias]
        elif has_weight:
            return [self.weight], [self.gradWeight]
        elif has_bias:
            return [self.bias], [self.gradBias]
        else:
            return

    def updateOutput(self, input):
        return self.output

    def forward(self, input):
        return self.updateOutput(input)

    def backward(self, input, gradOutput, scale=1):
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput, scale)
        return self.gradInput

    def backwardUpdate(self, input, gradOutput, lr):
        self.updateGradInput(input, gradOutput)
        self.accUpdateGradParameters(input, gradOutput, lr)
        return self.gradInput

    def updateGradInput(self, input, gradOutput):
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        pass

    def accUpdateGradParameters(self, input, gradOutput, lr):
        has_weight = hasattr(self, 'weight') and self.weight is not None
        has_bias = hasattr(self, 'bias') and self.bias is not None
        if has_weight:
            gradWeight = self.gradWeight
            self.gradWeight = self.weight
        if has_bias:
            gradBias = self.gradBias
            self.gradBias = self.bias
        self.accGradParameters(input, gradOutput, -lr)
        if has_weight:
            self.gradWeight = gradWeight
        if has_bias:
            self.gradBias = gradBias

    def sharedAccUpdateGradParameters(self, input, gradOutput, lr):
        if self.parameters():
            self.zeroGradParameters()
            self.accGradParameters(input, gradOutput, 1)
            self.updateParameters(lr)

    def zeroGradParameters(self):
        params = self.parameters()
        if params is not None:
            for grad in params[1]:
                grad.zero_()

    def updateParameters(self, learningRate):
        if self.parameters() is not None:
            params, gradParams = self.parameters()
            if params:
                for p, gp in zip(params, gradParams):
                    p.add_(-learningRate, gp)

    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False

    # TODO
    def share(self, mlp, *arg):
        raise NotImplementedError

    def clone(self, *arg):
        raise NotImplementedError

    def type(self, type=None, tensorCache=None):
        if type is None:
            return self._type

        tensorCache = tensorCache or {}

        # find all tensors and convert them
        for key, param in self.__dict__.items():
            setattr(self, key, recursiveType(param, type, tensorCache))

        self._backend = torch._thnn.type2backend[type]
        self._type = type
        return self

    def float(self, *args):
        return self.type('torch.FloatTensor', *args)

    def double(self, *args):
        return self.type('torch.DoubleTensor', *args)

    def cuda(self, *args):
        return self.type('torch.cuda.FloatTensor', *args)

    def reset(self):
        pass

    def write(self, f):
        raise NotImplementedError

    def read(self, f):
        raise NotImplementedError

    # This function is not easy to understand. It works as follows:
    #
    # - gather all parameter tensors for this module (and children);
    #   count all parameter values (floats)
    # - create one ginormous memory area (Storage object) with room for all
    #   parameters
    # - remap each parameter tensor to point to an area within the ginormous
    #   Storage, and copy it there
    #
    # It has the effect of making all parameters point to the same memory area,
    # which is: returned.
    #
    # The purpose is to allow operations over all parameters (such as momentum
    # updates and serialization), but it assumes that all parameters are of
    # the same type (and, in the case of CUDA, on the same device), which
    # is not always True. Use for_each() to iterate over this module and
    # children instead.
    #
    # Module._flattenTensorBuffer can be used by other packages (e.g. cunn)
    # to specify the type of temporary buffers. For example, the temporary
    # buffers for CudaTensor could be FloatTensor, to avoid GPU memory usage.
    #
    # TODO: This logically belongs to torch.Tensor, not nn.
    _flattenTensorBuffer = {}

    def _flatten(self, parameters=[]):

        # returns True if tensor occupies a contiguous region of memory (no holes)
        def isCompact(tensor):
            # isn't it enough to check if strides == size.cumprod(0)?
            sortedStride, perm = torch.sort(torch.LongTensor(tensor.stride()), 0, True)
            sortedSize = torch.LongTensor(list(tensor.size())).index_select(0, perm)
            nRealDim = int(torch.clamp(sortedStride, 0, 1).sum())
            sortedStride = sortedStride.narrow(0, 0, nRealDim).clone()
            sortedSize = sortedSize.narrow(0, 0, nRealDim).clone()
            t = tensor.new().set_(tensor.storage(), 0,
                                  tuple(sortedSize),
                                  tuple(sortedStride))
            return t.is_contiguous()

        if not parameters:
            return torch.Tensor()

        Tensor = parameters[0].new
        BufferTensor = Module._flattenTensorBuffer.get(type(parameters[0]), Tensor)

        # 1. construct the set of all unique storages referenced by parameter tensors
        storages = {}
        num_parameters = 0
        parameterMeta = []
        for i, param in enumerate(parameters):
            storage = param.storage()
            key = storage._cdata

            if key not in storages:
                storages[key] = (storage, num_parameters)
                num_parameters = num_parameters + storage.size()

            parameterMeta.append({
                'storage_offset': param.storage_offset() + storages[key][1],
                'size': param.size(),
                'stride': param.stride()
            })

        # 2. construct a single tensor that will hold all the parameters
        flatParameters = BufferTensor(num_parameters).zero_()

        # 3. determine if there are elements in the storage that none of the
        #    parameter tensors reference ('holes')
        tensorsCompact = True
        for meta in parameterMeta:
            tmp = BufferTensor().set_(flatParameters.storage(), meta['storage_offset'], meta['size'], meta['stride'])
            tmp.fill_(1)
            tensorsCompact = tensorsCompact and isCompact(tmp)

        maskParameters = flatParameters.byte().clone()
        compactOffsets = flatParameters.long().cumsum(0)
        used_parameters = compactOffsets[-1]

        # 4. copy storages into the flattened parameter tensor
        for storageAndOffset in storages.values():
            storage, offset = storageAndOffset
            flatParameters[slice(offset, offset + storage.size())].copy_(Tensor().set_(storage))

        # 5. allow garbage collection
        storages = None
        for param in parameters:
            param.set_()

        # 6. compact the flattened parameters if there were holes
        if used_parameters != num_parameters:
            assert tensorsCompact

            flatParameters = BufferTensor(used_parameters).copy_(
                flatParameters.masked_select(maskParameters))
            for meta in parameterMeta:
                meta['storage_offset'] = compactOffsets[meta['storage_offset']]

        if BufferTensor != Tensor:
            flatParameters = Tensor(flatParameters.nelement()).copy_(flatParameters)

        # 7. fix up the parameter tensors to point at the flattened parameters
        for param, meta in zip(parameters, parameterMeta):
            param.set_(flatParameters.storage(),
                       meta['storage_offset'],
                       meta['size'],
                       meta['stride'])

        return flatParameters

    def flattenParameters(self):
        _params = self.parameters()
        if _params is None:
            return
        parameters, gradParameters = _params
        p, g = self._flatten(parameters), self._flatten(gradParameters)

        assert p.nelement() == g.nelement()
        if parameters:
            for param, grad in zip(parameters, gradParameters):
                assert param.storage_offset() == grad.storage_offset()

        return p, g

    def apply(self, callback):
        callback(self)
        if hasattr(self, 'modules'):
            for module in self.modules:
                module.apply(callback)

    def findModules(self, cls, container=None):
        nodes = []
        containers = []
        if isinstance(self, cls):
            nodes.append(self)
            containers.append(container)

        # Recurse on nodes with 'modules'
        if hasattr(self, 'modules'):
            for child in self.modules:
                child_nodes, child_containers = child.findModules(cls, self)
                assert len(child_nodes) == len(child_containers)
                # add the list items from our child to our list (i.e. return a
                # flattened table of the return nodes).
                nodes.extend(child_nodes)
                containers.extend(child_containers)

        return nodes, containers

    def listModules(self):
        # include self first
        modules = [self]
        if hasattr(self, 'modules'):
            for child in self.modules:
                modules.extend(child.listModules())
        return modules

    def clearState(self):
        return clear(self, 'output', 'gradInput')

    def replace(self, callback):
        out = callback(self)
        # TODO: not out.modules?
        if hasattr(self, 'modules'):
            for i, module in enumerate(self.modules):
                self.modules[i] = module.replace(callback)
        return out
