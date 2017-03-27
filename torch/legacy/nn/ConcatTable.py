import torch
from .Container import Container


class ConcatTable(Container):

    def __init__(self, ):
        super(ConcatTable, self).__init__()
        self.modules = []
        self.output = []

    def updateOutput(self, input):
        self.output = [module.updateOutput(input) for module in self.modules]
        return self.output

    def _map_list(self, l1, l2, f):
        for i, v in enumerate(l2):
            if isinstance(v, list):
                res = self._map_list(l1[i] if i < len(l1) else [], v, f)
                if i >= len(l1):
                    assert i == len(l1)
                    l1.append(res)
                else:
                    l1[i] = res
            else:
                f(l1, i, v)
        for i in range(len(l1) - 1, len(l2) - 1, -1):
            del l1[i]
        return l1

    def _backward(self, method, input, gradOutput, scale=1):
        isTable = isinstance(input, list)
        wasTable = isinstance(self.gradInput, list)
        if isTable:
            for i, module in enumerate(self.modules):
                if method == 'updateGradInput':
                    currentGradInput = module.updateGradInput(input, gradOutput[i])
                elif method == 'backward':
                    currentGradInput = module.backward(input, gradOutput[i], scale)
                if not isinstance(currentGradInput, list):
                    raise RuntimeError("currentGradInput is not a table!")

                if len(input) != len(currentGradInput):
                    raise RuntimeError("table size mismatch")

                if i == 0:
                    self.gradInput = self.gradInput if wasTable else []

                    def fn(l, i, v):
                        if i >= len(l):
                            assert len(l) == i
                            l.append(v.clone())
                        else:
                            l[i].resize_as_(v)
                            l[i].copy_(v)
                    self._map_list(self.gradInput, currentGradInput, fn)
                else:
                    def fn(l, i, v):
                        if i < len(l):
                            l[i].add_(v)
                        else:
                            assert len(l) == i
                            l.append(v.clone())
                    self._map_list(self.gradInput, currentGradInput, fn)
        else:
            self.gradInput = self.gradInput if not wasTable else input.clone()
            for i, module in enumerate(self.modules):
                if method == 'updateGradInput':
                    currentGradInput = module.updateGradInput(input, gradOutput[i])
                elif method == 'backward':
                    currentGradInput = module.backward(input, gradOutput[i], scale)
                if i == 0:
                    self.gradInput.resize_as_(currentGradInput).copy_(currentGradInput)
                else:
                    self.gradInput.add_(currentGradInput)

        return self.gradInput

    def updateGradInput(self, input, gradOutput):
        return self._backward('updateGradInput', input, gradOutput)

    def backward(self, input, gradOutput, scale=1):
        return self._backward('backward', input, gradOutput, scale)

    def accGradParameters(self, input, gradOutput, scale=1):
        for i, module in ipairs(self.modules):
            self.rethrowErrors(module, i, 'accGradParameters', input, gradOutput[i], scale)

    def accUpdateGradParameters(self, input, gradOutput, lr):
        for i, module in ipairs(self.modules):
            self.rethrowErrors(module, i, 'accUpdateGradParameters', input, gradOutput[i], lr)

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = '  |`-> '
        ext = '  |    '
        extlast = '       '
        last = '   +. -> '
        res = torch.typename(self)
        res = res + ' {' + line + tab + 'input'
        for i in range(len(self.modules)):
            if i == len(self.modules) - 1:
                res = res + line + tab + next + '(' + str(i) + '): ' + \
                    str(self.modules[i]).replace(line, line + tab + extlast)
            else:
                res = res + line + tab + next + '(' + str(i) + '): ' + \
                    str(self.modules[i]).replace(line, line + tab + ext)

        res = res + line + tab + last + 'output'
        res = res + line + '}'
        return res
