from .module import Module
from .utils import _quadruple, _ntuple
from .._functions.padding import ConstantPad2d as F_ConstantPad2d

# TODO: grad_output size asserts in THNN


class ReflectionPad2d(Module):

    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return self._backend.ReflectionPad2d(*self.padding)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ' + str(self.padding)


class ReplicationPad2d(Module):

    def __init__(self, padding):
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return self._backend.ReplicationPad2d(*self.padding)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ' + str(self.padding)


class ReplicationPad3d(Module):

    def __init__(self, padding):
        super(ReplicationPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, input):
        return self._backend.ReplicationPad3d(*self.padding)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ' + str(self.padding)


class ZeroPad2d(Module):

    def __init__(self, padding):
        super(ZeroPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return F_ConstantPad2d(pad=self.padding, value=0)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ' + str(self.padding)

   
class ConstantPad2d(Module):

    def __init__(self, padding, value):
        super(ConstantPad2d, self).__init__()
        self.padding = _quadruple(padding)
        self.valud = value

    def forward(self, input):
        return F_ConstantPad2d(pad=self.padding, value=self.value)(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ' + str(self.padding)
