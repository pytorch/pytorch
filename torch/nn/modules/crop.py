import torch

from .module import Module
from .. import functional as F


class _CropBase(Module):

    def __init__(self, size, mode):
        self.size = size
        self.mode = mode
        if self.mode != 'center':
            raise NotImplementedError("CropNd only supports mode center crop")

    def _offset(self, initial, target):
        crop = torch.FloatTensor([initial]).sub(target).div(-2)
        return crop.ceil().int()[0], crop.floor().int()[0]

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'size=' + str(self.size) \
            + 'mode=' + self.mode + ')'


class Crop1d(_CropBase):

    def __init__(self, size, mode='center'):
        if isinstance(size, int):
            size = (size,)
        assert len(size) == 1, 'size should be int or one dimensional tuple'
        super(Crop1d, self).__init__(size, mode)

    def forward(self, x):
        length = x.size()[2]
        return F.pad(x, [*self._offset(length, self.size[0])])


class Crop2d(_CropBase):

    def __init__(self, size, mode='center'):
        if isinstance(size, int):
            size = (size, size)
        assert len(size) == 2, 'size should be int or two dimensional tuple'
        super(Crop2d, self).__init__(size, mode)

    def forward(self, x):
        height, width = x.size()[:2]
        return F.pad(x, [
            *self._offset(height, self.size[0]),
            *self._offset(width, self.size[1]),
        ])


class Crop3d(_CropBase):

    def __init__(self, size, mode='center'):
        if isinstance(size, int):
            size = (size, size, size)
        assert len(size) == 3, 'size should be int or three dimensional tuple'
        super(Crop3d, self).__init__(size, mode)

    def forward(self, x):
        height, width, depth = x.size()[:2]
        return F.pad(x, [
            *self._offset(height, self.size[0]),
            *self._offset(width, self.size[1]),
            *self._offset(depth, self.size[2]),
        ])
