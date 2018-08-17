import torch
from .Module import Module


class SpatialUpSamplingNearest(Module):
    """
    Applies a 2D up-sampling over an input image composed of several input planes.

    The upsampling is.ne using the simple nearest neighbor technique.

    The Y and X dimensions are assumed to be the last 2 tensor dimensions.  For
    instance, if the tensor is 4D,: dim 3 is the y dimension and dim 4 is the x.

    owidth  = width*scale_factor
    oheight  = height*scale_factor
    """

    def __init__(self, scale):
        super(SpatialUpSamplingNearest, self).__init__()

        self.scale_factor = scale
        if self.scale_factor < 1:
            raise ValueError('scale_factor must be greater than 1')
        if self.scale_factor % 1 != 0:
            raise ValueError('scale_factor must be integer')

    def updateOutput(self, input):
        assert input.dim() == 4

        # Copy the input size
        xdim = input.dim() - 1
        ydim = input.dim() - 2
        outputSize = list(input.size())
        outputSize[ydim] = outputSize[ydim] * self.scale_factor
        outputSize[xdim] = outputSize[xdim] * self.scale_factor
        self.output.resize_(*outputSize)
        self._backend.SpatialUpSamplingNearest_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            outputSize[ydim],
            outputSize[xdim]
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resize_as_(input)
        assert input.dim() == 4
        input_size = input.size()
        self._backend.SpatialUpSamplingNearest_updateGradInput(
            self._backend.library_state,
            gradOutput,
            self.gradInput,
            input_size[0],
            input_size[1],
            input_size[2],
            input_size[3],
            gradOutput.shape[2],
            gradOutput.shape[3]
        )
        return self.gradInput
