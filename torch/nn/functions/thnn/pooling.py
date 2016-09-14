from torch.autograd.function import Function
from torch._thnn import type2backend

from . import _all_functions


class _MaxPoolingBase(Function):
    def __init__(self, *args):
        super(_MaxPoolingBase, self).__init__()
        self.additional_args = args[:-1]
        self.save_indices = args[-1]

    def _fw_call(self, *args):
        self._backend.SpatialDilatedMaxPooling_updateOutput(*args)

    def _bw_call(self, *args):
        self._backend.SpatialDilatedMaxPooling_updateGradInput(*args)

    def forward(self, input):
        self._backend = type2backend[type(input)]
        indices, output = input.new(), input.new()
        self._fw_call(self._backend.library_state, input, output, indices,
                *self.additional_args)
        if self.save_indices:
            self.save_for_backward(input, indices)
            self.mark_non_differentiable(indices)
            return output, indices
        else:
            self.save_for_backward(input)
            self.indices = indices
            return output

    def backward(self, grad_output, _indices_grad=None):
        if self.save_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices
        grad_input = grad_output.new()
        self._bw_call(self._backend.library_state, input, grad_output,
                grad_input, indices, *self.additional_args)
        return grad_input


class MaxPool1d(_MaxPoolingBase):
    def __init__(self, *args):
        super(MaxPool1d, self).__init__(*args)
        aa = self.additional_args
        self.additional_args = (aa[0], 1, aa[1], 1, aa[2], 0, aa[3], 1, aa[4])


class MaxPool2d(_MaxPoolingBase):
    pass


class MaxPool3d(_MaxPoolingBase):

    def _fw_call(self, *args):
        self._backend.VolumetricDilatedMaxPooling_updateOutput(*args)

    def _bw_call(self, *args):
        args = args[:5] + args[8:-1]
        self._backend.VolumetricDilatedMaxPooling_updateGradInput(*args)


class MaxUnpool2d(Function):
    def __init__(self, out_w, out_h):
        super(MaxUnpool2d, self).__init__()
        self.out_w = out_w
        self.out_h = out_h

    def forward(self, input, indices):
        self.save_for_backward(input, indices)
        self._backend = type2backend[type(input)]
        output = input.new()
        self._backend.SpatialMaxUnpooling_updateOutput(
                self._backend.library_state, input, output, indices,
                self.out_w, self.out_h)
        return output

    def backward(self, grad_output):
        input, indices = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.SpatialMaxUnpooling_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                indices, self.out_w, self.out_h)
        return grad_input, None


class MaxUnpool3d(Function):
    def __init__(self, out_t, out_w, out_h, *args):
        super(MaxUnpool3d, self).__init__()
        self.out_t = out_t
        self.out_w = out_w
        self.out_h = out_h
        self.args = args
        assert len(args) == 6

    def forward(self, input, indices):
        self.save_for_backward(input, indices)
        self._backend = type2backend[type(input)]
        output = input.new()
        self._backend.VolumetricMaxUnpooling_updateOutput(
                self._backend.library_state, input, output, indices,
                self.out_t, self.out_w, self.out_h, *self.args)
        return output

    def backward(self, grad_output):
        input, indices = self.saved_tensors
        grad_input = grad_output.new()
        self._backend.VolumetricMaxUnpooling_updateGradInput(
                self._backend.library_state, input, grad_output, grad_input,
                indices, self.out_t, self.out_w, self.out_h, *self.args)
        return grad_input, None


class FractionalMaxPool2d(Function):

    def __init__(self, kh, kw, output_size=None, output_ratio=None,
            return_indices=False, _random_samples=None):
        super(FractionalMaxPool2d, self).__init__()

        # Pool size (how wide the pooling for each output unit is)
        self.kw, self.kh = kw, kh

        # Random samples are drawn for all
        # batch * plane * (height, width; i.e., 2) points. This determines
        # the 2d "pseudorandom" overlapping pooling regions for each
        # (batch element x input plane).
        self.random_samples = _random_samples

        self.return_indices = return_indices

        if output_size is not None:
            self.oh, self.ow = output_size
            self.rh, self.rw = None, None
        elif output_ratio is not None:
            self.oh, self.ow = None, None
            self.rh, self.rw = output_ratio
            assert 0 < self.rh < 1
            assert 0 < self.rw < 1
        else:
            assert False

    def forward(self, input):
        if self.random_samples is None:
            random_samples = input.new().resize_(input.size(0),
                    input.size(1), 2).uniform_()
        else:
            random_samples = self.random_samples
            self.random_samples = None

        if self.oh is None:
            self.oh = int(input.size(2) * self.rh)
            self.ow = int(input.size(3) * self.rw)
        assert isinstance(self.oh, int) and isinstance(self.ow, int)

        indices = input.new()
        output = input.new()
        self._backend = type2backend[type(input)]
        self._backend.SpatialFractionalMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            output,
            self.ow, self.oh,
            self.kw, self.kh,
            indices,
            random_samples
        )

        self.random_samples = None # Free unnecessary buffers
        if self.return_indices:
            self.save_for_backward(input, indices)
            return output, indices
        else:
            self.indices = indices
            self.save_for_backward(input)
            return output

    def backward(self, grad_output, _grad_indices=None):
        if self.return_indices:
            input, indices = self.saved_tensors
        else:
            input, = self.saved_tensors
            indices = self.indices

        grad_input = grad_output.new()
        self._backend.SpatialFractionalMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            grad_output,
            grad_input,
            self.ow, self.oh,
            self.kw, self.kh,
            indices)

        return grad_input


_all_functions.append(MaxPool1d)
_all_functions.append(MaxPool2d)
_all_functions.append(MaxPool3d)
_all_functions.append(MaxUnpool2d)
_all_functions.append(MaxUnpool3d)
_all_functions.append(FractionalMaxPool2d)

