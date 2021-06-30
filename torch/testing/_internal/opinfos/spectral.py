import numpy as np
from torch.testing import floating_and_complex_types
from torch.testing._internal.common_utils import make_tensor, slowTest
from torch.testing._internal.common_device_type import \
    (skipCPUIfNoMkl, skipCUDAIfRocm)
from .core import OpInfo, SampleInput, DecorateInfo, S, _getattr_qual


def sample_inputs_spectral_ops(self, device, dtype, requires_grad=False, **kwargs):
    nd_tensor = make_tensor((S, S + 1, S + 2), device, dtype, low=None, high=None,
                            requires_grad=requires_grad)
    tensor = make_tensor((31,), device, dtype, low=None, high=None,
                         requires_grad=requires_grad)

    if self.ndimensional:
        return [
            SampleInput(nd_tensor, kwargs=dict(s=(3, 10), dim=(1, 2), norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(s=(8,))),
            SampleInput(tensor),

            *(SampleInput(nd_tensor, kwargs=dict(dim=dim))
                for dim in [-1, -2, -3, (0, -1)]),
        ]
    else:
        return [
            SampleInput(nd_tensor, kwargs=dict(n=10, dim=1, norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(norm='ortho')),
            SampleInput(nd_tensor, kwargs=dict(n=7)),
            SampleInput(tensor),

            *(SampleInput(nd_tensor, kwargs=dict(dim=dim))
                for dim in [-1, -2, -3]),
        ]

# Metadata class for Fast Fourier Transforms in torch.fft.
class SpectralFuncInfo(OpInfo):
    """Operator information for torch.fft transforms. """

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref=None,  # Reference implementation (probably in np.fft namespace)
                 dtypes=floating_and_complex_types(),
                 ndimensional: bool,  # Whether dim argument can be a tuple
                 sample_inputs_func=sample_inputs_spectral_ops,
                 decorators=None,
                 **kwargs):
        decorators = list(decorators) if decorators is not None else []
        decorators += [
            skipCPUIfNoMkl,
            skipCUDAIfRocm,
            # gradgrad is quite slow
            DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'),
        ]

        super().__init__(name=name,
                         dtypes=dtypes,
                         decorators=decorators,
                         sample_inputs_func=sample_inputs_func,
                         **kwargs)
        self.ref = ref if ref is not None else _getattr_qual(np, name)
        self.ndimensional = ndimensional
