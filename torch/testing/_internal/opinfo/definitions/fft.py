import unittest
from typing import List

import numpy as np

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater
from torch.testing._internal.common_device_type import precisionOverride
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
)
from torch.testing._internal.common_utils import TEST_SCIPY, TEST_WITH_ROCM
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    OpInfo,
    SampleInput,
    SpectralFuncInfo,
    SpectralFuncType,
)

has_scipy_fft = False
if TEST_SCIPY:
    try:
        import scipy.fft

        has_scipy_fft = True
    except ModuleNotFoundError:
        pass


def sample_inputs_fftshift(op_info, device, dtype, requires_grad, **kwargs):
    def mt(shape, **kwargs):
        return make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )

    yield SampleInput(mt((9, 10)))
    yield SampleInput(mt((50,)), kwargs=dict(dim=0))
    yield SampleInput(mt((5, 11)), kwargs=dict(dim=(1,)))
    yield SampleInput(mt((5, 6)), kwargs=dict(dim=(0, 1)))
    yield SampleInput(mt((5, 6, 2)), kwargs=dict(dim=(0, 2)))


# Operator database
op_db: List[OpInfo] = [
    SpectralFuncInfo(
        "fft.fft",
        aten_name="fft_fft",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.fft,
        ndimensional=SpectralFuncType.OneD,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
    ),
    SpectralFuncInfo(
        "fft.fft2",
        aten_name="fft_fft2",
        ref=np.fft.fft2,
        decomp_aten_name="_fft_c2c",
        ndimensional=SpectralFuncType.TwoD,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})],
    ),
    SpectralFuncInfo(
        "fft.fftn",
        aten_name="fft_fftn",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.fftn,
        ndimensional=SpectralFuncType.ND,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})],
    ),
    SpectralFuncInfo(
        "fft.hfft",
        aten_name="fft_hfft",
        decomp_aten_name="_fft_c2r",
        ref=np.fft.hfft,
        ndimensional=SpectralFuncType.OneD,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        check_batched_gradgrad=False,
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
        ),
    ),
    SpectralFuncInfo(
        "fft.hfft2",
        aten_name="fft_hfft2",
        decomp_aten_name="_fft_c2r",
        ref=scipy.fft.hfft2 if has_scipy_fft else None,
        ndimensional=SpectralFuncType.TwoD,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_gradgrad=False,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
            ),
        ),
    ),
    SpectralFuncInfo(
        "fft.hfftn",
        aten_name="fft_hfftn",
        decomp_aten_name="_fft_c2r",
        ref=scipy.fft.hfftn if has_scipy_fft else None,
        ndimensional=SpectralFuncType.ND,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_gradgrad=False,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),
                "TestFFT",
                "test_reference_nd",
            ),
        ],
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
            ),
        ),
    ),
    SpectralFuncInfo(
        "fft.rfft",
        aten_name="fft_rfft",
        decomp_aten_name="_fft_r2c",
        ref=np.fft.rfft,
        ndimensional=SpectralFuncType.OneD,
        dtypes=all_types_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (TEST_WITH_ROCM or not SM53OrLater) else (torch.half,))
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,
        skips=(),
        check_batched_gradgrad=False,
    ),
    SpectralFuncInfo(
        "fft.rfft2",
        aten_name="fft_rfft2",
        decomp_aten_name="_fft_r2c",
        ref=np.fft.rfft2,
        ndimensional=SpectralFuncType.TwoD,
        dtypes=all_types_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (TEST_WITH_ROCM or not SM53OrLater) else (torch.half,))
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        decorators=[
            precisionOverride({torch.float: 1e-4}),
        ],
    ),
    SpectralFuncInfo(
        "fft.rfftn",
        aten_name="fft_rfftn",
        decomp_aten_name="_fft_r2c",
        ref=np.fft.rfftn,
        ndimensional=SpectralFuncType.ND,
        dtypes=all_types_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (TEST_WITH_ROCM or not SM53OrLater) else (torch.half,))
        ),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        decorators=[
            precisionOverride({torch.float: 1e-4}),
        ],
    ),
    SpectralFuncInfo(
        "fft.ifft",
        aten_name="fft_ifft",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.ifft,
        ndimensional=SpectralFuncType.OneD,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
    ),
    SpectralFuncInfo(
        "fft.ifft2",
        aten_name="fft_ifft2",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.ifft2,
        ndimensional=SpectralFuncType.TwoD,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    SpectralFuncInfo(
        "fft.ifftn",
        aten_name="fft_ifftn",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.ifftn,
        ndimensional=SpectralFuncType.ND,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    SpectralFuncInfo(
        "fft.ihfft",
        aten_name="fft_ihfft",
        decomp_aten_name="_fft_r2c",
        ref=np.fft.ihfft,
        ndimensional=SpectralFuncType.OneD,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (TEST_WITH_ROCM or not SM53OrLater) else (torch.half,))
        ),
        skips=(),
        check_batched_grad=False,
    ),
    SpectralFuncInfo(
        "fft.ihfft2",
        aten_name="fft_ihfft2",
        decomp_aten_name="_fft_r2c",
        ref=scipy.fft.ihfftn if has_scipy_fft else None,
        ndimensional=SpectralFuncType.TwoD,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (TEST_WITH_ROCM or not SM53OrLater) else (torch.half,))
        ),
        check_batched_grad=False,
        check_batched_gradgrad=False,
        decorators=(
            # The values for attribute 'shape' do not match: torch.Size([5, 6, 5]) != torch.Size([5, 6, 6]).
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warning"),
            DecorateInfo(
                precisionOverride({torch.float: 2e-4}), "TestFFT", "test_reference_nd"
            ),
            # Mismatched elements!
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out"),
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warnings"),
        ),
    ),
    SpectralFuncInfo(
        "fft.ihfftn",
        aten_name="fft_ihfftn",
        decomp_aten_name="_fft_r2c",
        ref=scipy.fft.ihfftn if has_scipy_fft else None,
        ndimensional=SpectralFuncType.ND,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archss
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (TEST_WITH_ROCM or not SM53OrLater) else (torch.half,))
        ),
        check_batched_grad=False,
        check_batched_gradgrad=False,
        decorators=[
            # The values for attribute 'shape' do not match: torch.Size([5, 6, 5]) != torch.Size([5, 6, 6]).
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warning"),
            # Mismatched elements!
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out"),
            DecorateInfo(
                precisionOverride({torch.float: 2e-4}), "TestFFT", "test_reference_nd"
            ),
        ],
    ),
    SpectralFuncInfo(
        "fft.irfft",
        aten_name="fft_irfft",
        decomp_aten_name="_fft_c2r",
        ref=np.fft.irfft,
        ndimensional=SpectralFuncType.OneD,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        check_batched_gradgrad=False,
    ),
    SpectralFuncInfo(
        "fft.irfft2",
        aten_name="fft_irfft2",
        decomp_aten_name="_fft_c2r",
        ref=np.fft.irfft2,
        ndimensional=SpectralFuncType.TwoD,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        check_batched_gradgrad=False,
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    SpectralFuncInfo(
        "fft.irfftn",
        aten_name="fft_irfftn",
        decomp_aten_name="_fft_c2r",
        ref=np.fft.irfftn,
        ndimensional=SpectralFuncType.ND,
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # rocFFT doesn't support Half/Complex Half Precision FFT
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(
                ()
                if (TEST_WITH_ROCM or not SM53OrLater)
                else (torch.half, torch.complex32)
            ),
        ),
        check_batched_gradgrad=False,
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    OpInfo(
        "fft.fftshift",
        dtypes=all_types_and_complex_and(
            torch.bool, torch.bfloat16, torch.half, torch.chalf
        ),
        sample_inputs_func=sample_inputs_fftshift,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    OpInfo(
        "fft.ifftshift",
        dtypes=all_types_and_complex_and(
            torch.bool, torch.bfloat16, torch.half, torch.chalf
        ),
        sample_inputs_func=sample_inputs_fftshift,
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
]
