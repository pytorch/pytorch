#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMasked.cu"
#else

#include <ATen/NamedTensorUtils.h>


void THCTensor_(maskedFill)(THCState* state,
                            THCTensor *tensor, THCudaByteTensor *mask, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, mask));
  THArgCheck(THCTensor_(nElement)(state, tensor) ==
             THCudaByteTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THC_pointwiseApply2<scalar_t, uint8_t>(state, tensor, mask,
                                          TensorMaskedFillOp<scalar_t, unsigned char>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(maskedFillBool)(THCState* state,
                                THCTensor *tensor, THCudaBoolTensor *mask, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, mask));
  THArgCheck(THCTensor_(nElement)(state, tensor) ==
             THCudaBoolTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THC_pointwiseApply2<scalar_t, bool>(state, tensor, mask,
                                           TensorMaskedFillOp<scalar_t, bool>(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(maskedFillByte)(THCState* state,
                                THCTensor *tensor, THByteTensor *mask, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, tensor));
  THCudaByteTensor* maskCuda = THTensor_wrap(mask).cuda().unsafeReleaseTensorImpl();
  THCTensor_(copy)(state, maskCuda, mask);
  THCTensor_(maskedFill)(state, tensor, maskCuda, value);
  THCudaByteTensor_free(state, maskCuda);
}

#endif
