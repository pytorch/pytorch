#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPairwise.cu"
#else

#include <ATen/NamedTensorUtils.h>

static int THCTensor_(equalImpl)(THCState *state, THCTensor *self_, THCTensor *src_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (!THCTensor_(isSameSizeAs(state, self_, src_))) {
    return 0;
  }

  // This is not as efficient as TH, but the basic idea: create a buffer that stores
  // 1 if the two tensors are equal at a position, otherwise 0. If the minimum value
  // in this buffer is 1, the two tensors are equal, otherwise they are not

  // Both tensors are empty
  if(THTensor_(nElement)(self_) == 0) return true;

  THCudaByteTensor *buf = at::empty_like(THTensor_wrap(self_), at::kByte).unsafeReleaseTensorImpl();

  if (!THC_pointwiseApply3<uint8_t, scalar_t, scalar_t>(state, buf, self_, src_, TensorEQOp<scalar_t, unsigned char>())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  unsigned char min = THCudaByteTensor_minall(state, buf);

  THCudaByteTensor_free(state, buf);

  return min != 0;
}

int THCTensor_(equal)(THCState *state, THCTensor *self_, THCTensor *src_) {
  if (!at::namedinference::are_names_equal(self_, src_)) {
    return 0;
  }
  at::NoNamesGuard guard;
  return THCTensor_(equalImpl)(state, self_, src_);
}

#if !defined(THC_REAL_IS_BOOL)

void THCTensor_(mul)(THCState *state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorMulConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorMulConstantOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(fmod)(THCState *state, THCTensor *self_, THCTensor *src_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THC_pointwiseApply1<scalar_t>(state, self_, TensorFmodOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THCTensor_(resizeAs)(state, self_, src_);

    if (!THC_pointwiseApply2<scalar_t, scalar_t>(state, self_, src_, TensorFmodOp<scalar_t>(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

#endif

#endif
