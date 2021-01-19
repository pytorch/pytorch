#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorRandom.h"
#else

#include <ATen/core/Generator.h>

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_HALF)

TORCH_CUDA_API void THCTensor_(multinomialAliasSetup)(struct THCState *state, THCTensor *probs, THCudaLongTensor *J, THCTensor *q);
TORCH_CUDA_API void THCTensor_(multinomialAliasDraw)(THCState *state, THCudaLongTensor *self, THCTensor *_q, THCudaLongTensor *_J, int n_sample, c10::optional<at::Generator> gen_);

#endif
#endif
