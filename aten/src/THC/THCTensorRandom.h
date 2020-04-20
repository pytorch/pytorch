#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include <THC/THCTensor.h>

#include <THC/generic/THCTensorRandom.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorRandom.h>
#include <THC/THCGenerateBoolType.h>

#include <ATen/CUDAGenerator.h>

THC_API void THCRandom_getRNGState(at::Generator gen_, THByteTensor *rng_state);
THC_API void THCRandom_setRNGState(at::Generator gen_, THByteTensor *rng_state);

#endif
