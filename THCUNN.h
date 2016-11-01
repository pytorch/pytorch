#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THCIndexTensor THCudaLongTensor
#define THCIndexTensor_(NAME) THCudaLongTensor_ ## NAME
typedef long THCIndex_t;

#define THNN_(NAME) TH_CONCAT_3(THNN_, CReal, NAME)

TH_API void THNN_CudaLookupTable_accGradParameters(
          THCState *state,
          THCIndexTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCIndexTensor *count,
          THCIndexTensor *sorted,        // [OPTIONAL]
          THCIndexTensor *indices,       // [OPTIONAL]
          bool scaleGradByFreq,
          int paddingValue,
          float scale);

TH_API void THNN_CudaLookupTable_renorm(
          THCState *state,
          THCIndexTensor *idx,
          THCudaTensor *weight,
          float maxNorm,
          float normType);

TH_API void THNN_CudaTemporalConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          int kW, int dW,
          int inputFrameSize,
          int outputFrameSize);

TH_API void THNN_CudaTemporalConvolution_updateGradInput(
          THCState* state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          int kW, int dW);

TH_API void THNN_CudaTemporalConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          int kW, int dW,
          float scale);

#include "generic/THCUNN.h"
#include "THCGenerateFloatTypes.h"
