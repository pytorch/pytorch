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

TH_API void THNN_CudaVolumetricAveragePooling_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int kT, int kW, int kH,
          int dT, int dW, int dH);
TH_API void THNN_CudaVolumetricAveragePooling_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH);

TH_API void THNN_CudaVolumetricConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH);
TH_API void THNN_CudaVolumetricConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *finput,
          int dT, int dW, int dH,
          int padT, int padW, int padH);
TH_API void THNN_CudaVolumetricConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          float scale);

TH_API void THNN_CudaVolumetricFullConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int adjT, int adjW, int adjH);
TH_API void THNN_CudaVolumetricFullConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int adjT, int adjW, int adjH);
TH_API void THNN_CudaVolumetricFullConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *finput,
          THCudaTensor *fgradInput,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int adjT, int adjW, int adjH,
          float scale);

TH_API void THNN_CudaVolumetricDilatedConvolution_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *weight,
          THCudaTensor *bias,
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_CudaVolumetricDilatedConvolution_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *weight,
          THCudaTensor *gradColumns,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_CudaVolumetricDilatedConvolution_accGradParameters(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          THCudaTensor *gradBias,
          THCudaTensor *columns,
          THCudaTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH,
          float scale);

TH_API void THNN_CudaVolumetricReplicationPadding_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          int pleft, int pright,
          int ptop, int pbottom,
          int pfront, int pback);
TH_API void THNN_CudaVolumetricReplicationPadding_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          int pleft, int pright,
          int ptop, int pbottom,
          int pfront, int pback);

#include "generic/THCUNN.h"
#include "THCGenerateFloatTypes.h"
