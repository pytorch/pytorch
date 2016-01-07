#include <THC/THC.h>
#include <THC/THCApply.cuh>

#define THIndexTensor THCudaTensor
#define THIndexTensor_(NAME) THCudaTensor_ ## NAME

#define THIntegerTensor THCudaTensor
#define THIntegerTensor_(NAME) THCudaTensor_ ## NAME

TH_API void THNN_CudaAbs_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaAbs_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

TH_API void THNN_CudaAbsCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaAbsCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);
TH_API void THNN_CudaClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);

TH_API void THNN_CudaDistKLDivCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *output,
          bool sizeAverage);
TH_API void THNN_CudaDistKLDivCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);

TH_API void THNN_CudaELU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          float alpha);
TH_API void THNN_CudaELU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output,
          float alpha);

TH_API void THNN_CudaHardTanh_updateOutput(
          THCState *state, 
          THCudaTensor *input,
          THCudaTensor *output,
          float min_val,
          float max_val);
TH_API void THNN_CudaHardTanh_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          float min_val,
          float max_val);

TH_API void THNN_CudaL1Cost_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaL1Cost_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

TH_API void THNN_CudaLeakyReLU_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          double negval, bool inplace);
TH_API void THNN_CudaLeakyReLU_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          double negval,
          bool inplace);

TH_API void THNN_CudaLogSigmoid_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output,
          THCudaTensor *buffer);
TH_API void THNN_CudaLogSigmoid_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *buffer);

TH_API void THNN_CudaLogSoftMax_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaLogSoftMax_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

TH_API void THNN_CudaLookupTable_accGradParameters(
          THCState *state,
          THIndexTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradWeight,
          float scale,
          bool scaleGradByFreq,
          THIntegerTensor *count,
          THCudaTensor *sorted,
          THCudaTensor *indices);
