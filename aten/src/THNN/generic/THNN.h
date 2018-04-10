#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THNN.h"
#else

TH_API void THNN_(Abs_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output);           // [OUT] Abs output
TH_API void THNN_(Abs_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. output
          THTensor *gradInput);        // [OUT] gradient w.r.t. input

TH_API void THNN_(AbsCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // tensor with target values
          THTensor *output,            // [OUT] a one-element tensor with loss
          bool sizeAverage,            // if true, the loss will be divided by batch size
          bool reduce);
TH_API void THNN_(AbsCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // tensor with target values
          THTensor *gradOutput,
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          bool sizeAverage,            // if true, the gradient will be normalized by batch size
          bool reduce);

TH_API void THNN_(BCECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          THTensor *weights,           // [OPTIONAL]
          bool reduce);
TH_API void THNN_(BCECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          THTensor *weights,           // [OPTIONAL]
          bool reduce);

TH_API void THNN_(ClassNLLCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (1D/2D)
          THIndexTensor *target,       // tensor containing indexes of target classes
          THTensor *output,            // [OUT] a one-element tensor with loss
          bool sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight,      // [BUFFER]
          int64_t ignore_index,        // target index to ignore (loss = 0, gradInput = 0)
          bool reduce);
TH_API void THNN_(ClassNLLCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (1D/2D)
          THIndexTensor *target,       // tensor containing indexes of target classes
          THTensor *gradOutput,
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          bool sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight,      // [BUFFER]
          int64_t ignore_index,        // target index to ignore (loss = 0, gradInput = 0)
          bool reduce);

TH_API void THNN_(SpatialClassNLLCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (4D)
          THIndexTensor *target,       // tensor containing indexes of target classes (3D)
          THTensor *output,            // [OUT] a one-element tensor with loss
          bool sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight,      // [BUFFER]
          int64_t ignore_index,        // target index to ignore (loss = 0, gradInput = 0)
          bool reduce);

TH_API void THNN_(SpatialClassNLLCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor (4D)
          THIndexTensor *target,       // tensor containing indexes of target classes (3D)
          THTensor *gradOutput,
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          bool sizeAverage,            // if true, the loss will be normalized by batch size and class weights
          THTensor *weights,           // [OPTIONAL] class weights
          THTensor *total_weight,      // [BUFFER]
          int64_t ignore_index,        // target index to ignore (loss = 0, gradInput = 0)
          bool reduce);

TH_API void THNN_(ELU_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] ELU output
          accreal alpha,               // an ELU parameter (as in paper)
          accreal scale,               // scaling factor
          bool inplace);               // if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)
TH_API void THNN_(ELU_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *gradOutput,        // gradient w.r.t. output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          THTensor *output,            // output from a forward pass
          accreal alpha,               // an ELU parameter (as in paper)
          accreal scale);

TH_API void THNN_(DistKLDivCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor
          THTensor *output,            // [OUT] a one-element tensor containing the loss
          bool sizeAverage,            // if true, the loss will be normalized **by total number of elements**
          bool reduce);                // if true, returns summed or averaged loss. if false, returns a loss per element.
TH_API void THNN_(DistKLDivCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor
          THTensor *gradOutput,        // grad output tensor
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          bool sizeAverage,            // if true, the loss will be normalized **by total number of elements**
          bool reduce);                // if true, returns summed or averaged loss. if false, returns a loss per element.

TH_API void THNN_(GatedLinear_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor, half size of input along dimension dim
          int dim);                    // dimension for halving operation
TH_API void THNN_(GatedLinear_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t input
          int dim);                    // dimension for halving operation

// HardShink outputs 0 on interval of (-lambda; lambda) or original value otherwise.
TH_API void THNN_(HardShrink_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor
          accreal lambda);             // HardShrink parameter
TH_API void THNN_(HardShrink_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          accreal lambda);             // HardShrink parameter

// HardTanh clamps the values to the interval [min_val; max_val].
TH_API void THNN_(HardTanh_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor
          accreal min_val,             // lower threshold
          accreal max_val,             // upper threshold
          bool inplace);
TH_API void THNN_(HardTanh_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. the input
          accreal min_val,             // lower threshold
          accreal max_val,             // upper threshold
          bool inplace);

TH_API void THNN_(Im2Col_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kH, int kW,
          int dH, int dW,
          int padH, int padW,
          int sH, int sW);

TH_API void THNN_(Im2Col_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int inputHeight, int inputWidth,
          int kH, int kW,
          int dH, int dW,
          int padH, int padW,
          int sH, int sW);

TH_API void THNN_(Col2Im_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int outputHeight, int outputWidth,
          int kH, int kW,
          int dH, int dW,
          int padH, int padW,
          int sH, int sW);

TH_API void THNN_(Col2Im_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kH, int kW,
          int dH, int dW,
          int padH, int padW,
          int sH, int sW);

TH_API void THNN_(L1Cost_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output);           // [OUT] output tensor
TH_API void THNN_(L1Cost_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // [OPTIONAL] gradient w.r.t module's output
          THTensor *gradInput);        // [OUT] gradient w.r.t the input

TH_API void THNN_(LeakyReLU_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // [MODIFIED] input tensor
          THTensor *output,            // [OUT] output tensor
          accreal negval,              // negative part slope
          bool inplace);               // if true, modifies the input tensor and sets the output tensor on it (no additional memory is allocated)
TH_API void THNN_(LeakyReLU_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // [MODIFIED] gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. the input
          accreal negval,              // negative part slope
          bool inplace);               // if true, modifies gradOutput and sets gradInput onto it (no additional memory is allocated)

TH_API void THNN_(GRUFused_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *hidden,
          THTensor *bias1, // [OPTIONAL]
          THTensor *bias2, // [OPTIONAL]
          THTensor *hx,
          THTensor *output,
          THTensor *storage);
TH_API void THNN_(GRUFused_updateGradInput)(
          THNNState *state,
          THTensor *gradInInput,
          THTensor *gradInHidden,
          THTensor *gradOutput,
          THTensor *gradInputHx,
          THTensor *storage);

TH_API void THNN_(LSTMFused_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *hidden,
          THTensor *bias1, // [OPTIONAL]
          THTensor *bias2, // [OPTIONAL]
          THTensor *cell,
          THTensor *output,
          THTensor *outputCell);
TH_API void THNN_(LSTMFused_updateGradInput)(
          THNNState *state,
          THTensor *storage,
          THTensor *gradInGates,
          THTensor *cx,
          THTensor *cy,
          THTensor *gradOutput,
          THTensor *gradOutputCell,
          THTensor *gradInputCx);

TH_API void THNN_(LogSigmoid_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // output tensor
          THTensor *buffer);           // [BUFFER]
TH_API void THNN_(LogSigmoid_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          THTensor *buffer);           // [BUFFER]

TH_API void THNN_(LogSoftMax_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *output,            // [OUT] output tensor
          int64_t dim);
TH_API void THNN_(LogSoftMax_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *gradOutput,        // gradient w.r.t. module's output
          THTensor *gradInput,         // [OUT] gradient w.r.t. input
          THTensor *output,            // module's output
          int64_t dim);

TH_API void THNN_(LookupTable_accGradParameters)(
          THNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,            // [OPTIONAL]
          THIndexTensor *indices,      // [OPTIONAL]
          bool scaleGradByFreq,
          int paddingValue,
          accreal scale);

TH_API void THNN_(LookupTable_renorm)(
          THNNState *state,            // library's state
          THIndexTensor *idx,          // vector containing row indices (modified in function)
          THTensor *weight,            // 2D tensor whose rows will be renormalized
          accreal maxNorm,             // maximum norm
          accreal normType);           // the norm type (e.g., normType=2, then it's 2-norm)

TH_API void THNN_(MarginCriterion_updateOutput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor (should contain only 1s and -1s)
          THTensor *output,            // [OUT] a one-element tensor containing the loss
          bool sizeAverage,            // if true, the loss is normalized by **total number of elements**
          accreal margin);             // a margin that is required for the loss to be 0

TH_API void THNN_(MarginCriterion_updateGradInput)(
          THNNState *state,            // library's state
          THTensor *input,             // input tensor
          THTensor *target,            // target tensor (should contin only 1s and -1s)
          THTensor *gradInput,         // [OUT] gradient w.r.t. module's input
          bool sizeAverage,            // if true, the gradient is normalized by **total number of elements**
          accreal margin);             // a margin that is required for the loss to be 0

TH_API void THNN_(SoftMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          bool reduce);

TH_API void THNN_(SoftMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          bool reduce);

TH_API void THNN_(MSECriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          bool reduce);
TH_API void THNN_(MSECriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          bool reduce);

TH_API void THNN_(MultiLabelMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          THTensor *isTarget,
          bool sizeAverage,
          bool reduce);
TH_API void THNN_(MultiLabelMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *isTarget,
          bool sizeAverage,
          bool reduce);

TH_API void THNN_(MultiMarginCriterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *output,
          bool sizeAverage,
          int p,
          THTensor* weights,      // [OPTIONAL]
          accreal margin,
          bool reduce);
TH_API void THNN_(MultiMarginCriterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THIndexTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          int p,
          THTensor *weights,      // [OPTIONAL]
          accreal margin,
          bool reduce);

TH_API void THNN_(PReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight);
TH_API void THNN_(PReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight);
TH_API void THNN_(PReLU_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *gradWeight,
          accreal scale);

TH_API void THNN_(Linear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *addBuffer);
TH_API void THNN_(Linear_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight);
TH_API void THNN_(Linear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *addBuffer,
          accreal scale);

TH_API void THNN_(RReLU_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *noise,
          accreal lower,
          accreal upper,
          bool train,
          bool inplace,
          THGenerator *generator);
TH_API void THNN_(RReLU_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *noise,
          accreal lower,
          accreal upper,
          bool train,
          bool inplace);

TH_API void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(SmoothL1Criterion_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *output,
          bool sizeAverage,
          bool reduce);
TH_API void THNN_(SmoothL1Criterion_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *target,
          THTensor *gradOutput,
          THTensor *gradInput,
          bool sizeAverage,
          bool reduce);

TH_API void THNN_(SoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int64_t dim);
TH_API void THNN_(SoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          int64_t dim);

TH_API void THNN_(SoftPlus_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal beta,
          accreal threshold);
TH_API void THNN_(SoftPlus_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          accreal beta,
          accreal threshold);

TH_API void THNN_(SoftShrink_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal lambda);
TH_API void THNN_(SoftShrink_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal lambda);


TH_API void THNN_(IndexLinear_updateOutput)(
          THNNState *state,
          THIndexTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THIndexTensor *sizes,
          THIndexTensor *cumSumSizes,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *normalizedValues,
          int   train);
TH_API void THNN_(IndexLinear_accGradParameters)(
          THNNState *state,
          THIndexTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THIndexTensor *sizes,
          THIndexTensor *cumSumSizes,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THTensor* valuesBuffer,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(IndexLinear_accUpdateGradParameters)(
          THNNState *state,
          THIndexTensor *keys,
          int64_t keysOffset,
          THTensor *values,
          THIndexTensor *sizes,
          THIndexTensor *cumSumSizes,
          THTensor *gradOutput,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(IndexLinear_updateParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          THIndexTensor *runningKeys,
          THIndexTensor *cumSumSizes,
          int64_t keysOffset,
          accreal weightDecay,
          accreal learningRate);

TH_API void THNN_(SparseLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias);
TH_API void THNN_(SparseLinear_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(SparseLinear_zeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
TH_API void THNN_(SparseLinear_updateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          accreal learningRate);
TH_API void THNN_(SparseLinear_legacyUpdateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias);
TH_API void THNN_(SparseLinear_legacyAccGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *weight,
          THTensor *bias,
          accreal weightDecay,
          accreal scale);
TH_API void THNN_(SparseLinear_legacyZeroGradParameters)(
          THNNState *state,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput);
TH_API void THNN_(SparseLinear_legacyUpdateParameters)(
          THNNState *state,
          THTensor *weight,
          THTensor *bias,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *lastInput,
          accreal learningRate);

TH_API void THNN_(Sqrt_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal eps);
TH_API void THNN_(Sqrt_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(Square_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Square_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(Tanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output);
TH_API void THNN_(Tanh_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output);

TH_API void THNN_(Threshold_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal threshold,
          accreal val,
          bool inplace);
TH_API void THNN_(Threshold_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          accreal threshold,
          accreal val,
          bool inplace);

TH_API void THNN_(TemporalConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW, int dW,
          int inputFrameSize,
          int outputFrameSize);
TH_API void THNN_(TemporalConvolution_updateGradInput)(
          THNNState* state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW, int dW);
TH_API void THNN_(TemporalConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW, int dW,
          accreal scale);
TH_API void THNN_(TemporalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kW, int dW);
TH_API void THNN_(TemporalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kW, int dW);
TH_API void THNN_(TemporalSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW, int dW,
          int inputFrameSize);
TH_API void THNN_(TemporalSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW, int dW);
TH_API void THNN_(TemporalSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW, int dW,
          accreal scale);

TH_API void THNN_(TemporalRowConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int dW,
          int padW,
          bool featFirst);
TH_API void THNN_(TemporalRowConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int dW,
          int padW,
          bool featFirst);
TH_API void THNN_(TemporalRowConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW,
          int dW,
          int padW,
          bool featFirst,
          accreal scale);

TH_API void THNN_(TemporalUpSamplingNearest_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int scale_factor);
TH_API void THNN_(TemporalUpSamplingNearest_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int scale_factor);

TH_API void THNN_(TemporalUpSamplingLinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeW,
          bool align_corners);
TH_API void THNN_(TemporalUpSamplingLinear_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeW,
          int osizeW,
          bool align_corners);

TH_API void THNN_(BatchNormalization_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,       // [OPTIONAL]
          THTensor *bias,         // [OPTIONAL]
          THTensor *running_mean, // [OPTIONAL] if train
          THTensor *running_var,  // [OPTIONAL] if train
          THTensor *save_mean,
          THTensor *save_std,
          bool train,
          double momentum,
          double eps);
TH_API void THNN_(BatchNormalization_backward)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,    // [OPTIONAL]
          THTensor *gradWeight,   // [OPTIONAL]
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *weight,       // [OPTIONAL]
          THTensor *running_mean, // [OPTIONAL] if train
          THTensor *running_var,  // [OPTIONAL] if train
          THTensor *save_mean,    // [OPTIONAL] if !train
          THTensor *save_std,     // [OPTIONAL] if !train
          bool train,
          double scale,
          double eps);

TH_API void THNN_(SpatialConvolutionMap_updateOutput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *output,       // [OUT] convolution output
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialConvolutionMap_updateGradInput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradInput,    // [OUT] gradient w.r.t. input
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialConvolutionMap_accGradParameters)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradWeight,   // 3D gradWeight tensor (connTable:size(1) x kH x kW)
          THTensor *gradBias,     // 1D gradBias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH,         // stride
          accreal scale);         // scaling factor

TH_API void THNN_(SpatialConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH);
TH_API void THNN_(SpatialConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          accreal scale);

TH_API void THNN_(SpatialConvolutionLocal_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int64_t inputWidth, int64_t inputHeight,
          int64_t outputWidth, int64_t outputHeight);
TH_API void THNN_(SpatialConvolutionLocal_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int64_t inputWidth, int64_t inputHeight,
          int64_t outputWidth, int64_t outputHeight);
TH_API void THNN_(SpatialConvolutionLocal_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,
          THTensor *fgradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int64_t inputWidth, int64_t inputHeight,
          int64_t outputWidth, int64_t outputHeight,
          accreal scale);

TH_API void THNN_(SpatialAdaptiveMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int osizeW, int osizeH);
TH_API void THNN_(SpatialAdaptiveMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices);

TH_API void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeW, int osizeH);
TH_API void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(SpatialAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);
TH_API void THNN_(SpatialAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode,
          bool count_include_pad);

TH_API void THNN_(SpatialFractionalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int outputW, int outputH,
          int kW, int kH,
          THIndexTensor *indices,
          THTensor *randomSamples);
TH_API void THNN_(SpatialFractionalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int outputW, int outputH,
          int kW, int kH,
          THIndexTensor *indices);

TH_API void THNN_(SpatialFullConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
TH_API void THNN_(SpatialFullConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *columns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH);
TH_API void THNN_(SpatialFullConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int adjW, int adjH,
          accreal scale);

TH_API void THNN_(SpatialFullConvolutionMap_updateOutput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *output,       // [OUT] convolution output
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialFullConvolutionMap_updateGradInput)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradInput,    // [OUT] gradient w.r.t. input
          THTensor *weight,       // 3D weight tensor (connTable:size(1) x kH x kW)
          THTensor *bias,         // 1D bias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH);        // stride
TH_API void THNN_(SpatialFullConvolutionMap_accGradParameters)(
          THNNState *state,       // library state
          THTensor *input,        // input tensor
          THTensor *gradOutput,   // gradient w.r.t. output
          THTensor *gradWeight,   // 3D gradWeight tensor (connTable:size(1) x kH x kW)
          THTensor *gradBias,     // 1D gradBias tensor (nOutputPlane)
          THTensor *connTable,    // connection table
          int nInputPlane,        // number of input planes
          int nOutputPlane,       // number of output planes
          int dW, int dH,         // stride
          accreal scale);         // scaling factor

TH_API void THNN_(SpatialDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH);

TH_API void THNN_(SpatialDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *columns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH);

TH_API void THNN_(SpatialDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          accreal scale);

TH_API void THNN_(SpatialFullDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,         // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          int adjW, int adjH);

TH_API void THNN_(SpatialFullDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *columns,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          int adjW, int adjH);

TH_API void THNN_(SpatialFullDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,     // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          int adjW, int adjH,
          accreal scale);

TH_API void THNN_(SpatialMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);
TH_API void THNN_(SpatialMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          bool ceil_mode);

TH_API void THNN_(SpatialDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          bool ceil_mode);
TH_API void THNN_(SpatialDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int dilationW, int dilationH,
          bool ceil_mode);

TH_API void THNN_(SpatialMaxUnpooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int owidth, int oheight);
TH_API void THNN_(SpatialMaxUnpooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int owidth, int oheight);

TH_API void THNN_(SpatialSubSampling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          int kW, int kH,
          int dW, int dH);
TH_API void THNN_(SpatialSubSampling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          int kW, int kH,
          int dW, int dH);
TH_API void THNN_(SpatialSubSampling_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          int kW, int kH,
          int dW, int dH,
          accreal scale);

TH_API void THNN_(SpatialUpSamplingNearest_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int scale_factor);
TH_API void THNN_(SpatialUpSamplingNearest_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int scale_factor);

TH_API void THNN_(SpatialUpSamplingBilinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeH,
          int osizeW,
          bool align_corners);
TH_API void THNN_(SpatialUpSamplingBilinear_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeH,
          int isizeW,
          int osizeH,
          int osizeW,
          bool align_corners);

TH_API void THNN_(SpatialGridSamplerBilinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *grid,
          THTensor *output,
          int padding_mode);

TH_API void THNN_(SpatialGridSamplerBilinear_updateGradInput)(
          THNNState *state,
          THTensor *input, THTensor *gradInput,
          THTensor *grid, THTensor *gradGrid,
          THTensor *gradOutput,
          int padding_mode);

TH_API void THNN_(VolumetricGridSamplerBilinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *grid,
          THTensor *output,
          int padding_mode);

TH_API void THNN_(VolumetricGridSamplerBilinear_updateGradInput)(
          THNNState *state,
          THTensor *input, THTensor *gradInput,
          THTensor *grid, THTensor *gradGrid,
          THTensor *gradOutput,
          int padding_mode);

TH_API void THNN_(unfolded_acc)(
          THTensor *finput,
          THTensor *input,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int nInputPlane,
          int inputWidth, int inputHeight,
          int osizeW, int outputHeight);
TH_API void THNN_(unfolded_copy)(
          THTensor *finput,
          THTensor *input,
          int kW, int kH,
          int dW, int dH,
          int padW, int padH,
          int nInputPlane,
          int inputWidth, int inputHeight,
          int outputWidth, int outputHeight);

TH_API void THNN_(VolumetricAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          bool ceil_mode, bool count_include_pad);
TH_API void THNN_(VolumetricAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          bool ceil_mode, bool count_include_pad);

// VolumetricConvolution is legacy and purposefully not bound by ATen
TH_API void THNN_(VolumetricConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          accreal scale);

TH_API void THNN_(VolumetricConvolutionMM_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,     // HACK to make signature line up with backward
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolutionMM_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput,
          THTensor *fgradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricConvolutionMM_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *finput,
          THTensor *fgradInput,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          accreal scale);

TH_API void THNN_(VolumetricFractionalMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int outputT, int outputW, int outputH,
          int poolSizeT, int poolSizeW, int poolSizeH,
          THIndexTensor *indices,
          THTensor *randomSamples);
TH_API void THNN_(VolumetricFractionalMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int outputT, int outputW, int outputH,
          int poolSizeT, int poolSizeW, int poolSizeH,
          THIndexTensor *indices);

TH_API void THNN_(VolumetricFullConvolution_updateOutput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *output,         // [OUT] volumetric convolution output
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *bias,           // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // [OUT] internal columns buffer
          THTensor *fgradInput,     // [OUT] internal ones buffer
          int kT, int kW, int kH,   // kenerl size
          int dT, int dW, int dH,   // stride of the convolution
          int pT, int pW, int pH,   // padding
          int aT, int aW, int aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullConvolution_updateGradInput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradInput,      // [OUT] gradient w.r.t. input
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int kT, int kW, int kH,   // kenerl size
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int aT, int aW, int aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullConvolution_accGradParameters)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradWeight,     // gradWeight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *gradBias,       // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int kT, int kW, int kH,   // kenerl size
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int aT, int aW, int aH,   // extra output adjustment
          accreal scale);           // scaling factor

TH_API void THNN_(VolumetricDilatedConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,           // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_(VolumetricDilatedConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *columns,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH);

TH_API void THNN_(VolumetricDilatedConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,       // [OPTIONAL]
          THTensor *columns,
          THTensor *ones,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int padT, int padW, int padH,
          int dilationT, int dilationW, int dilationH,
          accreal scale);

TH_API void THNN_(VolumetricFullDilatedConvolution_updateOutput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *output,         // [OUT] volumetric convolution output
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *bias,           // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // [OUT] internal columns buffer
          THTensor *fgradInput,     // [OUT] internal ones buffer
          int kT, int kW, int kH,   // kernel size
          int dT, int dW, int dH,   // stride of the convolution
          int pT, int pW, int pH,   // padding
          int dilationT, int dilationW, int dilationH,
          int aT, int aW, int aH);  // extra output adjustment
TH_API void THNN_(VolumetricFullDilatedConvolution_updateGradInput)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradInput,      // [OUT] gradient w.r.t. input
          THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int kT, int kW, int kH,   // kernel size
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int dilationT, int dilationW, int dilationH,
          int aT, int aW, int aH);  // extra output adjustment

TH_API void THNN_(VolumetricFullDilatedConvolution_accGradParameters)(
          THNNState *state,         // library state
          THTensor *input,          // 4D or 5D (batch) tensor
          THTensor *gradOutput,     // gradient w.r.t. output
          THTensor *gradWeight,     // gradWeight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
          THTensor *gradBias,       // [OPTIONAL] gradBias tensor (nOutputPlane)
          THTensor *finput,         // internal columns buffer
          THTensor *fgradInput,     // internal ones buffer
          int kT, int kW, int kH,   // kernel size
          int dT, int dW, int dH,   // stride
          int pT, int pW, int pH,   // padding
          int dilationT, int dilationW, int dilationH,
          int aT, int aW, int aH,   // extra output adjustment
          accreal scale);           // scaling factor

TH_API void THNN_(VolumetricMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          bool ceilMode);
TH_API void THNN_(VolumetricMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          bool ceilMode);

TH_API void THNN_(VolumetricDilatedMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          int dilationT, int dilationW, int dilationH,
          bool ceilMode);
TH_API void THNN_(VolumetricDilatedMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          int dilationT, int dilationW, int dilationH,
          bool ceilMode);

TH_API void THNN_(VolumetricMaxUnpooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int oT, int oW, int oH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);
TH_API void THNN_(VolumetricMaxUnpooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices,
          int oT, int oW, int oH,
          int dT, int dW, int dH,
          int pT, int pW, int pH);

TH_API void THNN_(VolumetricAdaptiveAveragePooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeT,
          int osizeW,
          int osizeH);
TH_API void THNN_(VolumetricAdaptiveAveragePooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput);

TH_API void THNN_(VolumetricAdaptiveMaxPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *indices,
          int osizeT, int osizeW, int osizeH);
TH_API void THNN_(VolumetricAdaptiveMaxPooling_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THIndexTensor *indices);

TH_API void THNN_(SpatialReflectionPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int pad_left, int pad_right,
          int pad_top, int pad_bottom);

TH_API void THNN_(SpatialReflectionPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int pad_left, int pad_right,
          int pad_top, int pad_bottom);

TH_API void THNN_(SpatialReplicationPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int pad_left, int pad_right,
          int pad_top, int pad_bottom);

TH_API void THNN_(SpatialReplicationPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int pad_left, int pad_right,
          int pad_top, int pad_bottom);

TH_API void THNN_(FeatureLPPooling_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          accreal power,
          int width,
          int stride,
          bool batchMode);

TH_API void THNN_(FeatureLPPooling_updateGradInput)(
          THNNState *state,
          THTensor* gradOutput,
          THTensor* input,
          THTensor* output,
          THTensor* gradInput,
          accreal power,
          int width,
          int stride,
          bool batchMode);

TH_API void THNN_(VolumetricReplicationPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int pad_left, int pad_right,
          int pad_top, int pad_bottom,
          int pad_front, int pad_back);

TH_API void THNN_(VolumetricReplicationPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int pad_left, int pad_right,
          int pad_top, int pad_bottom,
          int pad_front, int pad_back);

TH_API void THNN_(VolumetricUpSamplingNearest_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int scale_factor);

TH_API void THNN_(VolumetricUpSamplingNearest_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int scale_factor);

TH_API void THNN_(VolumetricUpSamplingTrilinear_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int osizeT,
          int osizeH,
          int osizeW,
          bool align_corners);

TH_API void THNN_(VolumetricUpSamplingTrilinear_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          int isizeB,
          int isizeC,
          int isizeT,
          int isizeH,
          int isizeW,
          int osizeT,
          int osizeH,
          int osizeW,
          bool align_corners);

TH_API void THNN_(TemporalReflectionPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int pad_left, int pad_right);

TH_API void THNN_(TemporalReflectionPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int pad_left, int pad_right);

TH_API void THNN_(TemporalReplicationPadding_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int pad_left, int pad_right);

TH_API void THNN_(TemporalReplicationPadding_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          int pad_left, int pad_right);

#endif
