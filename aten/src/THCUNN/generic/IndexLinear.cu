#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/IndexLinear.cu"
#else

static bool THCUNN_checkKeysValues(THCState *state, THCudaLongTensor* keys,
                                   THCTensor* values)
{
    return THCudaLongTensor_size(state, keys, 0) == THCTensor_(nElement)(state, values)
        && THCTensor_(nDimension)(state, values) == 1
        && THCudaLongTensor_nDimension(state, keys) == 1;
}

void THNN_(IndexLinear_updateOutput)(
    THCState *state,
    THCudaLongTensor *keys,
    int64_t keysOffset,
    THCTensor *values,
    THCudaLongTensor *sizes,
    THCudaLongTensor *cumSumSizes,
    THCTensor *output,
    THCTensor *weight,
    THCTensor *bias,
    THCTensor *normalizedValues,
    int   train)
{
    // Make sure these inputs are contiguous to accelerate computations
    THArgCheck(THCudaLongTensor_isContiguous(state, keys), 1,
               "keys vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, values), 3,
               "values vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, sizes), 4,
               "sizes vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, cumSumSizes), 5,
               "cumSumSizes vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, output), 6,
               "output vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, weight), 7,
               "weight matrix must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, bias), 8,
               "bias vector must be contiguous");
    THArgCheck(THCUNN_checkKeysValues(state, keys, values), 1,
               "Keys and values should have the same number of elements");

    int64_t batchSize = sizes->size[0];
    int64_t outDim = bias->size[0];
    int64_t wDim = weight->size[1];
    int64_t weightStride = weight->stride[0];
    int maxNormalize = wDim - outDim;
    int64_t keysSize = keys->size[0];
    int64_t nnzPerRow = divup(keysSize, batchSize);

    THCTensor_(resize2d)(state, output, batchSize, outDim);
    int64_t *keysData        = THCudaLongTensor_data (state, keys);
    real *valuesData      = THCTensor_(data)      (state, values);
    int64_t *cumSumSizesData = THCudaLongTensor_data (state, cumSumSizes);
    real *biasData        = THCTensor_(data)      (state, bias);
    real *weightData      = THCTensor_(data)      (state, weight);
    real *outData         = THCTensor_(data)      (state, output);

    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(THREADS_X, THREADS_Y);
    int blocks_x = divup(outDim, threads.x);
    int blocks_y = batchSize;
    int nnzPerBlock = ((outDim == 1 || batchSize == 1) ? THREADS_X : NNZ_PER_BLOCK_MAX);
    int blocks_z = divup(nnzPerRow, nnzPerBlock);

    dim3 blocks(blocks_x, blocks_y, blocks_z);

    if (blocks_z > 1) {
        THCudaCheck(cudaMemsetAsync(outData, 0, outDim * batchSize * sizeof(real), stream));
    }

    real *normalizedValuesData = NULL;
    if (maxNormalize && train) {
        THCTensor_(resize1d)(state, normalizedValues, keysSize);
        normalizedValuesData = THCTensor_(data)(state, normalizedValues);
        updateOutput<real, true><<<blocks, threads, 0, stream>>>
            (outData, normalizedValuesData, valuesData, cumSumSizesData, keysData,
             batchSize, outDim, weightData, biasData, weightStride, keysOffset, maxNormalize, nnzPerBlock);
    } else {
        updateOutput<real, false><<<blocks, threads, 0, stream>>>
            (outData, normalizedValuesData, valuesData, cumSumSizesData, keysData,
             batchSize, outDim, weightData, biasData, weightStride, keysOffset, maxNormalize, nnzPerBlock);
    }
}

void THNN_(IndexLinear_accGradParameters)(
    THCState *state,
    THCudaLongTensor *keys,
    int64_t keysOffset,
    THCTensor *values,
    THCudaLongTensor *sizes,
    THCudaLongTensor *cumSumSizes,
    THCTensor *gradOutput,
    THCTensor *gradWeight,
    THCTensor *gradBias,
    THCTensor *weight,
    THCTensor *bias,
    THCTensor* valuesBuffer,
    accreal weightDecay,
    accreal scale)
{
    int64_t keysSize = keys->size[0];
    int64_t batchSize = sizes->size[0];
    int64_t outDim = bias->size[0];
    int64_t wDim = weight->size[1];
    int maxNormalize = wDim - outDim;

    // Make sure these inputs are contiguous to accelerate computations
    THArgCheck(THCudaLongTensor_isContiguous(state, keys), 1,
               "keys vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, values), 3,
               "values vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, sizes), 4,
               "sizes vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, cumSumSizes), 5,
               "cumSumSizes vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, gradOutput), 6,
               "gradOutput vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, gradWeight), 7,
               "gradWeight matrix must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, gradBias), 8,
               "gradBias vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, weight), 9,
               "weight matrix must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, bias), 10,
               "bias vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, valuesBuffer), 11,
               "valuesBuffer vector must be contiguous");
    THArgCheck(THCUNN_checkKeysValues(state, keys, values), 1,
               "Keys and values should have the same number of elements");

    THCTensor_(resize2d)(state, gradWeight, keysSize, outDim * (maxNormalize > 0 ? 2 : 1));

    real *valuesData      = THCTensor_(data)      (state, values);
    int64_t *cumSumSizesData = THCudaLongTensor_data (state, cumSumSizes);
    real *gradOutputData  = THCTensor_(data)      (state, gradOutput);
    real *gradBiasData    = THCTensor_(data)      (state, gradBias);
    real *gradWeightData  = THCTensor_(data)      (state, gradWeight);
    int64_t gradWeightStride = gradWeight->stride[0];

    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(THREADS_X, THREADS_Y);
    int blocks_x = divup(outDim, threads.x);
    accGradBias<real, false><<<blocks_x, threads, 0, stream>>>
        (gradBiasData, gradOutputData, outDim, batchSize, scale, weightDecay);

    dim3 blocks(blocks_x, batchSize);
    accGradWeight<real><<<blocks, threads, 0, stream>>>
        (gradWeightData, gradOutputData, valuesData, cumSumSizesData, outDim,
         gradWeightStride, scale, weightDecay, maxNormalize);
}

void THNN_(IndexLinear_accUpdateGradParameters)(
    THCState *state,
    THCudaLongTensor *keys,
    int64_t keysOffset,
    THCTensor *values,
    THCudaLongTensor *sizes,
    THCudaLongTensor *cumSumSizes,
    THCTensor *gradOutput,
    THCTensor *weight,
    THCTensor *bias,
    accreal weightDecay,
    accreal scale)
{
    // Make sure these inputs are contiguous to accelerate computations
    THArgCheck(THCudaLongTensor_isContiguous(state, keys), 1,
               "keys vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, values), 3,
               "values vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, sizes), 4,
               "sizes vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, cumSumSizes), 5,
               "cumSumSizes vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, gradOutput), 6,
               "gradOutput vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, weight), 7,
               "weight matrix must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, bias), 8,
               "bias vector must be contiguous");
    THArgCheck(THCUNN_checkKeysValues(state, keys, values), 1,
               "Keys and values should have the same number of elements");

    int64_t batchSize = sizes->size[0];
    int64_t outDim = bias->size[0];
    int64_t keysSize = keys->size[0];
    int64_t wDim = weight->size[1];
    int maxNormalize = wDim - outDim;

    real *biasData         = THCTensor_(data)      (state, bias);
    real *weightData       = THCTensor_(data)      (state, weight);
    real *gradOutputData   = THCTensor_(data)      (state, gradOutput);
    real *valuesData       = THCTensor_(data)      (state, values);
    int64_t *keysData         = THCudaLongTensor_data (state, keys);
    int64_t *cumSumSizesData  = THCudaLongTensor_data (state, cumSumSizes);
    int64_t weightStride = weight->stride[0];

    cudaStream_t stream = THCState_getCurrentStream(state);
    dim3 threads(THREADS_X, THREADS_Y);
    int blocks_x = divup(outDim, threads.x);

    accGradBias<real, true><<<blocks_x, threads, 0, stream>>>
        (biasData, gradOutputData, outDim, batchSize, scale, weightDecay);

    int64_t nnzPerRow = divup(keysSize, batchSize);
    int blocks_y = divup(nnzPerRow, REPEAT * threads.y);
    dim3 blocks(blocks_x, blocks_y);

    for (int64_t batchId = 0; batchId < batchSize; batchId++) {
        accUpdateWeight<real><<<blocks, threads, 0, stream>>>
            (weightData, weightStride, gradOutputData, outDim, valuesData,
             cumSumSizesData, keysData, keysOffset, scale, weightDecay, maxNormalize,
             batchId);
    }
}

void THNN_(IndexLinear_updateParameters)(
    THCState *state,
    THCTensor *gradWeight,
    THCTensor *gradBias,
    THCTensor *weight,
    THCTensor *bias,
    THCudaLongTensor *runningKeys,
    THCudaLongTensor *cumSumSizes,
    int64_t keysOffset,
    accreal weightDecay,
    accreal learningRate)
{
    // Make sure these inputs are contiguous to accelerate computations
    THArgCheck(THCTensor_(isContiguous)(state, gradWeight), 1,
               "gradWeight matrix must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, gradBias), 2,
               "gradBias vector must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, weight), 3,
               "weight matrix must be contiguous");
    THArgCheck(THCTensor_(isContiguous)(state, bias), 4,
               "bias vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, runningKeys), 5,
               "runningKeys vector must be contiguous");
    THArgCheck(THCudaLongTensor_isContiguous(state, cumSumSizes), 6,
               "cumSumSizes vector must be contiguous");

    int64_t outDim = bias->size[0];
    int64_t wDim = weight->size[1];
    int maxNormalize = wDim - outDim;
    int64_t keysSize = runningKeys->size[0];
    int64_t batchSize = cumSumSizes->size[0];

    THCTensor_(cadd)(state, bias, bias, -learningRate, gradBias);
    int64_t gradWeightStride = gradWeight->stride[0];
    int64_t weightStride = weight->stride[0];

    int64_t *keysData        = THCudaLongTensor_data (state, runningKeys);
    int64_t *cumSumSizesData = THCudaLongTensor_data (state, cumSumSizes);
    real *gradWeightData  = THCTensor_(data)      (state, gradWeight);
    real *weightData      = THCTensor_(data)      (state, weight);

    dim3 threads(THREADS_X, THREADS_Y);
    int64_t nnzPerRow = divup(keysSize, batchSize);
    int blocks_x = divup(outDim, threads.x);
    int blocks_y = divup(nnzPerRow, REPEAT * threads.y);
    dim3 blocks(blocks_x, blocks_y);
    cudaStream_t stream = THCState_getCurrentStream(state);

    for (int64_t batchId = 0; batchId < batchSize; batchId++) {
        updateWeight<real><<<blocks, threads, 0, stream>>>
            (weightData, gradWeightData, keysData, cumSumSizesData, outDim,
             gradWeightStride, weightStride, keysOffset, learningRate, weightDecay,
             maxNormalize, batchId);
    }
}
#endif
