#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include "THCAtomics.cuh"

#define divup(a, b) ((a) + (b) - 1) / (b)
const int THREADS_PER_BLOCK = 256;
const int THREADS_X = 32;
const int THREADS_Y = THREADS_PER_BLOCK / THREADS_X;
const int REPEAT = 32;
const int64_t NNZ_PER_BLOCK_MAX = 1024;

/* sign MACRO */
#ifndef clamp
#define clamp(a, low, high) max(min((a), (high)), (low))
#endif

__device__ double atomicExch(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long res = atomicExch(address_as_ull, __double_as_longlong(val));
  return __longlong_as_double(res);
}

template<typename Ty, bool train>
__global__ static
void updateOutput(
    Ty *output,
    Ty *normalizedValues,
    const Ty *values,
    const int64_t *cumSumSizes,
    const int64_t *keys,
    const int64_t batchSize,
    const int64_t outDim,
    Ty *weight,
    const Ty *bias,
    const int64_t weightStride,
    const int64_t keysOffset,
    const int maxNormalize,
    const int nnzPerBlock)
{
    /*******************************************************
     * Adapted from the following file in arrayfire
     * https://github.com/arrayfire/arrayfire/blob/v3.4.1/src/backend/opencl/kernel/csrmm.cl
     *
     *******************************************************
     * Original copyright notice can be seen below:
     *
     * Copyright (c) 2016, ArrayFire
     * All rights reserved.
     *
     * This file is distributed under 3-clause BSD license.
     * The complete license agreement can be obtained at:
     * http://arrayfire.com/licenses/BSD-3-Clause
     ********************************************************/

    const int64_t tidx = threadIdx.x;
    const int64_t tidy = threadIdx.y;
    const int64_t tid  = tidy * blockDim.x + tidx;
    const int64_t gidx = blockIdx.x * blockDim.x + tidx;


    Ty *nWeight = weight;
     // Offset the number of elements specified by  maxNormalize
    weight += gidx + maxNormalize;
    output += gidx;

    bool within_N = (gidx < outDim);

    __shared__ Ty s_values[THREADS_PER_BLOCK];
    __shared__ int64_t s_keys[THREADS_PER_BLOCK];

    const int64_t rowId = blockIdx.y;
    // if (rowId >= batchSize) return;

    // Load the nonzero column offsets for current row
    const int64_t batchStart = (rowId == 0 ? 0 : cumSumSizes[rowId - 1]) + blockIdx.z * nnzPerBlock;
    const int64_t batchEnd   = min(batchStart + nnzPerBlock, cumSumSizes[rowId]);
    const int64_t batchStride = blockDim.x * blockDim.y;

    Ty outVal = 0;
    // Since the number of nonzero elements might be greater than local memory available,
    // Load only part of the row into local memory, perform partial dot, repeat until done.
    for (int64_t id = batchStart; id < batchEnd; id += batchStride) {
        // Load the current chunk of the row into local memory
        int64_t lim = min(batchEnd - id, (int64_t)batchStride);

        int64_t key = tid < lim ? keys[id + tid] + keysOffset : -1;
        Ty val = tid < lim ? values[id + tid] : 0;
        int64_t nWeightOffset = key * weightStride;

        if (tid < lim && maxNormalize) {
            Ty *nWeightCurr = nWeight + nWeightOffset;
            if (train) {
                Ty absVal = fabs(val);
                Ty maxVal = nWeightCurr[0];
                if (absVal > maxVal) {
                    // Updating maxVal and invMaxVal. Go hogwild!
                    Ty invAbsVal = 1.0 / absVal;
                    atomicExch(nWeightCurr + 0, absVal);
                    atomicExch(nWeightCurr + 1, invAbsVal);
                }
                val = clamp(val * nWeightCurr[1], -1.0, 1.0) + nWeightCurr[3];
                normalizedValues[id + tid] = val;
                nWeightCurr[2] = 1;
            } else {
                val = clamp(val * nWeightCurr[1], -1.0, 1.0) + nWeightCurr[3];
            }
        }

        s_keys[tid] = key;
        s_values[tid] = val;
        __syncthreads();

        // Perform a single "dot" operation for each thread
        for (int64_t idy = tidy; within_N && idy < lim; idy += blockDim.y) {
            outVal += s_values[idy] * weight[weightStride * s_keys[idy]];
        }
        __syncthreads();
    }

    // s_values is no longer used at this point. Reuse it for reducing outVal.
    // A reduction along the y dimension now gives a single output value along x.
    s_values[tid] = outVal;
    for (int64_t y = blockDim.y / 2; y >= 1; y /= 2) {
        __syncthreads();
        if (tidy < y) s_values[tid] = s_values[tid] + s_values[tid + y * blockDim.x];
    }

    if (within_N && tidy == 0) {
        Ty val = s_values[tid] + (blockIdx.z == 0 ? bias[gidx] : 0);
        if (gridDim.z == 1) {
            output[rowId * outDim] = val;
        } else {
            atomicAdd(output + rowId * outDim, val);
        }
    }
}

// This kernel takes in the following inputs:
// values of size [keysSize x 1] and gradOutput of size [batchSize x outDim],
// to generate gradWeight of size [keysSize x outDim]
// nth block along y dimension computes on the non zero elements from the nth batch.
template<typename Ty>
__global__ static
void accGradWeight(
    Ty *gradWeight,
    const Ty *gradOutput,
    const Ty *values,
    const int64_t  *cumSumSizes,
    const int64_t  outDim,
    const int64_t  gradWeightStride,
    const Ty scale,
    const Ty weightDecay,
    const int maxNormalize)
{
    const int64_t bidy = blockIdx.y;
    const int64_t tidx = threadIdx.x;
    const int64_t tidy = threadIdx.y;
    const int64_t tid  = tidy * blockDim.x + tidx;
    const int64_t ntid = blockDim.x * blockDim.y;
    const int64_t gidx = blockIdx.x * blockDim.x + tidx;

    // All the y threads in the block will use the same gradOutput value
    gradOutput += bidy * outDim;
    Ty gradOutVal = scale * (gidx < outDim ? gradOutput[gidx] : 0);

    // Calculate the amount of work for the current block / batch.
    const int64_t batchStart = bidy == 0 ? 0 : cumSumSizes[bidy - 1];
    const int64_t batchEnd   = cumSumSizes[bidy];
    const int64_t batchLimit = batchEnd - batchStart;

    // Number of iterations required to finish the work for the current batch.
    const int64_t iters    = divup(batchLimit, ntid);

    // Offset the values to the current batch.
    values += batchStart;

    // When maxNormalize is enabled, gradWeight will be twice the size.
    // The first half will contain the gradients required for maxNormalization.
    // The second half will contain the gradients required for updating weights.
    // if maxNormalize is false, both will evaluate to the same pointer.
    Ty *gradWeight0 = gradWeight + batchStart * gradWeightStride + gidx;
    Ty *gradWeight1 = gradWeight0 + (maxNormalize ? outDim : 0);

    __shared__ Ty s_values[THREADS_PER_BLOCK];

    // Using iters to avoid divergence + synchtreads
    for (int64_t n = 0; n < iters; n++) {
        int64_t off = n * ntid;
        int64_t id = off + tid;
        int64_t lim = min(ntid, batchLimit - off);

        // Read the values required for the current iteration.
        s_values[tid] = id < batchLimit ? values[id] : 0;
        __syncthreads();

        if (gidx < outDim) {
            if (maxNormalize) {
                for (int64_t idy = tidy; idy < lim; idy += blockDim.y) {
                    // gradOutVal is already scaled
                    gradWeight0[(off + idy) * gradWeightStride] = gradOutVal;
                }
            }

            for (int64_t idy = tidy; idy < lim; idy += blockDim.y) {
                gradWeight1[(off + idy) * gradWeightStride] = s_values[idy] * gradOutVal;
            }
        }
        __syncthreads();
    }
}

// The gradBias is just a reduction of gradOutput along the batches.
// There is only one block along y dimension performing the reduction.
template<typename Ty, bool update>
__global__ static
void accGradBias(
    Ty *buffer,
    const Ty *gradOutput,
    const int64_t  outDim,
    const int64_t  batchSize,
    const Ty scale,
    const Ty weightDecay)
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int tid = tidy * blockDim.x + tidx;
    const int64_t idx = blockIdx.x * blockDim.x + tidx;


    Ty gradBiasVal = 0;
    gradOutput += idx;
    __shared__ Ty s_gradBiasVals[THREADS_PER_BLOCK];

    // Each thread along y calculates the partial sum.
    if (idx < outDim) {
        for (int64_t idy = tidy; idy < batchSize; idy += blockDim.y) {
            gradBiasVal += gradOutput[idy * outDim];
        }
    }
    s_gradBiasVals[tid] = gradBiasVal * scale;
    __syncthreads();

    // Perform reduction is performed along y.
    for (int y = blockDim.y / 2; y >= 1; y /= 2) {
        if (tidy < y) {
            s_gradBiasVals[tid] += s_gradBiasVals[tid + y * blockDim.x];
        }
        __syncthreads();
    }

    // Write the output only from the first lane.
    if (tidy == 0 && idx < outDim) {
        if (update) {
            // If performing inplace update, subtract from bias.
            Ty *bias = buffer;
            bias[idx] = (bias[idx] - s_gradBiasVals[tid]);
        } else {
            // If just accumulating gradients, write to gradBias.
            Ty *gradBias = buffer;
            gradBias[idx] = s_gradBiasVals[tid];
        }
    }
}

// Use gradWeight from accGradWeight to update the weight.
// This kernel is launched batchSize number of times.
// At each step in the iteration, the weights are updated in a sparse manner.
template<typename Ty>
__global__ static
void updateWeight(
    Ty *weight,
    const Ty *gradWeight,
    const int64_t *keys,
    const int64_t *cumSumSizes,
    const int64_t outDim,
    const int64_t gradWeightStride,
    const int64_t weightStride,
    const int64_t keysOffset,
    const Ty learningRate,
    const Ty weightDecay,
    const int maxNormalize,
    const int64_t batchId)
{
    int64_t gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t gidy = blockIdx.y * blockDim.y + threadIdx.y;

    // Find the limits of the work to be done
    const int64_t batchStart = batchId == 0 ? 0 : cumSumSizes[batchId - 1];
    const int64_t batchEnd = cumSumSizes[batchId];

    // When maxNormalize is turned on, the weight tensor will contain
    // an extra "maxNormalize" number of terms per output at the beginning.
    // When maxNormalize is false, both will evaluate to same pointer.
    // when maxNormalize is true,
    // - nWeight[2] will contain the individual scaling factor.
    // - nWeight[3] will contain the individual bias for the normalized input.
    Ty *nWeight = weight;
    weight += maxNormalize + gidx;

    // When maxNormalize is enabled, gradWeight will be twice the size.
    // The first half will contain the gradients required for maxNormalization.
    // The second half will contain the gradients required for updating weights.
    // if maxNormalize is false, both will evaluate to the same pointer.
    const Ty *gradWeight0 = gradWeight + gidx;
    const Ty *gradWeight1 = gradWeight0 + (maxNormalize ? outDim : 0);

    if (gidx >= outDim) return;
    for (int64_t id = batchStart + gidy; id < batchEnd; id += blockDim.y * gridDim.y) {
        Ty lr = learningRate;
        Ty wd = weightDecay;
        int64_t weightOffset = (keys[id] + keysOffset) * weightStride;
        Ty weightVal = weight[weightOffset];

        if (maxNormalize) {
            Ty scale = nWeight[weightOffset + 2];
            lr *= scale;
            wd *= scale;
            // nWeight[3] needs to be updated in the following manner for a given input.
            // nWeight[3] = nWeight[3] - sum(gradWeight0[gidx] * weight[gidx]);
            // Since problem is parallelized along gidx, use atomicAdd for the update.
            Ty gradNormBias = lr * weightVal * gradWeight0[id * gradWeightStride];
            atomicAdd(nWeight + weightOffset + 3, -gradNormBias);
        }

        // Perform the regular update
        Ty gradWeightVal = lr * gradWeight1[id * gradWeightStride];
        if (weightDecay == 0) {
            weight[weightOffset] = weightVal - gradWeightVal;
        } else {
            weight[weightOffset] = weightVal * (1 - wd) - gradWeightVal;
        }
    }
}

// This kernel is launched batchSize number of times.
// At each step in the iteration, the weights are updated in place in a sparse manner.
template<typename Ty>
__global__ static
void accUpdateWeight(
    Ty *weight,
    const int64_t weightStride,
    const Ty *gradOutput,
    const int64_t outDim,
    const Ty *values,
    const int64_t *cumSumSizes,
    const int64_t *keys,
    const int64_t keysOffset,
    const Ty scale,
    const Ty weightDecay,
    const int maxNormalize,
    const int64_t batchId)
{
    // Parallel along outDim.
    int64_t gidx = blockIdx.x * blockDim.x + threadIdx.x;
    // Parallel along the sparse input size for current batch.
    int64_t gidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gidx >= outDim) return;

    // Find the limits of the work to be done.
    const int64_t batchStart = batchId == 0 ? 0 : cumSumSizes[batchId - 1];
    const int64_t batchEnd = cumSumSizes[batchId];

    gradOutput += batchId * outDim;
    Ty gradOutVal = scale * (gidx < outDim ? gradOutput[gidx] : 0);

    // When maxNormalize is turned on, the weight tensor will contain
    // an extra "maxNormalize" number of terms per output at the beginning.
    // When maxNormalize is false, both will evaluate to same pointer.
    // when maxNormalize is true,
    // - nWeight[2] will contain the individual scaling factor.
    // - nWeight[3] will contain the individual bias for the normalized input.
    Ty *nWeight = weight;
    weight += maxNormalize + gidx;

    for (int64_t id = batchStart + gidy; id < batchEnd; id += blockDim.y * gridDim.y) {
        Ty wd = weightDecay;
        int64_t weightOffset = (keys[id] + keysOffset) * weightStride;
        Ty gradWeightVal = gradOutVal * values[id];
        Ty weightVal = weight[weightOffset];

        if (maxNormalize) {
            Ty nScale = nWeight[weightOffset + 2];
            gradWeightVal *= nScale;
            wd *= nScale;
            // nWeight[3] needs to be updated in the following manner for a given input.
            // nWeight[3] = nWeight[3] - sum(gradOut[gidx] * weight[gidx]);
            // Since problem is parallelized along gidx, use atomicAdd for the update.
            Ty gradNormBias = nScale * weightVal * gradOutVal;
            atomicAdd(nWeight + weightOffset + 3, -gradNormBias);
        }

        // Perform the regular update
        if (weightDecay == 0) {
            weight[weightOffset] = weightVal - gradWeightVal;
        } else {
            weight[weightOffset] = weightVal * (1 - wd) - gradWeightVal;
        }
    }
}


#ifdef CUDA_HALF_TENSOR
void THNN_CudaHalfIndexLinear_updateOutput(
                  THCState *state,
                  THCudaLongTensor *keys,
                  int64_t keysOffset,
                  THCudaHalfTensor *values,
                  THCudaLongTensor *sizes,
                  THCudaLongTensor *cumSumSizes,
                  THCudaHalfTensor *output,
                  THCudaHalfTensor *weight,
                  THCudaHalfTensor *bias,
                  THCudaHalfTensor *normalizedValues,
                  int   train) {
    THError("THCudaHalfTensor not supported with IndexLinear");
}

void THNN_CudaHalfIndexLinear_accGradParameters(
                  THCState *state,
                  THCudaLongTensor *keys,
                  int64_t keysOffset,
                  THCudaHalfTensor *values,
                  THCudaLongTensor *sizes,
                  THCudaLongTensor *cumSumSizes,
                  THCudaHalfTensor *gradOutput,
                  THCudaHalfTensor *gradWeight,
                  THCudaHalfTensor *gradBias,
                  THCudaHalfTensor *weight,
                  THCudaHalfTensor *bias,
                  THCudaHalfTensor* valuesBuffer,
                  float weightDecay,
                  float scale) {
    THError("THCudaHalfTensor not supported with IndexLinear");
}

void THNN_CudaHalfIndexLinear_accUpdateGradParameters(
                  THCState *state,
                  THCudaLongTensor *keys,
                  int64_t keysOffset,
                  THCudaHalfTensor *values,
                  THCudaLongTensor *sizes,
                  THCudaLongTensor *cumSumSizes,
                  THCudaHalfTensor *gradOutput,
                  THCudaHalfTensor *weight,
                  THCudaHalfTensor *bias,
                  float weightDecay,
                  float scale) {
    THError("THCudaHalfTensor not supported with IndexLinear");
}

void THNN_CudaHalfIndexLinear_updateParameters(
                  THCState *state,
                  THCudaHalfTensor *gradWeight,
                  THCudaHalfTensor *gradBias,
                  THCudaHalfTensor *weight,
                  THCudaHalfTensor *bias,
                  THCudaLongTensor *runningKeys,
                  THCudaLongTensor *cumSumSizes,
                  int64_t keysOffset,
                  float weightDecay,
                  float learningRate) {
    THError("THCudaHalfTensor not supported with IndexLinear");
}
#endif

#include "generic/IndexLinear.cu"
#include "THCGenerateFloatType.h"
#include "generic/IndexLinear.cu"
#include "THCGenerateDoubleType.h"
