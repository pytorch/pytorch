/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <vector>
#include <cmath>
#include "../include/layer_kernels.cuh"

using namespace std;

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxEnergies:     (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * top5Probs:       (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 * 
 */
__global__ void kMultiSoftmaxCost(float* probs, float* labels, float* maxProbs,
                                  float* labelLogProbs, float* correctProbs, float* top5Probs,
                                  const int numCases, const int numOut, const int setSize) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        int numBiggerProbs = 0, numEqualsProbs = 0;
        for (int i = 0; i < numOut; ++i) {
            numBiggerProbs += probs[i * numCases + tx] > labelp;
            numEqualsProbs += probs[i * numCases + tx] == labelp;
        }

        const int slotsLeft = setSize - numBiggerProbs;
        
        top5Probs[tx] = slotsLeft <= 0.0f ? 0.0f : (numEqualsProbs <= slotsLeft ? 1.0f : float(slotsLeft) / numEqualsProbs);
        correctProbs[tx] = labelp != maxp ? 0.0f : 1.0f / float(numEqualsProbs);
    }
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * top5Probs:       (1, numCases)   (*out)
 * 
 * target:          (1, numCases) == log(y_l[labels,:]
 */
void computeMultiSoftmaxCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& maxProbs, NVMatrix& labelLogProbs_out,
                       NVMatrix& correctProbs_out, NVMatrix& top5Probs_out, int setSize) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
//    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    top5Probs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaStream_t stream = NVMatrix::getDefaultStream();

    cudaFuncSetCacheConfig(kMultiSoftmaxCost, cudaFuncCachePreferL1);
    kMultiSoftmaxCost<<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                    labelLogProbs_out.getDevData(), correctProbs_out.getDevData(), top5Probs_out.getDevData(),
                                    numCases, numOut, setSize);

    getLastCudaError("kMultiSoftmaxCost: Kernel execution failed");
//    cudaThreadSynchronize();
}

/*
 * E = sum(p_l * log(y_l))
 * probs:           (numOut, numCases)
 * labels:          (numOut, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kCrossEntCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        probs += tx;
        labels += tx;
        maxProbs += tx;
        labelLogProbs += tx;
        correctProbs += tx;
        
        const float maxp = maxProbs[0];

        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        float crossEnt = 0.0f;
        int numMax = 0;
        bool correctLabel = false;
        for (int i = 0; i < numOut; i++) {
            const float label_prob = labels[i * numCases];
            const float model_prob = probs[i * numCases];
            numMax += model_prob == maxp;
            crossEnt += label_prob * safelog(model_prob);
            correctLabel |= model_prob == maxp && label_prob > 0.0f;
        }
        labelLogProbs[0] = crossEnt;
        if (!correctLabel) {
            correctProbs[0] = 0.0f;
        } else {
            correctProbs[0] = 1.0f / float(numMax);
        }
    }
}

/*
 * E = sum(p_l * log(y_l))
 * y_l:     (numOut, numCases)
 * labels:  (numOut, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kCrossEntGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const float label_prob = labels[tidx];
        const float model_prob = y_l[tidx];
        const float v = gradCoeff * __fdividef(label_prob, model_prob);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * E = sum(p_l * log(y_l))
 * y_l:     (numOut, numCases)
 * labels:  (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kCrossEntSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const float model_prob = y_l[tidx];
        const float label_prob = labels[tidx];
        float v = gradCoeff * (label_prob - model_prob);
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
__global__ void kLogregCost(float* probs, float* labels, float* maxProbs, float* labelLogProbs, float* correctProbs,
                            const int numCases, const int numOut) {
    const int tx = blockIdx.x * LOGREG_ERR_THREADS_X + threadIdx.x;

    if (tx < numCases) {
        const int label = int(labels[tx]);
        const float maxp = maxProbs[tx];
        const float labelp = probs[label * numCases + tx];
        
        labelLogProbs[tx] = __logf(labelp);
        
        /*
         * Compute the probability of guessing the correct case if you take the most-probable label.
         * 
         * This is done like this:
         * 
         * - If the most probable label is not equal to the true label, then the probability is zero.
         * - Otherwise, the probability is 1 / (number of labels whose probability is equal to the maximum).
         * 
         * This is certainly overkill -- in practice, it's just about impossible for two labels to get assigned
         * maximum probability. But it's a safety measure to prevent over-estimating your accuracy.
         * Though it could never happen in reality. Well it could. But it wouldn't. Cool?
         */
        if (labelp != maxp) {
            correctProbs[tx] = 0;
        } else {
            int numMax = 0;
            for (int i = 0; i < numOut; i++) {
                numMax += probs[i * numCases + tx] == maxp;
            }
            correctProbs[tx] = 1.0f / float(numMax);
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dy_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregCostGrad(float* y_l, float* labels, float* dE_dy_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * (label == ty);
        v = __fdividef(v, y_l[tidx]);
        if (add) {
            dE_dy_l[tidx] += v;
        } else {
            dE_dy_l[tidx] = v;
        }
    }
}

/*
 * E = -log(y_t)
 * y_l:     (numOut, numCases)
 * labels:  (1, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kLogregSoftmaxGrad(float* y_l, float* labels, float* dE_dx_l, const int numCases,
                                 const int numOut, const float gradCoeff) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        const int label = int(labels[tx]);
        float v = gradCoeff * ((label == ty) - y_l[tidx]);
        if (add) {
            dE_dx_l[tidx] += v;
        } else {
            dE_dx_l[tidx] = v;
        }
    }
}

/*
 * dE_dy_l: (numOut, numCases)
 * y_l:     (numOut, numCases)
 * 
 * dE_dx_l: (numOut, numCases)
 */
template <bool add>
__global__ void kSoftmaxGrad(float* dE_dy_l, float* y_l, float* dE_dx_l, const int numCases, const int numOut, const float scaleTarget, const float scaleGrad) {
    const int tx = blockIdx.x * LOGREG_GRAD_THREADS_X + threadIdx.x;
    const int ty = blockIdx.y * LOGREG_GRAD_THREADS_Y + threadIdx.y;
    const int tidx = ty * numCases + tx;
    
    if (ty < numOut && tx < numCases) {
        float v = 0;
        for (int j = 0; j < numOut; j++) {
            v += dE_dy_l[j * numCases + tx] * ((j == ty) - y_l[j * numCases + tx]);
        }
        v *= y_l[tidx];
        
        if (add) {
            dE_dx_l[tidx] = scaleTarget * dE_dx_l[tidx] + scaleGrad * v;
        } else {
            dE_dx_l[tidx] = scaleGrad * v;
        }
    }
}

template <int B_X, bool add>
__global__ void kEltwiseMaxGrad(float* actGrad, float* input, float* output, float* target,
                                const int numElements) {
    for (int i = B_X * blockIdx.x + threadIdx.x; i < numElements; i += B_X * gridDim.x) {
        if (add) {
            target[i] += actGrad[i] * (output[i] == input[i]);
        } else {
            target[i] = actGrad[i] * (output[i] == input[i]);
        }
    }
}

void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add) {
    assert(actGrad.isContiguous());
    assert(output.isContiguous());
    assert(input.isContiguous());
    assert(actGrad.isSameDims(input));
    assert(actGrad.isSameDims(output));
    
    dim3 blocks(DIVUP(actGrad.getNumElements(), 128));
    dim3 threads(128);
    cudaStream_t stream = NVMatrix::getDefaultStream();
    if (add) {
        assert(actGrad.isSameDims(target));
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, true>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, true><<<blocks, threads, 0, stream>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    } else {
        target.resize(actGrad);
        cudaFuncSetCacheConfig(kEltwiseMaxGrad<128, false>, cudaFuncCachePreferL1);
        kEltwiseMaxGrad<128, false><<<blocks, threads, 0, stream>>>(actGrad.getDevData(), input.getDevData(), output.getDevData(), target.getDevData(), actGrad.getNumElements());
    }
    
    getLastCudaError("computeEltwiseMaxGrad: Kernel execution failed");
}

/*
 * E = sum_i{-p_i*log(y_i)}
 * probs:           (numOut, numCases)
 * labels:          (numOut, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases)
 */
void computeCrossEntCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.isSameDims(probs));
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());
    
    NVMatrix& maxProbs = probs.max(0);
    
    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaStream_t stream = NVMatrix::getDefaultStream();
    cudaFuncSetCacheConfig(kCrossEntCost, cudaFuncCachePreferL1);
    kCrossEntCost<<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    getLastCudaError("kCrossEntCost: Kernel execution failed");

    delete &maxProbs;
}

void computeCrossEntGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.isSameDims(probs));
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    cudaStream_t stream = NVMatrix::getDefaultStream();
    if (!add) {
        target.resize(probs);
        kCrossEntGrad<false><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kCrossEntGrad<true><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("kCrossEntGrad: Kernel execution failed");
}

void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, float scaleTarget, float scaleGrad) {
    int numCases = acts.getLeadingDim();
    int numOut = acts.getFollowingDim();

    assert(acts.isSameDims(actsGrad));
    assert(acts.isContiguous());
    assert(actsGrad.isContiguous());
    assert(target.isContiguous());
    assert(acts.isTrans());
    assert(actsGrad.isTrans());

    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    cudaStream_t stream = NVMatrix::getDefaultStream();

    if (scaleTarget == 0) {
        target.resize(acts);
        kSoftmaxGrad<false><<<blocks, threads, 0, stream>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut, scaleTarget, scaleGrad);
    } else {
        kSoftmaxGrad<true><<<blocks, threads, 0, stream>>>(actsGrad.getDevData(), acts.getDevData(), target.getDevData(), numCases, numOut, scaleTarget, scaleGrad);
    }
    getLastCudaError("computeSoftmaxGrad: Kernel execution failed");
}

void computeCrossEntSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getLeadingDim() == probs.getLeadingDim() && labels.getFollowingDim() == probs.getFollowingDim());
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    assert(!labels.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    cudaStream_t stream = NVMatrix::getDefaultStream();
    if (!add) {
        target.resize(probs);
        cudaFuncSetCacheConfig(kCrossEntSoftmaxGrad<false>, cudaFuncCachePreferL1);
        kCrossEntSoftmaxGrad<false><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                        numCases, numOut, coeff);
    } else {
        cudaFuncSetCacheConfig(kCrossEntSoftmaxGrad<true>, cudaFuncCachePreferL1);
        kCrossEntSoftmaxGrad<true><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                        numCases, numOut, coeff);
    }
    getLastCudaError("kCrossEntSoftmaxGrad: Kernel execution failed");
}

/*
 * E = -log(y_t)
 * probs:           (numOut, numCases)
 * labels:          (1, numCases)
 * maxProbs:        (1, numCases)
 * labelLogProbs:   (1, numCases)   (*out)
 * correctProbs:    (1, numCases)   (*out)
 * 
 * target:          (1, numCases) == log(y_l[labels,:]
 */
void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& maxProbs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out) {
    int numCases = probs.getNumCols(); 
    int numOut = probs.getNumRows(); 

    assert(labels.getNumElements() == numCases);
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    assert(labels.isContiguous());
    assert(probs.isContiguous());

    labelLogProbs_out.resize(1, numCases);
    correctProbs_out.resize(1, numCases);
    dim3 threads(LOGREG_ERR_THREADS_X, 1);
    dim3 blocks(DIVUP(numCases, LOGREG_ERR_THREADS_X), 1);
    cudaStream_t stream = NVMatrix::getDefaultStream();
    cudaFuncSetCacheConfig(kLogregCost, cudaFuncCachePreferL1);
    kLogregCost<<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), maxProbs.getDevData(),
                                     labelLogProbs_out.getDevData(), correctProbs_out.getDevData(),
                                     numCases, numOut);
    getLastCudaError("computeLogregCost: Kernel execution failed");
}

void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(!labels.isTrans());
    assert(!probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    cudaStream_t stream = NVMatrix::getDefaultStream();
    if (!add) {
        target.resize(probs);
        kLogregCostGrad<false><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregCostGrad<true><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("computeLogregGrad: Kernel execution failed");
}

void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff) {
    int numCases = probs.getLeadingDim(); 
    int numOut = probs.getFollowingDim(); 
    assert(labels.getNumElements() == numCases);
    assert(probs.isContiguous());
    assert(target.isContiguous());
    assert(labels.isContiguous());
    assert(probs.isTrans());
    
    dim3 threads(LOGREG_GRAD_THREADS_X, LOGREG_GRAD_THREADS_Y);
    dim3 blocks(DIVUP(numCases, LOGREG_GRAD_THREADS_X), DIVUP(numOut, LOGREG_GRAD_THREADS_Y));
    cudaStream_t stream = NVMatrix::getDefaultStream();
    if (!add) {
        target.resize(probs);
        kLogregSoftmaxGrad<false><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    } else {
        kLogregSoftmaxGrad<true><<<blocks, threads, 0, stream>>>(probs.getDevData(), labels.getDevData(), target.getDevData(),
                                                     numCases, numOut, coeff);
    }

    getLastCudaError("computeLogregSoftmaxGrad: Kernel execution failed");
}
