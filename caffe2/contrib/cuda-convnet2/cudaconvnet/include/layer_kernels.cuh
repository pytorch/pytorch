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

#ifndef LAYER_KERNELS_CUH
#define	LAYER_KERNELS_CUH

#include <vector>
#include <helper_cuda.h>
#include "../../nvmatrix/include/nvmatrix.cuh"

#define LOGREG_GRAD_THREADS_X      32
#define LOGREG_GRAD_THREADS_Y      4

#define LOGREG_ERR_THREADS_X        128
#define LOGREG_ERR_THREADS_Y        1

__device__ inline float safelog(const float x) {
    return x > 0.0f ? __logf(x) : -50.0f;
}

// The input matrix here is the squared norm.
// This replaces the squared norm with:
// 1 if it is below the threshold given by norm2
// norm/sqrt(a) otherwise -- i.e. the desired norm (not squared)
class MaxWeightConstraintOperator {
private:
    float _norm, _norm2;
public:
    MaxWeightConstraintOperator(float norm) : _norm(norm), _norm2(norm*norm) {
    }
    __device__ inline float operator()(const float a) const {
        return a > _norm2 ? __fdividef(_norm, sqrtf(a)) : 1.0f;
    }
};

class HardWeightConstraintOperator {
private:
    float _norm, _norm2;
public:
    HardWeightConstraintOperator(float norm) : _norm(norm), _norm2(norm*norm) {
    }
    __device__ inline float operator()(const float a) const {
        return __fdividef(_norm, sqrtf(a));
    }
};

class WeightContrastNormOperator {
private:
    float _min, _max, _scale;
public:
    WeightContrastNormOperator(float min, float max, float scale) : _min(min), _max(max), _scale(scale) {
    }
    __device__ inline float operator()(float a) const {
        a = sqrtf(a) * _scale;
        return a < _min ? __fdividef(_min, a) : a > _max ? __fdividef(_max, a) : 1.0f;
    }
};

void computeCrossEntCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out);
void computeCrossEntGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeSoftmaxGrad(NVMatrix& acts, NVMatrix& actsGrad, NVMatrix& target, float scaleTarget, float scaleGrad);

void computeLogregCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& maxProbs, NVMatrix& labelLogProbs_out, NVMatrix& correctProbs_out);
void computeLogregGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);


// Numerical stability optimization: this routine combines computeLogregGrad with computeSoftmaxGrad
// to avoi dividing and then multiplying by quantities that may be near zero.
void computeCrossEntSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeLogregSoftmaxGrad(NVMatrix& labels, NVMatrix& probs, NVMatrix& target, bool add, float coeff);
void computeEltwiseMaxGrad(NVMatrix& actGrad, NVMatrix& input, NVMatrix& output, NVMatrix& target, bool add);
void computeMultiSoftmaxCost(NVMatrix& labels, NVMatrix& probs, NVMatrix& maxProbs, NVMatrix& labelLogProbs_out,
                             NVMatrix& correctProbs_out, NVMatrix& top5Probs_out, int setSize);
#endif	/* LAYER_KERNELS_CUH */

