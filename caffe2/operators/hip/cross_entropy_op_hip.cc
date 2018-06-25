/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <cub/block/block_reduce.cuh>
#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/cross_entropy_op.h"
#include "caffe2/operators/operator_fallback_hip.h"

namespace caffe2 {

namespace {
__global__ void LabelCrossEntropyKernel(const int N,
                                        const int D,
                                        const float* Xdata,
                                        const int* labeldata,
                                        const float log_threshold,
                                        float* Ydata)
{
    HIP_1D_KERNEL_LOOP(i, N)
    {
        HIP_KERNEL_ASSERT(labeldata[i] >= 0 && labeldata[i] < D);
        Ydata[i] = -logf(fmaxf(Xdata[i * D + labeldata[i]], log_threshold));
    }
}
__global__ void LabelCrossEntropyGradientKernel(const int N,
                                                const int D,
                                                const float* Xdata,
                                                const int* labeldata,
                                                const float* dYdata,
                                                const float log_threshold,
                                                float* dXdata)
{
    HIP_1D_KERNEL_LOOP(i, N)
    {
        int idx     = i * D + labeldata[i];
        dXdata[idx] = -dYdata[i] / fmaxf(Xdata[idx], log_threshold);
    }
}
} // namespace

template <>
bool LabelCrossEntropyOp<float, HIPContext>::RunOnDevice()
{
    auto& X     = Input(0);
    auto& label = Input(1);
    auto* Y     = Output(0);
    int N, D;
    if(X.ndim() > 1)
    {
        N = X.dim32(0);
        D = X.size_from_dim(1);
    }
    else
    {
        N = 1;
        D = X.dim32(0);
    }
    CAFFE_ENFORCE((label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
    CAFFE_ENFORCE_EQ(label.dim32(0), N);
    Y->Resize(vector<TIndex>(size_t(1), N));
    hipLaunchKernelGGL((LabelCrossEntropyKernel),
                       dim3(CAFFE_GET_BLOCKS(N)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(N),
                       static_cast<const int>(D),
                       X.data<float>(),
                       label.data<int>(),
                       kLOG_THRESHOLD(),
                       Y->mutable_data<float>());
    return true;
}

template <>
bool LabelCrossEntropyGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& X     = Input(0);
    auto& label = Input(1);
    auto& dY    = Input(2);
    auto* dX    = Output(0);
    int N, D;
    if(X.ndim() > 1)
    {
        N = X.dim32(0);
        D = X.size_from_dim(1);
    }
    else
    {
        N = 1;
        D = X.dim32(0);
    }
    CAFFE_ENFORCE((label.ndim() == 1) || (label.ndim() == 2 && label.dim32(1) == 1));
    CAFFE_ENFORCE_EQ(label.dim32(0), N);
    CAFFE_ENFORCE_EQ(dY.ndim(), 1);
    CAFFE_ENFORCE_EQ(dY.dim32(0), N);
    dX->ResizeLike(X);
    math::Set<float, HIPContext>(dX->size(), 0.f, dX->mutable_data<float>(), &context_);
    hipLaunchKernelGGL((LabelCrossEntropyGradientKernel),
                       dim3(CAFFE_GET_BLOCKS(N)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(N),
                       static_cast<const int>(D),
                       X.data<float>(),
                       label.data<int>(),
                       dY.data<float>(),
                       kLOG_THRESHOLD(),
                       dX->mutable_data<float>());
    return true;
}

namespace {
__global__ void MakeTwoClassKernel(const int N, const float* Xdata, float* Ydata)
{
    HIP_1D_KERNEL_LOOP(i, N)
    {
        Ydata[i * 2]     = 1.0 - Xdata[i];
        Ydata[i * 2 + 1] = Xdata[i];
    }
}
__global__ void MakeTwoClassGradientKernel(const int N, const float* dYdata, float* dXdata)
{
    HIP_1D_KERNEL_LOOP(i, N) { dXdata[i] = dYdata[i * 2 + 1] - dYdata[i * 2]; }
}
} // namespace

template <>
bool MakeTwoClassOp<float, HIPContext>::RunOnDevice()
{
    auto& X    = Input(0);
    auto* Y    = Output(0);
    auto shape = X.dims();
    shape.push_back(2);
    CAFFE_ENFORCE_LT(X.size(), std::numeric_limits<int>::max() / 2);
    Y->Resize(shape);
    int N = X.size();
    hipLaunchKernelGGL((MakeTwoClassKernel),
                       dim3(CAFFE_GET_BLOCKS(N)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(N),
                       X.data<float>(),
                       Y->mutable_data<float>());
    return true;
}

template <>
bool MakeTwoClassGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& dY   = Input(0);
    auto* dX   = Output(0);
    auto shape = dY.dims();
    CAFFE_ENFORCE_GE(shape.size(), 1);
    CAFFE_ENFORCE_EQ(shape.back(), 2);
    shape.pop_back();
    CAFFE_ENFORCE_LT(dY.size(), std::numeric_limits<int>::max());
    dX->Resize(shape);
    int N = dX->size();
    hipLaunchKernelGGL((MakeTwoClassGradientKernel),
                       dim3(CAFFE_GET_BLOCKS(N)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(N),
                       dY.data<float>(),
                       dX->mutable_data<float>());
    return true;
}

namespace {

__device__ float sigmoid_xent_forward(float lgt, float tgt)
{
    return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

__device__ float sigmoid_xent_backward(float lgt, float tgt) { return tgt - 1. / (1. + exp(-lgt)); }

__global__ void SigmoidCrossEntropyWithLogitsKernel(const int outer_size,
                                                    const int inner_size,
                                                    const float* logits_ptr,
                                                    const float* targets_ptr,
                                                    float* out_ptr)
{
    int i        = hipBlockIdx_x;
    int last_idx = (i + 1) * inner_size;
    float value  = 0;
    for(int in_idx = i * inner_size + hipThreadIdx_x; in_idx < last_idx; in_idx += hipBlockDim_x)
    {
        value += sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]);
    }

    typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = BlockReduce(temp_storage).Sum(value);
    if(hipThreadIdx_x == 0)
    {
        out_ptr[i] = -sum / static_cast<const float>(inner_size);
    }
}

__global__ void SigmoidCrossEntropyGradientWithLogitsKernel(const int outer_size,
                                                            const int inner_size,
                                                            const float* g_ptr,
                                                            const float* logits_ptr,
                                                            const float* targets_ptr,
                                                            float* out_ptr)
{
    HIP_1D_KERNEL_LOOP(in_idx, outer_size * inner_size)
    {
        int i           = in_idx / inner_size;
        float g_factor  = -g_ptr[i] / static_cast<const float>(inner_size);
        out_ptr[in_idx] = g_factor * sigmoid_xent_backward(logits_ptr[in_idx], targets_ptr[in_idx]);
    }
}
} // namespace

template <>
bool SigmoidCrossEntropyWithLogitsOp<float, HIPContext>::RunOnDevice()
{
    auto& logits  = Input(0);
    auto& targets = Input(1);
    CAFFE_ENFORCE(logits.dims() == targets.dims());
    const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
    const auto outer_size = logits.size() / inner_size;

    auto* out = Output(0);
    if(logits.ndim() == 0)
    {
        out->Resize(std::vector<TIndex>{});
    }
    else
    {
        std::vector<TIndex> dims(logits.dims().begin(), logits.dims().end() - 1);
        out->Resize(dims);
    }
    auto* out_ptr = out->mutable_data<float>();

    auto* logits_ptr  = logits.data<float>();
    auto* targets_ptr = targets.data<float>();

    hipLaunchKernelGGL((SigmoidCrossEntropyWithLogitsKernel),
                       dim3(outer_size),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(outer_size),
                       static_cast<const int>(inner_size),
                       logits_ptr,
                       targets_ptr,
                       out_ptr);
    return true;
}

template <>
bool SigmoidCrossEntropyWithLogitsGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& g       = Input(0);
    auto& logits  = Input(1);
    auto& targets = Input(2);
    CAFFE_ENFORCE(logits.dims() == targets.dims());
    const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
    const auto outer_size = logits.size() / inner_size;
    CAFFE_ENFORCE(g.size() == outer_size);

    auto* out = Output(0);
    out->ResizeLike(logits);
    auto* out_ptr = out->mutable_data<float>();

    auto* logits_ptr  = logits.data<float>();
    auto* targets_ptr = targets.data<float>();
    auto* g_ptr       = g.data<float>();

    hipLaunchKernelGGL((SigmoidCrossEntropyGradientWithLogitsKernel),
                       dim3(CAFFE_GET_BLOCKS(outer_size * inner_size)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(outer_size),
                       static_cast<const int>(inner_size),
                       g_ptr,
                       logits_ptr,
                       targets_ptr,
                       out_ptr);
    return true;
}

namespace {

__global__ void WeightedSigmoidCrossEntropyWithLogitsKernel(const int outer_size,
                                                            const int inner_size,
                                                            const float* logits_ptr,
                                                            const float* targets_ptr,
                                                            const float* weights_ptr,
                                                            float* out_ptr)
{
    int i        = hipBlockIdx_x;
    int last_idx = (i + 1) * inner_size;
    float value  = 0;
    for(int in_idx = i * inner_size + hipThreadIdx_x; in_idx < last_idx; in_idx += hipBlockDim_x)
    {
        value +=
            sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) * weights_ptr[in_idx];
    }

    typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = BlockReduce(temp_storage).Sum(value);
    if(hipThreadIdx_x == 0)
    {
        out_ptr[i] = -sum / inner_size;
    }
}

__global__ void WeightedSigmoidCrossEntropyGradientWithLogitsKernel(const int outer_size,
                                                                    const int inner_size,
                                                                    const float* g_ptr,
                                                                    const float* logits_ptr,
                                                                    const float* targets_ptr,
                                                                    const float* weights_ptr,
                                                                    float* out_ptr)
{
    HIP_1D_KERNEL_LOOP(in_idx, outer_size * inner_size)
    {
        int i           = in_idx / inner_size;
        float g_factor  = -g_ptr[i] / inner_size;
        out_ptr[in_idx] = g_factor *
                          sigmoid_xent_backward(logits_ptr[in_idx], targets_ptr[in_idx]) *
                          weights_ptr[in_idx];
    }
}
} // namespace

template <>
bool WeightedSigmoidCrossEntropyWithLogitsOp<float, HIPContext>::RunOnDevice()
{
    auto& logits  = Input(0);
    auto& targets = Input(1);
    auto& weights = Input(2);
    CAFFE_ENFORCE(logits.dims() == targets.dims());
    CAFFE_ENFORCE(weights.dims() == targets.dims());
    const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
    const auto outer_size = logits.size() / inner_size;

    auto* out = Output(0);
    if(logits.ndim() == 0)
    {
        out->Resize(std::vector<TIndex>{});
    }
    else
    {
        std::vector<TIndex> dims(logits.dims().begin(), logits.dims().end() - 1);
        out->Resize(dims);
    }
    auto* out_ptr = out->mutable_data<float>();

    auto* logits_ptr  = logits.data<float>();
    auto* targets_ptr = targets.data<float>();
    auto* weights_ptr = weights.data<float>();

    hipLaunchKernelGGL((WeightedSigmoidCrossEntropyWithLogitsKernel),
                       dim3(outer_size),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(outer_size),
                       static_cast<const int>(inner_size),
                       logits_ptr,
                       targets_ptr,
                       weights_ptr,
                       out_ptr);
    return true;
}

template <>
bool WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, HIPContext>::RunOnDevice()
{
    auto& g       = Input(0);
    auto& logits  = Input(1);
    auto& targets = Input(2);
    auto& weights = Input(3);
    CAFFE_ENFORCE(logits.dims() == targets.dims());
    CAFFE_ENFORCE(weights.dims() == targets.dims());
    const auto inner_size = logits.ndim() > 0 ? logits.dims().back() : 1;
    const auto outer_size = logits.size() / inner_size;
    CAFFE_ENFORCE(g.size() == outer_size);

    auto* out = Output(0);
    out->ResizeLike(logits);
    auto* out_ptr = out->mutable_data<float>();

    auto* logits_ptr  = logits.data<float>();
    auto* targets_ptr = targets.data<float>();
    auto* weights_ptr = weights.data<float>();
    auto* g_ptr       = g.data<float>();

    hipLaunchKernelGGL((WeightedSigmoidCrossEntropyGradientWithLogitsKernel),
                       dim3(CAFFE_GET_BLOCKS(outer_size * inner_size)),
                       dim3(CAFFE_HIP_NUM_THREADS),
                       0,
                       context_.hip_stream(),
                       static_cast<const int>(outer_size),
                       static_cast<const int>(inner_size),
                       g_ptr,
                       logits_ptr,
                       targets_ptr,
                       weights_ptr,
                       out_ptr);
    return true;
}

REGISTER_HIP_OPERATOR(LabelCrossEntropy, LabelCrossEntropyOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(LabelCrossEntropyGradient, LabelCrossEntropyGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(SigmoidCrossEntropyWithLogits,
                      SigmoidCrossEntropyWithLogitsOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(SigmoidCrossEntropyWithLogitsGradient,
                      SigmoidCrossEntropyWithLogitsGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(WeightedSigmoidCrossEntropyWithLogits,
                      WeightedSigmoidCrossEntropyWithLogitsOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(WeightedSigmoidCrossEntropyWithLogitsGradient,
                      WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(MakeTwoClass, MakeTwoClassOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(MakeTwoClassGradient, MakeTwoClassGradientOp<float, HIPContext>);

// TODO(surya) Add full GPU/CUDA support for the CrossEntropyOp
REGISTER_HIP_OPERATOR(CrossEntropy, GPUFallbackOp<CrossEntropyOp<float, CPUContext>>);
REGISTER_HIP_OPERATOR(CrossEntropyGradient,
                      GPUFallbackOp<CrossEntropyGradientOp<float, CPUContext>>);

} // namespace caffe2
