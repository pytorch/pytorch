#include <algorithm>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/adagrad_op.h"
#include "caffe2/utils/math.h"

#ifdef __HIP_PLATFORM_HCC__
#define SEGREDUCE_MINBLOCKS 8
#else
#define SEGREDUCE_MINBLOCKS 16
#endif

namespace caffe2 {

namespace {

void inclusive_scan_wrapper(
    const int* length_data,
    int len_length,
    Tensor* temp_buffer,
    Tensor* prefix_sum_out,
    CUDAContext* context_) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(
      NULL,
      temp_storage_bytes,
      length_data,
      prefix_sum_out->template mutable_data<int>(),
      len_length,
      context_->cuda_stream());
  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(int)) / sizeof(int);
  temp_buffer->Resize(buffer_size);
  void* d_temp_storage =
      static_cast<void*>(temp_buffer->template mutable_data<int>());
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      length_data,
      prefix_sum_out->template mutable_data<int>(),
      len_length,
      context_->cuda_stream());
}

template <typename SIndex, typename THalf, typename T, bool ExactBlock = false>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void sparse_adagrad_fused_length_sum_gradient_kernel(
    const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                    // (offsets for the
                                                    // segments)
    int N, // number of rows (hash size) of embedding table
    int post, // embedding dimension size
    int len_length, // number of segments
    const float epsilon,
    THalf* param,
    THalf* param_mom,
    const SIndex* indices,
    const T* __restrict__ grad,
    const float* lr) {
  const float LR = lr[0];
  // len_length blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  if (ExactBlock) {
    const size_t gradIdx = group * post + threadIdx.x; // index for grad
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      // line: the idx in the indices
      // i: index in the embedding dimension, which is equal to threadIdx.x
      const SIndex index =
          indices[line]; // the index-th row in the embedding table
      const size_t paramIdx = index * post + threadIdx.x; // index for param

      float mom_new = grad[gradIdx] * grad[gradIdx] + param_mom[paramIdx];
      param_mom[paramIdx] = mom_new;
      float param_new =
          LR * grad[gradIdx] / (sqrtf(mom_new) + epsilon) + param[paramIdx];
      param[paramIdx] = param_new;
    }
  } else {
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      // i: index in the embedding dimension
      const size_t gradIdx = group * post + i; // index for grad
      for (int line = start; line < end; ++line) {
        // line: the idx in the indices
        const SIndex index =
            indices[line]; // the index row in the embedding table
        const size_t paramIdx = index * post + i; // index for param

        float mom_new = grad[gradIdx] * grad[gradIdx] + param_mom[paramIdx];
        param_mom[paramIdx] = mom_new;
        float param_new =
            LR * grad[gradIdx] / (sqrtf(mom_new) + epsilon) + param[paramIdx];
        param[paramIdx] = param_new;
      }
    }
  }
}

template <typename SIndex, typename THalf, typename T, int NumThreads>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void sparse_adagrad_fused_length_weighted_sum_gradient_kernel(
    const int* __restrict__ prefix_sum_length_data,
    int N, // number of rows (hash size) of embedding table
    int post, // embedding dimension size
    int len_length, // number of segments
    const float epsilon,
    THalf* param,
    THalf* param_mom,
    const SIndex* indices,
    const T* __restrict__ grad,
    const T* __restrict__ weights,
    T* __restrict__ weights_grad_out,
    const float* lr) {
  const float LR = lr[0];
  // len_length blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  // TODO: Tuning NumThreads for w_grad
  typedef cub::BlockReduce<float, NumThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // TODO(jianyuhuang): parallelize this outer loop
  for (int line = start; line < end; ++line) {
    T w_grad = 0;
    // line: the idx in the indices
    const SIndex index =
        indices[line]; // the index-th row in the embedding table

    // SparseAdagradFusedWithSparseLengthsWeightedSumGradientOp also fuses
    // LengthsRangeFill + Gather operator. In the normal SLWS operator weight is
    // accessed via weights[line] but in most cases the weights are
    // generated by LengthsRangeFill and Gather operator.
    // For example, if lengths is [2, 3, 1] LengthsRangeFill will generate [0,
    // 1; 0, 1, 2; 0] and they are used as indices of Gather.
    // So if we fuse all of these, weights[line] just becomes
    // weights[line - start].
    auto in_weight_temp = weights[line - start];

    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      // i: index in the embedding dimension
      const size_t gradIdx = group * post + i; // index for in_grad
      const size_t paramIdx = index * post + i; // index for param

      // TODO: trying to reduce the variable number (common subexpression
      // elimination).
      auto in_grad_temp = grad[gradIdx];
      auto out_grad_temp = in_weight_temp * in_grad_temp;
      w_grad += in_grad_temp * param[paramIdx];

      // TODO: split it into two kernels to make it more similar to exact fusion
      // kernel (not Approx on CPUs).
      float mom_new = out_grad_temp * out_grad_temp + param_mom[paramIdx];
      param_mom[paramIdx] = mom_new;
      float param_new =
          LR * out_grad_temp / (sqrtf(mom_new) + epsilon) + param[paramIdx];
      param[paramIdx] = param_new;
    }

    w_grad = BlockReduce(temp_storage).Reduce(w_grad, cub::Sum());

    // Alternative:
    // int valid = min(post, blockDim.x);
    // float w_reduce_result = BlockReduce(temp_storage).Sum(w_grad, valid);

    if (threadIdx.x == 0) {
      weights_grad_out[line] = w_grad;
    }
    __syncthreads();
  }
}

template <typename SIndex, typename THalf, typename T>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void rowwise_sparse_adagrad_fused_length_sum_gradient_kernel(
    const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                    // (offsets for the
                                                    // segments)
    int N, // number of rows (hash size) of embedding table
    int post, // embedding dimension size
    int len_length, // number of segments
    const float epsilon,
    THalf* param,
    THalf* param_mom,
    const SIndex* indices,
    const T* __restrict__ grad,
    const float* lr) {
  const float LR = lr[0];
  // len_length blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  // TODO: Tuning NumThreads for sum_squares
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;
  // int valid = min(post, CAFFE_CUDA_NUM_THREADS);
  int valid = min(post, blockDim.x);

  for (int line = start; line < end; ++line) {
    // line: the idx in the indices
    const SIndex index = indices[line]; // the index row in the embedding table
    float sum_squares = 0.0;
    __shared__ float row_sum_squares_avg;

    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      // i: index in the embedding dimension
      const float x_ij = grad[group * post + i];
      sum_squares += x_ij * x_ij;
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum_squares, valid);
    // float reduce_result =
    //     BlockReduce(temp_storage).Sum(sum_squares, blockDim.x);

    if (threadIdx.x == 0) {
      row_sum_squares_avg = reduce_result / static_cast<float>(post);
      param_mom[index] += row_sum_squares_avg;
    }
    __syncthreads();

    // update param
    float step = LR / (sqrtf(param_mom[index]) + epsilon);
    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      const size_t paramIdx = index * post + i; // index for param
      param[paramIdx] = param[paramIdx] + grad[group * post + i] * step;
    }
  }
}

template <typename SIndex, typename THalf, typename T, int NumThreads>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__
    void rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel(
        const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                        // (offsets for the
                                                        // segments)
        int N, // number of rows (hash size) of embedding table
        int post, // embedding dimension size
        int len_length, // number of segments
        const float epsilon,
        THalf* param,
        THalf* param_mom,
        const SIndex* indices,
        const T* __restrict__ grad,
        const T* __restrict__ weights,
        T* __restrict__ weights_grad_out,
        const float* lr) {
  const float LR = lr[0];
  // len_length blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  // TODO: Tuning NumThreads for w_grad
  typedef cub::BlockReduce<float, NumThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // for avg_square_weight. Can we reuse temp_storage
  __shared__ typename BlockReduce::TempStorage temp_storage2;
  // Why do we need to add typename here?

  // TODO(jianyuhuang): parallelize this outer loop
  for (int line = start; line < end; ++line) {
    T w_grad = 0;
    // i: index in the embedding dimension
    const SIndex index = indices[line];

    auto in_weight_temp = weights[line - start];

    float sum_squares = 0.0;
    __shared__ float row_sum_squares_avg;

    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      const float x_ij = grad[group * post + i];
      sum_squares += x_ij * x_ij;
    }
    // float reduce_result = BlockReduce(temp_storage).Sum(sum_squares, valid);
    float reduce_result =
        BlockReduce(temp_storage2).Sum(sum_squares, NumThreads);

    if (threadIdx.x == 0) {
      row_sum_squares_avg = reduce_result / static_cast<float>(post);
      param_mom[index] += row_sum_squares_avg * in_weight_temp * in_weight_temp;
    }
    __syncthreads();

    // update param
    float step = LR / (sqrtf(param_mom[index]) + epsilon);

    for (int i = threadIdx.x; i < post; i += blockDim.x) {
      const size_t gradIdx = group * post + i; // index for in_grad
      const size_t paramIdx = index * post + i; // index for param
      // TODO: trying to reduce the variable number (common subexpression
      // elimination).
      auto in_grad_temp = grad[gradIdx];
      auto out_grad_temp = in_weight_temp * in_grad_temp;
      w_grad += in_grad_temp * param[paramIdx];

      // TODO: split it into two kernels to make it more similar to exact fusion
      // kernel (not Approx on CPUs).
      param[paramIdx] = out_grad_temp * step + param[paramIdx];
    }
    w_grad = BlockReduce(temp_storage).Reduce(w_grad, cub::Sum());

    // int valid = min(post, blockDim.x);
    // float w_reduce_result = BlockReduce(temp_storage).Sum(w_grad, valid);

    if (threadIdx.x == 0) {
      weights_grad_out[line] = w_grad;
    }
    __syncthreads();
  }
}

} // namespace

template <typename T, typename TLengths, class Context>
class CUDASparseAdagradFusedWithSparseLengthsSumGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseAdagradFusedWithSparseLengthsSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {
    const T decay = this->template GetSingleArgument<T>("decay", 1.0f);
    CAFFE_ENFORCE_EQ(decay, 1.0, "Decay is not supported for SparseAdagradOp");
  }

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_1).size());
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }
    return DispatchHelper<TensorTypes2<float, at::Half>, IndexType>::call(
        this, Input(PARAM));
  }

  template <typename IndexType, typename THalf>
  bool DoRunWithType2() {
    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);
    auto& indicesInput = Input(INDICES);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_GT(Input(GRAD).dim(), 0);

    // Enforce:
    // input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel(),
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Input Moment size: ",
        Input(MOMENT_1).numel());

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const T* grad = Input(GRAD).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<THalf>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<THalf>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (post <= maxThreads) {
      int multiple = std::min(maxThreads / post, SEGREDUCE_MINBLOCKS);
      dim3 block(post, multiple);

      // calling cuda kernel with ExactBlock = true
      // T should be rename as TGRAD ?
      sparse_adagrad_fused_length_sum_gradient_kernel<IndexType, THalf, T, true>
          <<<len_length, block, 0, context_.cuda_stream()>>>(
              prefix_sum_length_data,
              N,
              post,
              len_length,
              epsilon_,
              paramOut,
              momentOut,
              indices,
              grad,
              lr);
    } else {
      // calling cuda kernel with ExactBlock = false
      sparse_adagrad_fused_length_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          false><<<len_length, maxThreads, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          lr);
    }
    return true;
  }

 private:
  // menber field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename T, typename TLengths, class Context>
class CUDASparseAdagradFusedWithSparseLengthsWeightedSumGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseAdagradFusedWithSparseLengthsWeightedSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {
    const T decay = this->template GetSingleArgument<T>("decay", 1.0f);
    CAFFE_ENFORCE_EQ(decay, 1.0, "Decay is not supported for SparseAdagradOp");
  }

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_1).size());
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto n = Input(INDICES).size();
    if (n == 0) {
      // Allocate output to an empty tensor
      Output(AUX_GRAD, n, at::dtype<T>());
      return true;
    }
    return DispatchHelper<TensorTypes2<float, at::Half>, IndexType>::call(
        this, Input(PARAM));
  }

  template <typename IndexType, typename THalf>
  bool DoRunWithType2() {
    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);
    auto& indicesInput = Input(INDICES);
    auto& weightsInput = Input(AUX_PARAM);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(1, weightsInput.dim(), "WEIGHTS must be a vector");
    CAFFE_ENFORCE_GT(Input(GRAD).dim(), 0);

    // Enforce:
    // input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).numel(),
        Input(MOMENT_1).numel(),
        "Input Param size: ",
        Input(PARAM).numel(),
        " Input Moment size: ",
        Input(MOMENT_1).numel());

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);
    auto* weightGradsOutput =
        Output(AUX_GRAD, indicesInput.sizes(), at::dtype<T>());

    T* out_weight_grads = weightGradsOutput->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const T* grad = Input(GRAD).template data<T>();
    const T* weights = weightsInput.template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<THalf>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<THalf>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (post > 128) {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          512><<<len_length, 512, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    } else if (post > 64) {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          128><<<len_length, 128, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    } else if (post > 32) {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          64><<<len_length, 64, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    } else {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          32><<<len_length, 32, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    }
    return true;
  }

 private:
  // menber field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

template <typename T, typename TLengths, class Context>
class CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {
    const T decay = this->template GetSingleArgument<T>("decay", 1.0f);
    CAFFE_ENFORCE_EQ(decay, 1.0, "Decay is not supported for SparseAdagradOp");
  }

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }
    return DispatchHelper<TensorTypes2<float, at::Half>, IndexType>::call(
        this, Input(PARAM));
  }

  template <typename IndexType, typename THalf>
  bool DoRunWithType2() {
    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);
    auto& indicesInput = Input(INDICES);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_GT(Input(GRAD).dim(), 0);

    // Enforce:
    // number of rows: input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).dim(0),
        Input(MOMENT_1).dim(0),
        "Input Param number of rows: ",
        Input(PARAM).dim(0),
        " Input Moment size: ",
        Input(MOMENT_1).dim(0));

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const T* grad = Input(GRAD).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<THalf>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<THalf>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    rowwise_sparse_adagrad_fused_length_sum_gradient_kernel<IndexType, THalf, T>
        <<<len_length, std::min(maxThreads, post), 0, context_.cuda_stream()>>>(
            prefix_sum_length_data,
            N,
            post,
            len_length,
            epsilon_,
            paramOut,
            momentOut,
            indices,
            grad,
            lr);

    return true;
  }

 private:
  // menber field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename T, typename TLengths, class Context>
class CUDARowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDARowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {
    const T decay = this->template GetSingleArgument<T>("decay", 1.0f);
    CAFFE_ENFORCE_EQ(decay, 1.0, "Decay is not supported for SparseAdagradOp");
  }

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto n = Input(INDICES).size();
    if (n == 0) {
      Output(AUX_GRAD, n, at::dtype<T>());
      return true;
    }
    return DispatchHelper<TensorTypes2<float, at::Half>, IndexType>::call(
        this, Input(PARAM));
  }

  template <typename IndexType, typename THalf>
  bool DoRunWithType2() {
    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);
    auto& indicesInput = Input(INDICES);
    auto& weightsInput = Input(AUX_PARAM);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(1, weightsInput.dim(), "WEIGHTS must be a vector");
    CAFFE_ENFORCE_GT(Input(GRAD).dim(), 0);

    // Enforce:
    // number of rows: input(embedding/momentum) == outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).dim(0),
        Input(MOMENT_1).dim(0),
        "Input Param number of rows: ",
        Input(PARAM).dim(0),
        " Input Moment size: ",
        Input(MOMENT_1).dim(0));

    const int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);
    auto* weightGradsOutput =
        Output(AUX_GRAD, indicesInput.sizes(), at::dtype<T>());

    T* out_weight_grads = weightGradsOutput->template mutable_data<T>();

    if (len_length <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const T* grad = Input(GRAD).template data<T>();
    const T* weights = weightsInput.template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<THalf>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<THalf>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (post > 128) {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          512><<<len_length, 512, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    } else if (post > 64) {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          128><<<len_length, 128, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    } else if (post > 32) {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          64><<<len_length, 64, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    } else {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          THalf,
          T,
          32><<<len_length, 32, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          post,
          len_length,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr);
    }

    return true;
  }

 private:
  // menber field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};

 protected:
  T epsilon_;
  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

REGISTER_CUDA_OPERATOR(
    SparseAdagradFusedWithSparseLengthsSumGradient,
    CUDASparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    CUDASparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        float,
        int,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    CUDASparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        float,
        int,
        CUDAContext>);

REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradient,
    CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradient,
    CUDARowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        float,
        int,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientApprox,
    CUDARowWiseSparseAdagradFusedWithSparseLengthsWeightedSumGradientOp<
        float,
        int,
        CUDAContext>);

} // namespace caffe2

#undef SEGREDUCE_MINBLOCKS
