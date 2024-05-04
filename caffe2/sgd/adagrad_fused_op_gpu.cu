#include <ATen/ATen.h>
#include <c10/core/GeneratorImpl.h>
#include <algorithm>

#include "caffe2/utils/cub_namespace.cuh"
#include <cub/device/device_radix_sort.cuh>
#include "caffe2/sgd/adagrad_fused_op_gpu.cuh"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

void inclusive_scan_wrapper(
    const int* length_data,
    int num_lengths,
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
      num_lengths,
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
      num_lengths,
      context_->cuda_stream());
}

template <typename SIndex>
void sort_pairs_wrapper(
    int num_indices,
    int num_rows,
    Tensor* temp_buffer,
    const Tensor* linear_ind_buffer_,
    Tensor* sorted_linear_ind_buffer_,
    const Tensor* seg_id_buffer_,
    Tensor* sorted_seg_id_buffer_,
    CUDAContext* context_) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      temp_storage_bytes,
      linear_ind_buffer_->template data<SIndex>(),
      sorted_linear_ind_buffer_->template mutable_data<SIndex>(),
      seg_id_buffer_->template data<int>(),
      sorted_seg_id_buffer_->template mutable_data<int>(),
      num_indices,
      0,
      int(log2(float(num_rows)) + 1),
      context_->cuda_stream(),
      false);

  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(int)) / sizeof(int);
  temp_buffer->Resize(buffer_size);
  void* d_temp_storage =
      static_cast<void*>(temp_buffer->template mutable_data<int>());

  cub::DeviceRadixSort::SortPairs(
      d_temp_storage,
      temp_storage_bytes,
      linear_ind_buffer_->template data<SIndex>(),
      sorted_linear_ind_buffer_->template mutable_data<SIndex>(),
      seg_id_buffer_->template data<int>(),
      sorted_seg_id_buffer_->template mutable_data<int>(),
      num_indices,
      0,
      int(log2(float(num_rows)) + 1),
      context_->cuda_stream(),
      false);
}

template <typename T>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void gradient_mean_kernel(
    const T* __restrict__ grad_in,
    const int* __restrict__ lengths,
    T* __restrict__ grad_out,
    int block_size) {
  int group = blockIdx.x;

  for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    grad_out[group * block_size + i] = lengths[group] > 0
        ? grad_in[group * block_size + i] / lengths[group]
        : grad_in[group * block_size + i];
  }
}

template <typename SIndex, typename TParam, typename T, bool ExactBlock = false>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void sparse_adagrad_fused_length_sum_gradient_kernel(
    const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                    // (offsets for the
                                                    // segments)
    int N, // number of rows (hash size) of embedding table
    int block_size, // embedding dimension size
    const float epsilon,
    TParam* param,
    TParam* param_mom,
    const SIndex* indices,
    const T* __restrict__ grad,
    const float* lr,
    float weight_decay = 0.f) {
  const float LR = lr[0];
  // num_lengths blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  if (ExactBlock) {
    const size_t gradIdx = group * block_size + threadIdx.x; // index for grad
    for (int line = start + threadIdx.y; line < end; line += blockDim.y) {
      // line: the idx in the indices
      // threadIdx.x: index in the embedding dimension
      const SIndex index =
          indices[line]; // the index-th row in the embedding table
      const size_t paramIdx =
          index * block_size + threadIdx.x; // index for param

      float gi = grad[gradIdx] + weight_decay * param[paramIdx];

      float mom_new = gi * gi + param_mom[paramIdx];
      param_mom[paramIdx] = mom_new;
      float param_new = LR * gi / (sqrtf(mom_new) + epsilon) + param[paramIdx];
      param[paramIdx] = param_new;
    }
  } else {
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      // i: index in the embedding dimension
      const size_t gradIdx = group * block_size + i; // index for grad
      for (int line = start; line < end; ++line) {
        // line: the idx in the indices
        const SIndex index =
            indices[line]; // the index row in the embedding table
        const size_t paramIdx = index * block_size + i; // index for param

        float gi = grad[gradIdx] + weight_decay * param[paramIdx];

        float mom_new = gi * gi + param_mom[paramIdx];
        param_mom[paramIdx] = mom_new;
        float param_new =
            LR * gi / (sqrtf(mom_new) + epsilon) + param[paramIdx];
        param[paramIdx] = param_new;
      }
    }
  }
}

template <typename SIndex, typename TParam, typename T, int NumThreads>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void sparse_adagrad_fused_length_weighted_sum_gradient_kernel(
    const int *const __restrict__ prefix_sum_length_data,
    const int N, // number of rows (hash size) of embedding table
    const int block_size, // embedding dimension size
    const float epsilon,
    TParam *const param,
    TParam *const param_mom,
    const SIndex *const indices,
    const T *const __restrict__ grad,
    const T *const __restrict__ weights,
    T *const __restrict__ weights_grad_out,
    const float *const lr,
    const float weight_decay = 0.f) {
  const float LR = lr[0];
  // num_lengths blocks, each block process one segment
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
    // LengthsRangeFill + Gather operator. In the normal SLWS operator weight
    // is accessed via weights[line] but in most cases the weights are
    // generated by LengthsRangeFill and Gather operator.
    // For example, if lengths is [2, 3, 1] LengthsRangeFill will generate [0,
    // 1; 0, 1, 2; 0] and they are used as indices of Gather.
    // So if we fuse all of these, weights[line] just becomes
    // weights[line - start].
    auto in_weight_temp = weights[line - start];

    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      // i: index in the embedding dimension
      const size_t gradIdx = group * block_size + i; // index for in_grad
      const size_t paramIdx = index * block_size + i; // index for param

      // TODO: trying to reduce the variable number (common subexpression
      // elimination).
      auto in_grad_temp = grad[gradIdx];
      w_grad += in_grad_temp * param[paramIdx];

      auto out_grad_temp =
          in_weight_temp * in_grad_temp + weight_decay * param[paramIdx];

      // TODO: split it into two kernels to make it more similar to exact
      // fusion kernel (not Approx on CPUs).
      float mom_new = out_grad_temp * out_grad_temp + param_mom[paramIdx];
      param_mom[paramIdx] = mom_new;
      float param_new =
          LR * out_grad_temp / (sqrtf(mom_new) + epsilon) + param[paramIdx];
      param[paramIdx] = param_new;
    }

    w_grad = BlockReduce(temp_storage).Reduce(w_grad, cub::Sum());

    if (threadIdx.x == 0) {
      weights_grad_out[line] = w_grad;
    }
    __syncthreads();
  }
}

// Construct a reverse map of offset_of_idx -> segment_id.
template <typename SIndex>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void linear_index_weight_offsets_dedup_kernel(
    const int* __restrict__ prefix_sum_length_data, // prefix of lengths
    int* __restrict__ seg_id_data // segment id
) {
  // num_lengths blocks, each block process one segment
  int group = blockIdx.x; // the group-th segment
  int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  int end = prefix_sum_length_data[group]; // end offset of the segment

  for (int line = start + threadIdx.x; line < end; line += blockDim.x) {
    // line: the idx in the indices
    seg_id_data[line] = group;
  }
}

template <
    typename SIndex,
    typename TParam,
    typename T,
    bool ExactBlock = false,
    roundOption roundOpt = NEAREST>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__ void rowwise_sparse_adagrad_fused_length_sum_gradient_dedup_kernel(
    const int block_size, // embedding dimension size
    const int num_indices, // number of indices
    const float epsilon,
    TParam *const param,
    T *const param_mom,
    const T *const __restrict__ grad,
    const SIndex *const sorted_linear_ind_data, // sorted linear indices
    const int *const __restrict__ sorted_seg_id_data, // sorted segment id
    const float *const lr,
    ulong2 seed,
    float weight_decay = 0.f) {

  class randFactor<TParam, T, roundOpt> rand_factor(
      seed,
      blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x);

  const float LR = lr[0];
  // num_indices blocks, each block process one index
  int sorted_linear_indice_id;
  if (ExactBlock) {
    sorted_linear_indice_id =
        blockIdx.x * blockDim.y + threadIdx.y; // the index of sorted_linear_ind
  } else {
    sorted_linear_indice_id = blockIdx.x; // the index of sorted_linear_ind
  }
  if (sorted_linear_indice_id >= num_indices) {
    // don't have warp divergence when embedding dim is multiple of 32
    return;
  }

  // the index row in the embedding table
  SIndex index = sorted_linear_ind_data[sorted_linear_indice_id];

  // check if this thread block is responsible for this whole linear index
  bool linear_index_start =
      (sorted_linear_indice_id == 0 ||
       sorted_linear_ind_data[sorted_linear_indice_id - 1] != index);

  if (!linear_index_start) {
    // don't have warp divergence when embedding dim is multiple of 32
    return;
  }

  if (ExactBlock) {
    // find the num of duplicated indices.
    int num_dup = 1;
    while (true) {
      int segment_continue = 0;
      if (sorted_linear_indice_id + num_dup + threadIdx.x < num_indices) {
        segment_continue =
            sorted_linear_ind_data[sorted_linear_indice_id + num_dup + threadIdx.x] ==
            index;
      }
#if !defined(USE_ROCM)
      int32_t num_dup_incr = __popc(__ballot_sync(0xFFFFFFFF, segment_continue));
#else
      int32_t num_dup_incr = __popc(__ballot(segment_continue));
#endif
      num_dup += num_dup_incr;
      if (num_dup_incr != kWarpSize) {
        break;
      }
    }

    float sum_squares = 0.0;
    extern __shared__ float x_ij[];

    // we need to avoid index collision for the threads in the same block.
    // Different threadIdx.y works on different `index`.
    int sm_offset = threadIdx.y * block_size;

    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      // i: index in the embedding dimension
      float t_x_ij = 0.0;

      for (int dup_id = 0; dup_id < num_dup; dup_id++) {
        int group = sorted_seg_id_data[sorted_linear_indice_id + dup_id];
        t_x_ij += grad[group * block_size + i];
      }
      t_x_ij += weight_decay *
          rand_factor.convertTypeFromParamToTarget(param[index * block_size + i]);
      sum_squares += t_x_ij * t_x_ij;

      x_ij[sm_offset + i] = t_x_ij;
    }

    // We have a strong assumption that blockDim.x = 32, which is equal to the warp size.
    float row_sum_squares_avg = warpReduceAllSum<float>(sum_squares) / static_cast<float>(block_size);
    float mom_new = param_mom[index] + row_sum_squares_avg;
    param_mom[index] = mom_new;

    // update param
    float step = LR / (sqrtf(mom_new) + epsilon);
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      const size_t paramIdx = index * block_size + i; // index for param
      param[paramIdx] = rand_factor.convertTypeFromTargetToParam(
          rand_factor.convertTypeFromParamToTarget(param[paramIdx]) + x_ij[sm_offset + i] * step);
    }
  } else {
    // find the num of duplicated indices.
    int num_dup = 1;
    while (sorted_linear_indice_id + num_dup < num_indices &&
          sorted_linear_ind_data[sorted_linear_indice_id + num_dup] == index) {
      num_dup += 1;
    }

    // TODO: Tuning NumThreads for sum_squares
    typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
    __shared__ BlockReduce::TempStorage temp_storage;
    int valid = min(block_size, blockDim.x);

    float sum_squares = 0.0;
    __shared__ float row_sum_squares_avg;
    extern __shared__ float x_ij[];

    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      // i: index in the embedding dimension
      float t_x_ij = 0.0;

      for (int dup_id = 0; dup_id < num_dup; dup_id++) {
        int group = sorted_seg_id_data[sorted_linear_indice_id + dup_id];
        t_x_ij += grad[group * block_size + i];
      }
      t_x_ij += weight_decay *
          rand_factor.convertTypeFromParamToTarget(param[index * block_size + i]);
      sum_squares += t_x_ij * t_x_ij;
      x_ij[i] = t_x_ij;
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum_squares, valid);

    if (threadIdx.x == 0) {
      row_sum_squares_avg = reduce_result / static_cast<float>(block_size);
      float mom_new = param_mom[index] + row_sum_squares_avg;
      param_mom[index] = mom_new;
    }
    __syncthreads();

    // update param
    float step = LR / (sqrtf(param_mom[index]) + epsilon);
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      const size_t paramIdx = index * block_size + i; // index for param
      param[paramIdx] = rand_factor.convertTypeFromTargetToParam(
        rand_factor.convertTypeFromParamToTarget(param[paramIdx]) + x_ij[i] * step);
    }
  }
}

template <typename SIndex, typename TParam, typename T, int NumThreads>
#if defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(1024, SEGREDUCE_MINBLOCKS)
#endif
__global__
    void rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel(
        const int* __restrict__ prefix_sum_length_data, // prefix of lengths
                                                        // (offsets for the
                                                        // segments)
        const int N, // number of rows (hash size) of embedding table
        const int block_size, // embedding dimension size
        const float epsilon,
        TParam* param,
        T* param_mom,
        const SIndex* indices,
        const T* __restrict__ grad,
        const T* __restrict__ weights,
        T* __restrict__ weights_grad_out,
        const float* lr,
        float weight_decay = 0.f) {
  const float LR = lr[0];
  // num_lengths blocks, each block process one segment
  const int group = blockIdx.x; // the group-th segment
  const int start = group == 0
      ? 0
      : prefix_sum_length_data[group - 1]; // start offset of the segment
  const int end = prefix_sum_length_data[group]; // end offset of the segment
  CUDA_KERNEL_ASSERT(start <= N);
  CUDA_KERNEL_ASSERT(end <= N);

  // TODO: Tuning NumThreads for w_grad
  typedef cub::BlockReduce<float, NumThreads> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const int valid = min(block_size, blockDim.x);

  // for avg_square_weight. Can we reuse temp_storage
  __shared__ typename BlockReduce::TempStorage temp_storage2;

  // TODO(jianyuhuang): parallelize this outer loop
  for (int line = start; line < end; ++line) {
    T w_grad = 0;
    // i: index in the embedding dimension
    const SIndex index = indices[line];

    auto in_weight_temp = weights[line - start];

    float sum_squares = 0.0;
    __shared__ float row_sum_squares_avg;

    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      const float x_ij = grad[group * block_size + i] * in_weight_temp +
          weight_decay * param[index * block_size + i];
      sum_squares += x_ij * x_ij;
    }
    float reduce_result = BlockReduce(temp_storage2).Sum(sum_squares, valid);

    if (threadIdx.x == 0) {
      row_sum_squares_avg = reduce_result / static_cast<float>(block_size);
      param_mom[index] += static_cast<T>(row_sum_squares_avg);
    }
    __syncthreads();

    // update param
    float step = LR / (sqrtf(param_mom[index]) + epsilon);

    for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
      const size_t gradIdx = group * block_size + i; // index for in_grad
      const size_t paramIdx = index * block_size + i; // index for param
      // TODO: trying to reduce the variable number (common subexpression
      // elimination).
      auto in_grad_temp = grad[gradIdx];
      w_grad += in_grad_temp * param[paramIdx];
      auto out_grad_temp =
          in_weight_temp * in_grad_temp + weight_decay * param[paramIdx];

      // TODO: split it into two kernels to make it more similar to exact
      // fusion kernel (not Approx on CPUs).
      param[paramIdx] = out_grad_temp * step + param[paramIdx];
    }
    w_grad = BlockReduce(temp_storage).Reduce(w_grad, cub::Sum());

    if (threadIdx.x == 0) {
      weights_grad_out[line] = w_grad;
    }
    __syncthreads();
  }
}

} // namespace

template <typename T, typename TLengths, bool is_mean, class Context>
class CUDASparseAdagradFusedWithSparseLengthsSumGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseAdagradFusedWithSparseLengthsSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "CUDASparseAdagradFusedWithSparseLengthSumGradientOp"
            << " weight_decay_=" << weight_decay_;

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

  template <typename IndexType, typename TParam>
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

    const int num_lengths = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(num_lengths == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);

    if (num_lengths <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        num_lengths,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const auto* lengths = lengthsInput.template data<int>();
    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const T* grad = Input(GRAD).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<TParam>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<TParam>();

    int N = output_0dim;
    int block_size = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (is_mean) {
      grad_buffer_.ResizeLike(segmentGradsInput);
    }
    auto* grad_buffer_data =
        is_mean ? grad_buffer_.template mutable_data<T>() : NULL;
    if (is_mean) {
      gradient_mean_kernel<T>
          <<<num_lengths,
             std::min(maxThreads, block_size),
             0,
             context_.cuda_stream()>>>(
              grad, lengths, grad_buffer_data, block_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (block_size <= maxThreads) {
      int multiple = std::min(maxThreads / block_size, SEGREDUCE_MINBLOCKS);
      dim3 block(block_size, multiple);
      // calling cuda kernel with ExactBlock = true
      sparse_adagrad_fused_length_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          true><<<num_lengths, block, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          is_mean ? grad_buffer_data : grad,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      // calling cuda kernel with ExactBlock = false
      sparse_adagrad_fused_length_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          false><<<num_lengths, maxThreads, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          is_mean ? grad_buffer_data : grad,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
  Tensor grad_buffer_{CUDA};

 protected:
  T epsilon_;
  T weight_decay_;
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
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "CUDASparseAdagradFusedWithSparseLengthWeightedSumGradientOp"
            << " weight_decay_=" << weight_decay_;

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

  template <typename IndexType, typename TParam>
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

    const int num_lengths = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(num_lengths == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);
    auto* weightGradsOutput =
        Output(AUX_GRAD, indicesInput.sizes(), at::dtype<T>());

    T* out_weight_grads = weightGradsOutput->template mutable_data<T>();

    if (num_lengths <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        num_lengths,
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
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<TParam>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<TParam>();

    int N = output_0dim;
    int block_size = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (block_size > 128) {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          512><<<num_lengths, 512, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (block_size > 64) {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          128><<<num_lengths, 128, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (block_size > 32) {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          64><<<num_lengths, 64, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          32><<<num_lengths, 32, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};

 protected:
  T epsilon_;
  T weight_decay_;
  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

template <typename T, typename TLengths, bool is_mean, class Context>
class CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        round_option_((roundOption)this->template GetSingleArgument<int>(
            "round_option",
            NEAREST)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "CUDARowWiseSparseAdagradFusedWithSparseLengthSumGradientOp"
            << " weight_decay_=" << weight_decay_;

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

  template <typename IndexType, typename TParam>
  bool DoRunWithType2() {
    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);
    auto& indicesInput = Input(INDICES);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_GT(Input(GRAD).dim(), 0);

    // Enforce:
    // number of rows: input(embedding/momentum) ==
    // outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).dim(0),
        Input(MOMENT_1).dim(0),
        "Input Param number of rows: ",
        Input(PARAM).dim(0),
        " Input Moment size: ",
        Input(MOMENT_1).dim(0));

    const int num_lengths = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(num_lengths == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);

    if (num_lengths <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        num_lengths,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const auto* lengths = lengthsInput.template data<int>();
    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const T* grad = Input(GRAD).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<TParam>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    int N = output_0dim;
    int block_size = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;
    ulong2 seed;

    if (is_mean) {
      grad_buffer_.ResizeLike(segmentGradsInput);
    }
    auto* grad_buffer_data =
        is_mean ? grad_buffer_.template mutable_data<T>() : NULL;
    if (is_mean) {
      gradient_mean_kernel<T>
          <<<num_lengths,
             std::min(maxThreads, block_size),
             0,
             context_.cuda_stream()>>>(
              grad, lengths, grad_buffer_data, block_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // 0: nearest rounding
    // 1: stochastic rounding
    if (round_option_ == STOCHASTIC) {
      seed.x = default_rng_seed_val;
      seed.y = maxThreads * block_size;
    }
    if (block_size <= maxThreads / 2 && block_size % 32 == 0) {
      // Fast path when the embedding dimension is a multiple of 32, using
      // WarpReduce.
      int multiple = std::min(maxThreads / block_size, SEGREDUCE_MINBLOCKS);
      dim3 block(block_size, multiple);
      if (round_option_ == STOCHASTIC) {
        rowwise_sparse_adagrad_fused_length_sum_gradient_kernel<
            IndexType,
            TParam,
            T,
            true,
            STOCHASTIC><<<num_lengths, block, 0, context_.cuda_stream()>>>(
            prefix_sum_length_data,
            N,
            block_size,
            num_lengths,
            epsilon_,
            paramOut,
            momentOut,
            indices,
            is_mean ? grad_buffer_data : grad,
            lr,
            seed,
            weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        rowwise_sparse_adagrad_fused_length_sum_gradient_kernel<
            IndexType,
            TParam,
            T,
            true,
            NEAREST><<<num_lengths, block, 0, context_.cuda_stream()>>>(
            prefix_sum_length_data,
            N,
            block_size,
            num_lengths,
            epsilon_,
            paramOut,
            momentOut,
            indices,
            is_mean ? grad_buffer_data : grad,
            lr,
            seed,
            weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      if (round_option_) {
        rowwise_sparse_adagrad_fused_length_sum_gradient_kernel<
            IndexType,
            TParam,
            T,
            false,
            STOCHASTIC>
            <<<num_lengths,
               std::min(maxThreads, block_size),
               0,
               context_.cuda_stream()>>>(
                prefix_sum_length_data,
                N,
                block_size,
                num_lengths,
                epsilon_,
                paramOut,
                momentOut,
                indices,
                is_mean ? grad_buffer_data : grad,
                lr,
                seed,
                weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        rowwise_sparse_adagrad_fused_length_sum_gradient_kernel<
            IndexType,
            TParam,
            T,
            false,
            NEAREST>
            <<<num_lengths,
               std::min(maxThreads, block_size),
               0,
               context_.cuda_stream()>>>(
                prefix_sum_length_data,
                N,
                block_size,
                num_lengths,
                epsilon_,
                paramOut,
                momentOut,
                indices,
                is_mean ? grad_buffer_data : grad,
                lr,
                seed,
                weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};
  Tensor grad_buffer_{CUDA};

 protected:
  T epsilon_;
  roundOption round_option_;
  T weight_decay_;
  INPUT_TAGS(PARAM, MOMENT_1, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1);
};

template <typename T, typename TLengths, bool is_mean, class Context>
class CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientExactOp final
    : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientExactOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        round_option_((roundOption)this->template GetSingleArgument<int>(
            "round_option",
            NEAREST)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1) << "gradient optimization operator in use: "
            << "CUDARowWiseSparseAdagradFusedWithSparseLengthSumGradientOp"
            << " weight_decay_=" << weight_decay_;

    CAFFE_ENFORCE(
        round_option_ == STOCHASTIC || round_option_ == NEAREST,
        "round_option_ should be either NEAREST or STOCHATIC");

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

  template <typename IndexType, typename TParam>
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

    const int num_lengths = lengthsInput.dim(0);
    const int num_indices = indicesInput.dim(0);
    const int num_rows = Input(PARAM).dim(0);

    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(num_lengths == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);

    if (num_lengths <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        num_lengths,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    // compute output size using length
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();

    const auto* lengths = lengthsInput.template data<int>();
    const auto* lr = Input(LR).template data<T>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const T* grad = Input(GRAD).template data<T>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<TParam>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    int block_size = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (is_mean) {
      grad_buffer_.ResizeLike(segmentGradsInput);
    }
    auto* grad_buffer_data =
        is_mean ? grad_buffer_.template mutable_data<T>() : NULL;
    if (is_mean) {
      gradient_mean_kernel<T>
          <<<num_lengths,
             std::min(maxThreads, block_size),
             0,
             context_.cuda_stream()>>>(
              grad, lengths, grad_buffer_data, block_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    sorted_linear_ind_buffer_.ResizeLike(indicesInput);
    seg_id_buffer_.ResizeLike(indicesInput);
    sorted_seg_id_buffer_.ResizeLike(indicesInput);

    linear_index_weight_offsets_dedup_kernel<IndexType>
        <<<num_lengths, 32, 0, context_.cuda_stream()>>>(
            prefix_sum_length_data,
            seg_id_buffer_.template mutable_data<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    sort_pairs_wrapper<IndexType>(
        num_indices,
        num_rows,
        &sort_buffer_,
        &indicesInput,
        &sorted_linear_ind_buffer_,
        &seg_id_buffer_,
        &sorted_seg_id_buffer_,
        &context_);

    ulong2 seed;

    // 0: nearest rounding
    // 1: stochastic rounding
    if (round_option_ == STOCHASTIC) {
      seed.x = default_rng_seed_val;
      seed.y = maxThreads * block_size;
    }

    if (block_size <= maxThreads / 2 && block_size % 32 == 0) {
      // Fast path when the embedding dimension is a multiple of 32, using
      // WarpReduce.
      constexpr int kWarpNum = 8;
      const dim3 threads(kWarpSize, kWarpNum);
      const dim3 blocks((num_indices + kWarpNum - 1) / kWarpNum);
      CAFFE_ENFORCE_LE(
          kWarpNum * kWarpSize,
          maxThreads,
          "the total number of threads in a block should be smaller than or equal to maxThreads");

      const int sm_size = block_size * kWarpNum * sizeof(float);
      // Maximum shared memory allocated per thread block is 48 KB on Maxwell/Pascal
      CAFFE_ENFORCE_LE(
        sm_size,
        1024 * 48,
        "Block size is too big and will exceed the max size of the shared memory");

      if (round_option_ == STOCHASTIC) {
        rowwise_sparse_adagrad_fused_length_sum_gradient_dedup_kernel<
            IndexType,
            TParam,
            T,
            true,
            STOCHASTIC>
            <<<blocks,
               threads,
               sm_size,
               context_.cuda_stream()>>>(
                block_size,
                num_indices,
                epsilon_,
                paramOut,
                momentOut,
                is_mean ? grad_buffer_data : grad,
                sorted_linear_ind_buffer_.template data<IndexType>(),
                sorted_seg_id_buffer_.template data<int>(),
                lr,
                seed,
                weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        rowwise_sparse_adagrad_fused_length_sum_gradient_dedup_kernel<
            IndexType,
            TParam,
            T,
            true,
            NEAREST>
            <<<blocks,
               threads,
               sm_size,
               context_.cuda_stream()>>>(
                block_size,
                num_indices,
                epsilon_,
                paramOut,
                momentOut,
                is_mean ? grad_buffer_data : grad,
                sorted_linear_ind_buffer_.template data<IndexType>(),
                sorted_seg_id_buffer_.template data<int>(),
                lr,
                seed,
                weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      const int sm_size = block_size * sizeof(float);
      // Maximum shared memory allocated per thread block is 48 KB on Maxwell/Pascal
      CAFFE_ENFORCE_LE(
        sm_size,
        1024 * 48,
        "Block size is too big and will exceed the max size of the shared memory");
      if (round_option_ == STOCHASTIC) {
        rowwise_sparse_adagrad_fused_length_sum_gradient_dedup_kernel<
            IndexType,
            TParam,
            T,
            false,
            STOCHASTIC>
            <<<num_indices,
               std::min(maxThreads, block_size),
               sm_size,
               context_.cuda_stream()>>>(
                block_size,
                num_indices,
                epsilon_,
                paramOut,
                momentOut,
                is_mean ? grad_buffer_data : grad,
                sorted_linear_ind_buffer_.template data<IndexType>(),
                sorted_seg_id_buffer_.template data<int>(),
                lr,
                seed,
                weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        rowwise_sparse_adagrad_fused_length_sum_gradient_dedup_kernel<
            IndexType,
            TParam,
            T,
            false,
            NEAREST>
            <<<num_indices,
               std::min(maxThreads, block_size),
               sm_size,
               context_.cuda_stream()>>>(
                block_size,
                num_indices,
                epsilon_,
                paramOut,
                momentOut,
                is_mean ? grad_buffer_data : grad,
                sorted_linear_ind_buffer_.template data<IndexType>(),
                sorted_seg_id_buffer_.template data<int>(),
                lr,
                seed,
                weight_decay_);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }

    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};

  Tensor sort_buffer_{CUDA};
  Tensor sorted_linear_ind_buffer_{CUDA};
  Tensor seg_id_buffer_{CUDA};
  Tensor sorted_seg_id_buffer_{CUDA};
  Tensor grad_buffer_{CUDA};

 protected:
  T epsilon_;
  roundOption round_option_;
  T weight_decay_;
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
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)),
        weight_decay_(
            this->template GetSingleArgument<float>("weight_decay", 0.f)) {
    VLOG(1)
        << "gradient optimization operator in use: "
        << "CUDARowWiseSparseAdagradFusedWithSparseLengthWeightedSumGradientOp"
        << " weight_decay_=" << weight_decay_;

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

  template <typename IndexType, typename TParam>
  bool DoRunWithType2() {
    auto& segmentGradsInput = Input(GRAD);
    auto& lengthsInput = Input(LENGTHS);
    auto& indicesInput = Input(INDICES);
    auto& weightsInput = Input(AUX_PARAM);

    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    CAFFE_ENFORCE_EQ(1, weightsInput.dim(), "WEIGHTS must be a vector");
    CAFFE_ENFORCE_GT(Input(GRAD).dim(), 0);

    // Enforce:
    // number of rows: input(embedding/momentum) ==
    // outputs(embedding/momentum)
    CAFFE_ENFORCE_EQ(
        Input(PARAM).dim(0),
        Input(MOMENT_1).dim(0),
        "Input Param number of rows: ",
        Input(PARAM).dim(0),
        " Input Moment size: ",
        Input(MOMENT_1).dim(0));

    const int num_lengths = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.dim() > 0);
    CAFFE_ENFORCE(num_lengths == segmentGradsInput.dim(0));

    int output_0dim = indicesInput.dim(0);
    auto* weightGradsOutput =
        Output(AUX_GRAD, indicesInput.sizes(), at::dtype<T>());

    T* out_weight_grads = weightGradsOutput->template mutable_data<T>();

    if (num_lengths <= 0) {
      // return early to avoid invalid empty kernel
      return true;
    }

    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        num_lengths,
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
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<TParam>();
    auto* momentOut = Output(OUTPUT_MOMENT_1)->template mutable_data<T>();

    int N = output_0dim;
    int block_size = segmentGradsInput.size_from_dim(1);

    auto maxThreads =
        GetDeviceProperty(CaffeCudaGetDevice()).maxThreadsPerBlock;

    if (block_size > 128) {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          512><<<num_lengths, 512, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (block_size > 64) {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          128><<<num_lengths, 128, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (block_size > 32) {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          64><<<num_lengths, 64, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      rowwise_sparse_adagrad_fused_length_weighted_sum_gradient_kernel<
          IndexType,
          TParam,
          T,
          32><<<num_lengths, 32, 0, context_.cuda_stream()>>>(
          prefix_sum_length_data,
          N,
          block_size,
          epsilon_,
          paramOut,
          momentOut,
          indices,
          grad,
          weights,
          out_weight_grads,
          lr,
          weight_decay_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return true;
  }

 private:
  // member field to manage memory
  Tensor inclusive_scan_buffer_{CUDA};
  Tensor inclusive_scan_length_buffer_{CUDA};

 protected:
  T epsilon_;
  T weight_decay_;
  INPUT_TAGS(PARAM, MOMENT_1, AUX_PARAM, INDICES, GRAD, LR, LENGTHS);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_1, AUX_GRAD);
};

// For GPU, the implementation of the exact and approx (RowWise)SparseAdagrad
// fusion are both approximate implementations.
// When we don't have the duplicated indices, the outputs are the same as the
// CPU implementation.
REGISTER_CUDA_OPERATOR(
    SparseAdagradFusedWithSparseLengthsSumGradient,
    CUDASparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        false,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseAdagradFusedWithSparseLengthsSumGradientApprox,
    CUDASparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        false,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseAdagradFusedWithSparseLengthsMeanGradient,
    CUDASparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        true,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseAdagradFusedWithSparseLengthsMeanGradientApprox,
    CUDASparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        true,
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
    CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientExactOp<
        float,
        int,
        false,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsSumGradientApprox,
    CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        false,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradient,
    CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientExactOp<
        float,
        int,
        true,
        CUDAContext>);
REGISTER_CUDA_OPERATOR(
    RowWiseSparseAdagradFusedWithSparseLengthsMeanGradientApprox,
    CUDARowWiseSparseAdagradFusedWithSparseLengthsSumGradientOp<
        float,
        int,
        true,
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
