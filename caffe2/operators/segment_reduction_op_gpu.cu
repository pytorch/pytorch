#include <cub/device/device_scan.cuh>
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

#include <cub/block/block_reduce.cuh>

namespace caffe2 {

template <
    typename T,
    class Context = CUDAContext,
    bool FIRSTDIMS = true,
    bool NORMALIZE = false>
class ReduceDimsOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReduceDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  ~ReduceDimsOp() {}

  bool RunOnDevice() override {
    const auto& input = Input(0);
    const auto* input_data = input.template data<T>();
    auto* Y = Output(0);

    CHECK_LT(num_reduce_dims_, input.dims().size());
    const int M = FIRSTDIMS
        ? input.size_to_dim(num_reduce_dims_)
        : input.size_to_dim(input.ndim() - num_reduce_dims_);
    const int N = FIRSTDIMS
        ? input.size_from_dim(num_reduce_dims_)
        : input.size_from_dim(input.ndim() - num_reduce_dims_);

    vector<TIndex> output_shape;
    int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
    int end_index = FIRSTDIMS ? input.dims().size()
                              : input.dims().size() - num_reduce_dims_;
    for (int i = start_index; i < end_index; ++i) {
      output_shape.push_back(input.dims()[i]);
    }

    Y->Resize(output_shape);

    int in_dim = FIRSTDIMS ? M : N;

    if (ones_.size() != in_dim) {
      ones_.Resize(in_dim);
      math::Set<T, Context>(
          in_dim,
          static_cast<T>(1),
          ones_.template mutable_data<T>(),
          &context_);
    }

    T alpha = 1.0;
    if (NORMALIZE) { // Static if
      alpha = 1.0 / in_dim;
    }

    math::Gemv<T, Context>(
        FIRSTDIMS ? CblasTrans : CblasNoTrans,
        M,
        N,
        alpha,
        input_data,
        ones_.template data<T>(),
        0.0,
        Y->template mutable_data<T>(),
        &context_);

    return true;
  }

 private:
  Tensor<Context> ones_;
  int num_reduce_dims_;
};

namespace {
template <typename T>
__global__ void
StripedScaleKernel(const int N, const int M, const T* alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, M * N) {
    int k = i / N;
    y[i] = x[i] * alpha[k];
  }
}

template <typename T>
__global__ void StripedAxpbyKernel(
    const int N,
    const int M,
    const T a,
    const T* x,
    const T b,
    T* y) {
  CUDA_1D_KERNEL_LOOP(index, N * M) {
    y[index] = x[index % N] * a + y[index] * b;
  }
}
} // namespace

template <
    typename T,
    class Context = CUDAContext,
    bool FIRSTDIMS = true,
    bool NORMALIZE = false>
class ReduceDimsGradientOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReduceDimsGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        num_reduce_dims_(
            OperatorBase::GetSingleArgument<int32_t>("num_reduce_dim", 1)) {}

  ~ReduceDimsGradientOp() {}

  bool RunOnDevice() override {
    const auto& grad_in = Input(0);
    const auto& in_shape = Input(1);

    shape_.CopyFrom(in_shape);
    // Copy first dims
    vector<TIndex> output_shape(
        shape_.template data<TIndex>(),
        shape_.template data<TIndex>() + shape_.size());

    auto* out_grad = Output(0);
    out_grad->Resize(output_shape);

    const int M = FIRSTDIMS
        ? out_grad->size_to_dim(num_reduce_dims_)
        : out_grad->size_to_dim(out_grad->ndim() - num_reduce_dims_);
    const int N = FIRSTDIMS
        ? out_grad->size_from_dim(num_reduce_dims_)
        : out_grad->size_from_dim(out_grad->ndim() - num_reduce_dims_);

    int in_dim = FIRSTDIMS ? M : N;

    T alpha = 1.0;
    if (NORMALIZE) { // Static if
      alpha = 1.0 / in_dim;
    }

    math::Set<T, CUDAContext>(
        out_grad->size(),
        FIRSTDIMS ? static_cast<T>(0) : static_cast<T>(alpha),
        out_grad->template mutable_data<T>(),
        &context_);

    if (FIRSTDIMS) {
      StripedAxpbyKernel<T>
          <<<CAFFE_GET_BLOCKS(N * M),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(
              N,
              M,
              alpha,
              grad_in.template data<T>(),
              static_cast<T>(0),
              out_grad->template mutable_data<T>());
    } else {
      StripedScaleKernel<T>
          <<<CAFFE_GET_BLOCKS(N * M),
             CAFFE_CUDA_NUM_THREADS,
             0,
             context_.cuda_stream()>>>(
              N,
              M,
              grad_in.template data<T>(),
              out_grad->template data<T>(),
              out_grad->template mutable_data<T>());
    }

    return true;
  }

 private:
  int num_reduce_dims_;
  Tensor<CPUContext> shape_;
};

REGISTER_CUDA_OPERATOR(ReduceFrontSum, ReduceDimsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontSumGradient,
    ReduceDimsGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMean,
    ReduceDimsOp<float, CUDAContext, true, true>);
REGISTER_CUDA_OPERATOR(
    ReduceFrontMeanGradient,
    ReduceDimsGradientOp<float, CUDAContext, true, true>);

REGISTER_CUDA_OPERATOR(ReduceBackSum, ReduceDimsOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR(
    ReduceBackSumGradient,
    ReduceDimsGradientOp<float, CUDAContext, false, false>);

REGISTER_CUDA_OPERATOR(
    ReduceBackMean,
    ReduceDimsOp<float, CUDAContext, false, true>);
REGISTER_CUDA_OPERATOR(
    ReduceBackMeanGradient,
    ReduceDimsGradientOp<float, CUDAContext, false, true>);

namespace {

void inclusive_scan_wrapper(
    const int* length_data,
    int len_length,
    Tensor<CUDAContext>* prefix_sum_buffer,
    Tensor<CUDAContext>* prefix_sum_length_buffer,
    CUDAContext* context_) {
  // inclusive scan
  size_t temp_storage_bytes = 0;
  // Retrieve buffer size
  cub::DeviceScan::InclusiveSum(
      NULL,
      temp_storage_bytes,
      length_data,
      prefix_sum_length_buffer->template mutable_data<int>(),
      len_length,
      context_->cuda_stream());
  // Allocate temporary storage
  auto buffer_size = temp_storage_bytes / sizeof(int);
  buffer_size += temp_storage_bytes % sizeof(int) != 0 ? 1 : 0;
  prefix_sum_buffer->Resize(buffer_size);
  void* d_temp_storage =
      static_cast<void*>(prefix_sum_buffer->template mutable_data<int>());
  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      length_data,
      prefix_sum_length_buffer->template mutable_data<int>(),
      len_length,
      context_->cuda_stream());
}

template <typename T>
__global__ void length_sum_kernel(
    const T* in,
    T* out,
    const int* prefix_sum_length_data,
    int N,
    int post,
    int len_length) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start < N);
  CUDA_KERNEL_ASSERT(end <= N);

  for (int i = threadIdx.x; i < post; i += blockDim.x) {
    T sum = (T)0;
    for (int line = start; line < end; ++line) {
      sum += in[line * post + i];
    }
    out[group * post + i] = sum;
  }
}

template <typename T>
__global__ void length_sum_gradient_kernel(
    const T* grad_in,
    T* grad_out,
    const int* prefix_sum_length_data,
    int N,
    int post,
    int len_length) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start < N);
  CUDA_KERNEL_ASSERT(end <= N);

  for (int i = threadIdx.x; i < post; i += blockDim.x) {
    for (int line = start; line < end; ++line) {
      grad_out[line * post + i] = grad_in[group * post + i];
    }
  }
}

template <typename T>
__global__ void sparse_length_sum_kernel(
    const T* in,
    T* out,
    const int* prefix_sum_length_data,
    const TIndex* indicies,
    int N,
    int post,
    int len_length,
    int len_indicies) {
  // len_length blocks
  int group = blockIdx.x;

  int start = group == 0 ? 0 : prefix_sum_length_data[group - 1];
  int end = prefix_sum_length_data[group];
  CUDA_KERNEL_ASSERT(start < len_indicies);
  CUDA_KERNEL_ASSERT(end <= len_indicies);

  for (int i = threadIdx.x; i < post; i += blockDim.x) {
    T sum = (T)0;
    for (int line = start; line < end; ++line) {
      sum += in[indicies[line] * post + i];
    }
    out[group * post + i] = sum;
  }
}

} // namespace

template <typename T, class Context = CUDAContext, bool SparseFused = true>
class CUDASparseLengthsSumOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsSumOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsSumOp() {}

  bool RunOnDevice() override {
    auto& dataInput = Input(0);
    auto& lengthsInput = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");
    const TIndex dataSize = dataInput.dim(0);
    // Either first dim the data or how much we pull in indexies from it
    TIndex dataToReduceSize;
    const TIndex outputSize = lengthsInput.dim(0);
    int len_length = outputSize;

    const TIndex* indicies;
    if (SparseFused) { // static if
      auto& indiciesInput = Input(INDICES);
      CAFFE_ENFORCE_EQ(1, indiciesInput.ndim(), "INDICES must be a vector");
      indicies = indiciesInput.template data<TIndex>();
      dataToReduceSize = indiciesInput.dim(0);
    } else {
      dataToReduceSize = dataSize;
    }

    auto shape = dataInput.dims();
    shape[0] = outputSize;
    output->Resize(shape);

    // only compute this the first time
    inclusive_scan_length_buffer_.ResizeLike(lengthsInput);
    inclusive_scan_wrapper(
        lengthsInput.template data<int>(),
        len_length,
        &inclusive_scan_buffer_,
        &inclusive_scan_length_buffer_,
        &context_);

    const T* in_data = dataInput.template data<T>();
    T* out_data = output->template mutable_data<T>();
    auto* prefix_sum_length_data =
        inclusive_scan_length_buffer_.template data<int>();
    int N = dataSize;
    int post = dataInput.size_from_dim(1);

    if (SparseFused) {
      if (post > 128) {
        sparse_length_sum_kernel<T>
            <<<len_length, 512, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indicies,
                N,
                post,
                len_length,
                dataToReduceSize);
      } else if (post > 64) {
        sparse_length_sum_kernel<T>
            <<<len_length, 128, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indicies,
                N,
                post,
                len_length,
                dataToReduceSize);
      } else if (post > 32) {
        sparse_length_sum_kernel<T>
            <<<len_length, 64, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indicies,
                N,
                post,
                len_length,
                dataToReduceSize);
      } else {
        sparse_length_sum_kernel<T>
            <<<len_length, 32, 0, context_.cuda_stream()>>>(
                in_data,
                out_data,
                prefix_sum_length_data,
                indicies,
                N,
                post,
                len_length,
                dataToReduceSize);
      }
    } else {
      if (post > 128) {
        length_sum_kernel<T><<<len_length, 512, 0, context_.cuda_stream()>>>(
            in_data, out_data, prefix_sum_length_data, N, post, len_length);
      } else if (post > 64) {
        length_sum_kernel<T><<<len_length, 128, 0, context_.cuda_stream()>>>(
            in_data, out_data, prefix_sum_length_data, N, post, len_length);
      } else if (post > 32) {
        length_sum_kernel<T><<<len_length, 64, 0, context_.cuda_stream()>>>(
            in_data, out_data, prefix_sum_length_data, N, post, len_length);
      } else {
        length_sum_kernel<T><<<len_length, 32, 0, context_.cuda_stream()>>>(
            in_data, out_data, prefix_sum_length_data, N, post, len_length);
      }
    }
    return true;
  }

  enum { INDICES = 1, LENGTHS = 1 + (SparseFused ? 1 : 0) };

 private:
  // menber field to manage memory
  Tensor<Context> inclusive_scan_buffer_;
  Tensor<Context> inclusive_scan_length_buffer_;
};

template <typename T, class Context = CUDAContext>
class CUDASparseLengthsSumGradientOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  CUDASparseLengthsSumGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDASparseLengthsSumGradientOp() {}

  bool RunOnDevice() override {
    auto& segmentGradsInput = Input(0);
    auto& lengthsInput = Input(1);
    auto* dataGradsOutput = Output(0);
    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");
    int len_length = lengthsInput.dim(0);
    CAFFE_ENFORCE(segmentGradsInput.ndim() > 0);
    CAFFE_ENFORCE(len_length == segmentGradsInput.dim(0));

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
    output_size_buffer_.Resize(1);
    auto* output_size = output_size_buffer_.template mutable_data<int>();

    context_.template CopyItems<Context, CPUContext>(
        inclusive_scan_length_buffer_.meta(),
        1,
        prefix_sum_length_data + len_length - 1,
        output_size);
    context_.FinishDeviceComputation();

    auto shape = segmentGradsInput.dims();
    int output_0dim = output_size[0];
    shape[0] = output_0dim;
    dataGradsOutput->Resize(shape);

    const T* in_data = segmentGradsInput.template data<T>();
    T* out_data = dataGradsOutput->template mutable_data<T>();

    int N = output_0dim;
    int post = segmentGradsInput.size_from_dim(1);

    if (post > 128) {
      length_sum_gradient_kernel<T>
          <<<len_length, 512, 0, context_.cuda_stream()>>>(
              in_data, out_data, prefix_sum_length_data, N, post, len_length);
    } else if (post > 64) {
      length_sum_gradient_kernel<T>
          <<<len_length, 128, 0, context_.cuda_stream()>>>(
              in_data, out_data, prefix_sum_length_data, N, post, len_length);
    } else if (post > 32) {
      length_sum_gradient_kernel<T>
          <<<len_length, 64, 0, context_.cuda_stream()>>>(
              in_data, out_data, prefix_sum_length_data, N, post, len_length);
    } else {
      length_sum_gradient_kernel<T>
          <<<len_length, 32, 0, context_.cuda_stream()>>>(
              in_data, out_data, prefix_sum_length_data, N, post, len_length);
    }

    return true;
  }

 private:
  // menber field to manage memory
  Tensor<Context> inclusive_scan_buffer_;
  Tensor<Context> inclusive_scan_length_buffer_;
  Tensor<CPUContext> output_size_buffer_;
};

template <typename SIndex>
__global__ void
MaxSegmentKernel(int n, const SIndex* segment_ids, SIndex* max_segment) {
  typedef cub::BlockReduce<SIndex, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int mx = 0;

  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    mx = segment_ids[j] > mx ? segment_ids[j] : mx;
  }
  SIndex max_seg = BlockReduce(temp_storage).Reduce(mx, cub::Max());
  if (threadIdx.x == 0) {
    *max_segment = max_seg;
  }
}

template <typename SIndex, typename T>
__global__ void UnsortedSegmentSumKernel(
    int n,
    int slize_sz,
    const SIndex* segments,
    const T* data,
    T* out,
    int* scales) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    int slice_idx = i / slize_sz;
    int j = i % slize_sz;
    SIndex segment = segments[slice_idx];
    atomicAdd(&out[segment * slize_sz + j], data[i]);
    if (scales && j == 0) {
      atomicAdd(&scales[segment], 1);
    }
  }
}

template <typename SIndex, typename T>
__global__ void
SegmentScalingKernel(int m, int slize_sz, const int* scales, T* out) {
  CUDA_1D_KERNEL_LOOP(i, m) {
    int scale = scales[i / slize_sz];
    out[i] = scale > 0 ? out[i] / scale : 0.0; // avoid 0/0 division
  }
}

template <typename T, typename SIndex, bool mean>
class CUDAUnsortedSegmentSumOp : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);
  CUDAUnsortedSegmentSumOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws) {}

  ~CUDAUnsortedSegmentSumOp() {}

  bool RunOnDevice() override {
    auto& data = Input(0);
    auto& segment_ids = Input(1);
    auto* output = Output(0);

    if (segment_ids.size() == 0 || data.size() == 0) {
      // Special handling for empty input
      auto dims = data.dims();
      if (dims.size() > 0) {
        dims[0] = 0;
      }
      output->Resize(dims);
      output->template mutable_data<T>();
      return true;
    }

    CAFFE_ENFORCE_EQ(1, segment_ids.ndim(), "SEGMENT_IDS must be a vector");
    TIndex slize_sz = data.size_from_dim(1);

    K_tensor_.Resize(1);

    // Get maximum segment id so we can size the output.
    // This must be done synchronously with host.
    MaxSegmentKernel<SIndex><<<
        std::min((int)segment_ids.size(), CAFFE_MAXIMUM_NUM_BLOCKS),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        segment_ids.size(),
        segment_ids.template data<SIndex>(),
        K_tensor_.mutable_data<SIndex>());

    SIndex K = 0;
    context_.CopyBytes<CUDAContext, CPUContext>(
        sizeof(SIndex), K_tensor_.data<SIndex>(), &K);
    context_.FinishDeviceComputation();

    auto dims = data.dims();
    dims[0] = K + 1;
    output->Resize(dims);

    // Clear the output as we will be accumulating the values
    math::Set<T, CUDAContext>(
        output->size(), T(0), output->template mutable_data<T>(), &context_);

    if (!mean) {
      UnsortedSegmentSumKernel<SIndex, T><<<
          CAFFE_GET_BLOCKS(data.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          data.size(),
          slize_sz,
          segment_ids.template data<SIndex>(),
          data.template data<T>(),
          output->template mutable_data<T>(),
          nullptr);
    } else {
      // For mean, we need to compute scaling factors
      scaling_factors_.Resize(K + 1);
      math::Set<int, CUDAContext>(
          scaling_factors_.size(),
          int(0),
          scaling_factors_.template mutable_data<int>(),
          &context_);
      UnsortedSegmentSumKernel<SIndex, T><<<
          CAFFE_GET_BLOCKS(data.size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          data.size(),
          slize_sz,
          segment_ids.template data<SIndex>(),
          data.template data<T>(),
          output->template mutable_data<T>(),
          scaling_factors_.template mutable_data<int>());
      // Divide by the scaling factors to get means
      SegmentScalingKernel<SIndex, T><<<
          CAFFE_GET_BLOCKS(output->size()),
          CAFFE_CUDA_NUM_THREADS,
          0,
          context_.cuda_stream()>>>(
          output->size(),
          slize_sz,
          scaling_factors_.template data<int>(),
          output->template mutable_data<T>());
    }
    return true;
  }

 private:
  Tensor<CUDAContext> K_tensor_;
  Tensor<CUDAContext> scaling_factors_; // for mean
};

REGISTER_CUDA_OPERATOR(
    LengthsSum,
    CUDASparseLengthsSumOp<float, CUDAContext, false>);
REGISTER_CUDA_OPERATOR(
    SparseLengthsSum,
    CUDASparseLengthsSumOp<float, CUDAContext, true>);
REGISTER_CUDA_OPERATOR(
    LengthsSumGradient,
    CUDASparseLengthsSumGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SparseLengthsSumGradient,
    CUDASparseLengthsSumGradientOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    UnsortedSegmentSum,
    CUDAUnsortedSegmentSumOp<float, int, false>);
REGISTER_CUDA_OPERATOR(
    UnsortedSegmentMean,
    CUDAUnsortedSegmentSumOp<float, int, true>);

} // namespace caffe2
