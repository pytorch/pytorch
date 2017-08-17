#include "caffe2/operators/layer_norm_op.h"

#include <cub/cub.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {
template <typename T>
struct SqrTransform {
  inline __host__ __device__ T operator()(const T v) const {
    return v * v;
  }
};

// X = X - Y^2
__global__ void
sqrtXMinusYSquaredKernel(const int N, float* x, const float* y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    x[i] = sqrtf(x[i] - y[i] * y[i]);
  }
}

// out[i, j] = (X[i, j] - mu[i]) / sigma[i]
__global__ void normalizeKernel(
    const int row_dim,
    const int N,
    const float epsilon,
    const float* x,
    const float* mu,
    const float* sigma,
    float* out) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    out[i] = (x[i] - mu[i / row_dim]) / (sigma[i / row_dim] + epsilon);
  }
}

} //  namespace

template <>
template <>
bool LayerNormOp<CUDAContext>::DoRunWithType<float>() {
  const auto& input = Input(0);
  auto* output = Output(0);
  auto* mean = Output(1);
  auto* stdev = Output(2);

  CAFFE_ENFORCE_GE(input.dims().size(), 2, "LayerNorm requires input dim >= 2");

  const auto canonical_axis = input.canonical_axis_index(axis_);
  const int left = input.size_to_dim(canonical_axis);
  const int right = input.size_from_dim(canonical_axis);

  output->ResizeLike(input);
  std::vector<TIndex> stats_dims(
      input.dims().begin(), input.dims().begin() + canonical_axis);
  stats_dims.push_back(1);
  mean->Resize(stats_dims);
  stdev->Resize(stats_dims);

  std::vector<int> segs(left + 1);
  std::iota(segs.begin(), segs.end(), 0);
  std::transform(
      segs.begin(),
      segs.end(),
      segs.begin(),
      std::bind1st(std::multiplies<int>(), right));

  seg_indices_.Resize(vector<size_t>{segs.size()});
  context_.CopyBytes<CPUContext, CUDAContext>(
      sizeof(int) * segs.size(),
      static_cast<void*>(segs.data()),
      static_cast<void*>(seg_indices_.mutable_data<int>()));

  if (right == 1) {
    mean->CopyFrom(input);
    mean->Resize(stats_dims);
    math::Set<float, CUDAContext>(
        left, 0.0f, stdev->mutable_data<float>(), &context_);
  } else {
    // Calculate row-wise means
    // First stage: sum up feature vectors
    size_t temp_storage_bytes;
    cub::DeviceSegmentedReduce::Sum(
        NULL, // To retrieve required temporary storage size
        temp_storage_bytes, // size_t &temp_storage_bytes
        input.data<float>(), // InputIteratorT d_i
        mean->mutable_data<float>(), // OutputIteratorT d_out
        left, // int num_segments
        seg_indices_.mutable_data<int>(), // int *d_begin_offsets
        seg_indices_.mutable_data<int>() + 1, // int *d_end_offsets
        context_.cuda_stream() // cudaStream_t stream=0
        );
    size_t temp_storage_floats = temp_storage_bytes / sizeof(float) +
        (temp_storage_bytes % sizeof(float) ? 1 : 0);
    scratch_.Resize(vector<size_t>{temp_storage_floats});

    cub::DeviceSegmentedReduce::Sum(
        scratch_.mutable_data<float>(), // void *d_temp_storage
        temp_storage_bytes, // size_t &temp_storage_bytes
        input.data<float>(), // InputIteratorT d_i
        mean->mutable_data<float>(), // OutputIteratorT d_out
        left, // int num_segments
        seg_indices_.mutable_data<int>(), // int *d_begin_offsets
        seg_indices_.mutable_data<int>() + 1, // int *d_end_offsets
        context_.cuda_stream() // cudaStream_t stream=0
        );

    // Second stage: Normalize by feature vector dim
    math::Scale<float, CUDAContext>(
        left,
        1.0f / right,
        mean->mutable_data<float>(),
        mean->mutable_data<float>(),
        &context_);

    // Calculate row-wise standard deviation

    // First stage: sum up row-wise squared values
    SqrTransform<float> transform;
    cub::TransformInputIterator<float, SqrTransform<float>, const float*> it(
        input.data<float>(), transform);
    cub::DeviceSegmentedReduce::Sum(
        NULL, // To retrieve required temporary storage size
        temp_storage_bytes, // size_t &temp_storage_bytes
        it, // InputIteratorT d_i
        stdev->mutable_data<float>(), // OutputIteratorT d_out
        left, // int num_segments
        seg_indices_.mutable_data<int>(), // int *d_begin_offsets
        seg_indices_.mutable_data<int>() + 1, // int *d_end_offsets
        context_.cuda_stream() // cudaStream_t stream=0
        );
    temp_storage_floats = temp_storage_bytes / sizeof(float) +
        (temp_storage_bytes % sizeof(float) ? 1 : 0);
    scratch_.Resize(vector<size_t>{temp_storage_floats});

    cub::DeviceSegmentedReduce::Sum(
        scratch_.mutable_data<float>(), // void *d_temp_storage
        temp_storage_bytes, // size_t &temp_storage_bytes
        it, // InputIteratorT d_i
        stdev->mutable_data<float>(), // OutputIteratorT d_out
        left, // int num_segments
        seg_indices_.mutable_data<int>(), // int *d_begin_offsets
        seg_indices_.mutable_data<int>() + 1, // int *d_end_offsets
        context_.cuda_stream() // cudaStream_t stream=0
        );

    // Second stage: Normalize by feature vector dim
    math::Scale<float, CUDAContext>(
        left,
        1.0f / right,
        stdev->mutable_data<float>(),
        stdev->mutable_data<float>(),
        &context_);

    // stddev = sqrt(E(x^2) - E(x)^2)
    sqrtXMinusYSquaredKernel<<<
        CAFFE_GET_BLOCKS(left),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        left, stdev->mutable_data<float>(), mean->mutable_data<float>());
  }

  // out[i, j] = (in[i,j] - mu[i]) / (sigma[i] + epsilon)
  normalizeKernel<<<
      CAFFE_GET_BLOCKS(left),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      right,
      left * right,
      epsilon_,
      input.data<float>(),
      mean->data<float>(),
      stdev->data<float>(),
      output->mutable_data<float>());

  return true;
}

REGISTER_CUDA_OPERATOR(LayerNorm, LayerNormOp<CUDAContext>);

} // namespace caffe2
