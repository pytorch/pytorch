#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/reverse_packed_segs_op.h"

namespace caffe2 {

namespace {

template <typename T, typename LengthType>
__global__
void ReversePackedSegments_kernel(
      size_t max_length,
      size_t batch_size,
      size_t block_size,
      const LengthType* lengths_ptr,
      const T* data_ptr,
      T* rev_data_ptr) {

  const int block_id = blockIdx.x;

  // index into [0, batch_size)
  const int batch = block_id / max_length;
  // index into [0, segment)
  const int segment = block_id % max_length;

  if (batch >= batch_size || segment >= max_length) return;

  const int seg_length = lengths_ptr[batch];

  // unique data pointer for this CTA
  const T* local_data_ptr = data_ptr + (segment * batch_size + batch) * block_size;

  // unique pointer for result
  T* local_rev_data_ptr;
  if (segment < seg_length) {
    local_rev_data_ptr = rev_data_ptr + ((seg_length - 1 - segment) * batch_size + batch) * block_size;
  } else {
    local_rev_data_ptr = rev_data_ptr + (segment * batch_size + batch) * block_size;
  }

  // copy using 1 element / thread for now
  for (int idx = threadIdx.x; idx < block_size; idx+=blockDim.x) {
    local_rev_data_ptr[idx] = local_data_ptr[idx];
  }
}

} // namespace

// specialization of DoRunWithLengthType
template <>
template <typename T, typename LengthType>
void ReversePackedSegsOp<CUDAContext>::DoRunWithLengthType() {
  const auto& data = Input(DATA);
  const auto& lengths = Input(LENGTHS);

  CAFFE_ENFORCE(
      data.dim() == 3,
      "DATA should be 3-D tensor <lengths, "
      "segments, embeddings>");
  CAFFE_ENFORCE(lengths.dim() == 1, "LENGTH should be 1-D");

  auto* output = Output(0, data.sizes(), at::dtype<T>());

  const auto max_length = data.size(0);
  const auto batch_size = data.size(1);
  const auto block_size = data.size(2);
  CAFFE_ENFORCE(
      lengths.sizes()[0] == batch_size,
      "lenths size should be"
      " equal to batch size");

  const T* data_ptr = data.template data<T>();
  const LengthType* lengths_ptr = lengths.template data<LengthType>();

  // reversed data
  T* rev_data_ptr = output->template mutable_data<T>();

  const int grid = max_length * batch_size;

  ReversePackedSegments_kernel<T,LengthType><<<grid, 512, 0, context_.cuda_stream()>>>(
        max_length,
        batch_size,
        block_size,
        lengths_ptr,
        data_ptr,
        rev_data_ptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

REGISTER_CUDA_OPERATOR(ReversePackedSegs, ReversePackedSegsOp<CUDAContext>);
} // namespace caffe2
