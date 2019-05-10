#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/gather_op.h"

namespace caffe2 {

// This maintains kernels and index-mapping functions shared
//  by Gather and BatchGather ops.
namespace gather_helper {

template <typename T_INDEX, typename TData>
__global__ void BatchGatherKernel(
    const TData* src_base,
    TData* out,
    const T_INDEX* indices,
    const int M,
    const int N,
    const int data_batch_size,
    const int gathered_batch_size,
    const int block_size,
    const int indexing_axis_dim,
    const bool wrap_indices) {
  const int begin_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int num_items = M * N * block_size;
  for (int s = begin_idx; s < num_items; s += blockDim.x * gridDim.x) {
    const int k = s % block_size;
    const int j = s / block_size % N;
    const int i = s / block_size / N;
    T_INDEX idx = indices[j];
    if (wrap_indices && (idx < 0)) {
        idx = idx + (T_INDEX) indexing_axis_dim;
    }
    const float* src_offset = src_base + i * data_batch_size + idx * block_size;
    float* dst_offset = out + i * gathered_batch_size + j * block_size;
    dst_offset[k] = src_offset[k];
  }
}

// Actual gather implementation - resizes output and copies indexed data.
template <typename Index>
static bool gather_impl_cuda(
    Operator<CUDAContext>* op,
    int dataIdx,
    int indicesIdx,
    int outputIdx,
    int axis,
    bool wrap_indices) {
  const Tensor& data = op->Input(dataIdx);
  const Tensor& indices = op->Input(indicesIdx);
  const TypeMeta dataType = data.dtype();
  size_t item_bytesize = dataType.itemsize();

  // ONNX allows negative axis to index from the back, valid range: [-r, r].
  if (axis < 0) {
    axis = data.dim() + axis;
  }
  CAFFE_ENFORCE_GE(
      data.dim(), axis + 1, "DATA should be at least [axis+1]-D");
  CAFFE_ENFORCE_GE(axis, 0, "Axis should be non-negative");
  CAFFE_ENFORCE_LT(axis, data.dim(), "Axis out of range");

  // New shape:
  //  [data dims before axis] + [indices dims] + [data dims after axis]
  vector<int64_t> shape =
      calc_output_shape_vector<int64_t>(data.sizes(), indices.sizes(), axis);
  Tensor* output = op->Output(outputIdx, shape, at::dtype(dataType));
  float* out = static_cast<float*>(output->raw_mutable_data(dataType));

  // Succeed if size of output is zero, which can happen for empty batch which
  // would have data dimension size of 0.
  // This *must* be done AFTER output->raw_mutable_data() above as that has
  // important allocation side effect that we must see.
  if (output->numel() == 0) {
    return true;
  }

  const Index* idxs = indices.template data<Index>();
  const float* src_base = static_cast<const float*>(data.raw_data());

  const int outer_dims_product = data.size_to_dim(axis);
  const int block_size = data.size_from_dim(axis + 1);

  const int src_indexing_axis_dim = data.size(axis);
  // Treat indices as a single block even if they have multiple dimensions.
  // The "gathered batch" is a cumulative result combining indexed blocks.
  const int N = indices.numel();
  auto gathered_batch_size = N * block_size;
  const auto src_batch_size = data.size_from_dim(axis);

  // Only run kernel if input is not empty.
  if (N > 0) {
    BatchGatherKernel<<<
        std::min(outer_dims_product, CAFFE_MAXIMUM_NUM_BLOCKS),
        std::min(N * block_size, CAFFE_CUDA_NUM_THREADS),
        0,
        op->getContext()->cuda_stream()>>>(
        src_base,
        out,
        idxs,
        outer_dims_product,
        N,
        src_batch_size,
        gathered_batch_size,
        block_size,
        src_indexing_axis_dim,
        wrap_indices);
  }
  return true;
}

} // namespace gather_helper
} // namespace caffe2
