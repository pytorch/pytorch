#include "caffe2/operators/sparse_to_dense_op.h"

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/GpuAtomics.cuh"

namespace caffe2 {

  template <typename TInd, typename TData>
  __global__ void SparseToDenseKernel(
    size_t N, int64_t block_nitems, const TInd* indices, const TData* vals, TData* dst) {
    CUDA_1D_KERNEL_LOOP(i, N) {
      int idx = indices[i / block_nitems];
      int dst_idx = block_nitems * idx + i % block_nitems;
      gpu_atomic_add(&dst[dst_idx], vals[i]);
    }
  }

  template <>
  bool SparseToDenseOp<CUDAContext>::RunOnDevice() {
    return DispatchHelper<TensorTypes<int32_t>>::call(
        this, Input(INDICES));
  }

  template <>
  template <typename TInd>
  bool SparseToDenseOp<CUDAContext>::DoRunWithType() {
    return DispatchHelper<
        TensorTypes2<
            float,
            int32_t>,
        TInd>::call(this, Input(VALUES));
  }

  template <>
  template <typename TInd, typename TData>
  bool SparseToDenseOp<CUDAContext>::DoRunWithType2() {
    auto& sparse_indices = Input(INDICES);
    CAFFE_ENFORCE_EQ(sparse_indices.dim(), 1);
    auto& sparse_values = Input(VALUES);
    CAFFE_ENFORCE_GE(sparse_values.dim(), 1);
    CAFFE_ENFORCE_EQ(sparse_indices.numel(), sparse_values.dim(0));

    const TInd* sparse_indices_vec = sparse_indices.template data<TInd>();
    const int32_t sparse_indices_len = sparse_indices.dim32(0);
    const int output_first_dim =
        GetOutputFirstDim(sparse_indices_vec, sparse_indices_len);

    auto shape = sparse_values.sizes().vec();
    shape[0] = output_first_dim;

    auto* output = Output(0, shape, at::dtype<TData>());

    TData* output_data = output->template mutable_data<TData>();
    math::Set<TData>(output->numel(), TData(0), output_data, &context_);

    const auto block_nitems = sparse_values.size_from_dim(1);
    const TData* sparse_values_vec = sparse_values.template data<TData>();

    size_t N = block_nitems * sparse_indices_len;
    CAFFE_ENFORCE_EQ(output->numel(), output_first_dim * block_nitems);
    SparseToDenseKernel<TInd, TData><<<
      CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
      context_.cuda_stream()>>>(
        N,
        block_nitems,
        sparse_indices_vec,
        sparse_values_vec,
        output_data
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return true;
  }


REGISTER_CUDA_OPERATOR(SparseToDense, SparseToDenseOp<CUDAContext>);

} // namespace caffe2
