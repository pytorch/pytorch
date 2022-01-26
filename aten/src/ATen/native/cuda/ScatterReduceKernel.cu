#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include "ScatterReduceReducer.cuh"

// TODO: how to define?
#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

namespace at { namespace native {

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const scalar_t *src_data,
               const at::cuda::detail::TensorInfo<int64_t, int> index_info,
               scalar_t *out_data, int E, int K, int N, int numel) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int b = thread_idx / (E * K);
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    Reducer<scalar_t, REDUCE>::atomic_write(out_data + b * N * K + idx * K + k,
                                            src_data[thread_idx]);
  }
}

template <typename scalar_t>
__global__ void
scatter_arg_kernel(const scalar_t *src_data,
                   const at::cuda::detail::TensorInfo<int64_t, int> index_info,
                   const scalar_t *out_data, int64_t *arg_out_data, int E,
                   int K, int N, int numel) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int b = thread_idx / (E * K);
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    if (src_data[thread_idx] == out_data[b * N * K + idx * K + k]) {
      arg_out_data[b * N * K + idx * K + k] = e;
    }
  }
}


Tensor scatter_reduce_two_cuda(const Tensor& self,
                               int64_t dim,
                               const Tensor& index,
                               const c10::string_view reduce,
                               const c10::optional<int64_t> output_size) {


  TORCH_CHECK(self.dim() == index.dim(),
      "Shape mismatch between `self` (got ", self.sizes(), ") and `index` (got ", index.sizes(), ")");
  for (int64_t i = 0; i < self.dim(); i++) {
    TORCH_CHECK(self.size(i) == index.size(i),
        "Shape mismatch between `self` (got ", self.sizes(), ") and `index` (got ", index.sizes(), ")");
  }

  auto self_cont = self.contiguous();
  auto index_cont = index.contiguous();

  auto sizes = self.sizes().vec();
  if (output_size.has_value())
    sizes[dim] = output_size.value();
  else if (index.numel() == 0)
    sizes[dim] = 0;
  else {
    auto d_size = index.max().data_ptr<int64_t>();
    auto h_size = (int64_t *)malloc(sizeof(int64_t));
    cudaMemcpy(h_size, d_size, sizeof(int64_t), cudaMemcpyDeviceToHost);
    sizes[dim] = 1 + *h_size;
  }

  Tensor out = at::empty(sizes, self.options());

  if (self.numel() == 0) {
    return out.zero_();
  }

  auto B = 1;
  for (auto i = 0; i < dim; i++)
    B *= self.size(i);
  auto E = self.size(dim);
  auto K = self.numel() / (B * E);
  auto N = out.size(dim);

  auto index_info = at::cuda::detail::getTensorInfo<int64_t, int>(index);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "scatter_reduce", [&] {
    auto self_data = self_cont.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      out.fill_(Reducer<scalar_t, REDUCE>::init());

      scatter_kernel<scalar_t, REDUCE>
          <<<BLOCKS(self.numel()), THREADS, 0, stream>>>(
              self_data, index_info, out_data, E, K, N, self.numel());

      if (REDUCE == MIN || REDUCE == MAX)
        out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);
    });
  });

  return out;
}


}} // namespace at::native
