#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh> //for MAX_DIMS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/nonzero_native.h>
#endif

namespace at::native {

namespace {
template <typename T>
struct NonZeroOp {
  __host__ __device__ __forceinline__ bool operator()(const T& a) const {
    return (a != T(0));
  }
};

// TODO: actually support int64_t index_t
template <typename index_t>
struct TensorDims {
  index_t sizes[MAX_DIMS];
};

template <typename index_t>
__global__ void write_indices(
    int64_t* inp,
    TensorDims<index_t> dims,
    int ndim,
    index_t n,
    int64_t * total = nullptr,
    int64_t fill_value = -1) {
  auto index = threadIdx.x + (int64_t)blockIdx.x * blockDim.x;
  bool cond = (total == nullptr || index < *total);
  if (index < n && cond) {
    index_t div = 1;
    int64_t idx_flat = inp[index];
#pragma unroll
    for (int dim = MAX_DIMS; dim >= 0; dim--) {
      if (dim > ndim - 1)
        continue;
      auto dim_size = dims.sizes[dim];
      inp[index + dim * n] = (idx_flat / div) % dim_size;
      div *= dim_size;
    }
  } else if (index < n) {
    // 0th dim has correct values already
    for (int dim = ndim - 1; dim > 0; dim--) {
      inp[index + dim * n] = fill_value;
    }
  }
}

__global__ void write_fill_value(int64_t * inp, int64_t * total, int64_t fill_value, int64_t n){
  int64_t total_val = *total;
  // not aiming for vectorized stores

  for (int64_t idx = total_val + (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
      inp[idx] = fill_value;
  }
}

template <int BLOCK_THREADS>
__global__ void compute_agg(int32_t * agg, int64_t * agg_cum, uint32_t n_blocks) {

  using BlockScanT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<int64_t, BLOCK_THREADS, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  int agg_data;
  int64_t agg_cum_data;
  agg_data = threadIdx.x < n_blocks ? agg[threadIdx.x] : 0;
  BlockScanT(temp_storage).InclusiveSum(agg_data, agg_cum_data);
  if (threadIdx.x < n_blocks) {
    agg_cum[threadIdx.x] = agg_cum_data;
  }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T>
__global__ void flag_kernel(const T* d_in, int64_t * d_out, const int64_t * agg, int64_t input_nelem, int64_t output_nelem, int iters_per_cta) {
  int64_t start_idx = BLOCK_THREADS * ITEMS_PER_THREAD * iters_per_cta * (int64_t)blockIdx.x;
  if (start_idx >= input_nelem) return;
  d_in += start_idx;

  using BlockLoadT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_LOAD_WARP_TRANSPOSE>;

  // Specialize BlockScan type for our thread block
  using BlockScanT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<int, BLOCK_THREADS, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_SCAN_WARP_SCANS>;
  using TransformInputIteratorT = ROCM_HIPCUB(at_cuda_detail::cub)::TransformInputIterator<int, NonZeroOp<T>, const T*>;
  using BlockExchangeT =  ROCM_HIPCUB(at_cuda_detail::cub)::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD>;

  // Shared memory
  __shared__ union TempStorage
  {
    typename BlockLoadT::TempStorage load;
    typename BlockScanT::TempStorage scan;
    typename BlockExchangeT::TempStorage exchange;
  } temp_storage;

  int64_t aggregate = blockIdx.x == 0 ? 0 : agg[blockIdx.x - 1];
  d_out += aggregate;

  TransformInputIteratorT t_input_itr(d_in, NonZeroOp<T>());

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];
  int out_indices[ITEMS_PER_THREAD];

  int64_t remaining =  input_nelem - start_idx;
  int64_t out_remaining = output_nelem - aggregate;
  for (int i=0; i<iters_per_cta; i++){

  // Load items into a blocked arrangement
    if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
      BlockLoadT(temp_storage.load).Load(t_input_itr, data);
    } else {
      BlockLoadT(temp_storage.load).Load(t_input_itr, data, remaining, int(0));
    }

    // Barrier for smem reuse
    __syncthreads();

    // Compute inclusive prefix sum
    int aggregate;
    __shared__ int aggregate_sh;
    BlockScanT(temp_storage.scan).ExclusiveSum(data, out_indices, aggregate);

    if (threadIdx.x == 0){
      aggregate_sh = aggregate;
    }

    // Barrier for smem reuse
    __syncthreads();
    // striped arrangement will provide a slightly better
    // coalescing for writes (although it's still bad because it's indirect indexing)
    BlockExchangeT(temp_storage.exchange).BlockedToStriped(data);
    __syncthreads();
    BlockExchangeT(temp_storage.exchange).BlockedToStriped(out_indices);
    for (int ii=0; ii<ITEMS_PER_THREAD; ii++){
      if (data[ii] != 0 && out_indices[ii] < out_remaining) {
        int64_t inp_idx = start_idx + threadIdx.x + blockDim.x * ii;
        d_out[out_indices[ii]] = inp_idx;
      }
    }

    out_remaining -= aggregate_sh;
    remaining -= BLOCK_THREADS * ITEMS_PER_THREAD;
    if (remaining <= 0 || out_remaining <= 0) return;
    d_out += aggregate_sh;
    t_input_itr += BLOCK_THREADS * ITEMS_PER_THREAD;
    start_idx += BLOCK_THREADS * ITEMS_PER_THREAD;
    __syncthreads();
  }

}



} // anonymous namespace

template <typename scalar_t>
void nonzero_cuda_out_impl(const Tensor& self, Tensor& out) {
  Tensor self_ = self.contiguous();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t chunk_size, num_chunks;
  if (self.numel() < std::numeric_limits<int>::max()) {
    chunk_size = self.numel();
    num_chunks = 1;
  } else {
    chunk_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
    num_chunks = (self.numel() + chunk_size - 1) / chunk_size;
  }
  // compute number of nonzero elements
  size_t temp_storage_bytes = 0;
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto num_nonzeros = allocator.allocate(sizeof(int) * num_chunks);
  for (int64_t idx = 0; idx < num_chunks; idx++) {
    int64_t remaining = std::min(chunk_size, self.numel() - idx * chunk_size);
    cub::TransformInputIterator<bool, NonZeroOp<scalar_t>, const scalar_t*> itr(
        self_.const_data_ptr<scalar_t>() + idx * chunk_size,
        NonZeroOp<scalar_t>());
    cub::DeviceReduce::Sum(
        nullptr,
        temp_storage_bytes,
        itr,
        ((int*)num_nonzeros.get()) + idx,
        remaining,
        stream);
    auto temp_storage = allocator.allocate(temp_storage_bytes);
    cub::DeviceReduce::Sum(
        temp_storage.get(),
        temp_storage_bytes,
        itr,
        ((int*)num_nonzeros.get()) + idx,
        remaining,
        stream);
  }
  auto pinned_num_nonzeros_h = at::detail::empty_cpu(
      {num_chunks}, /* size */
      c10::CppTypeToScalarType<int>(), /* dtype */
      std::nullopt, /* layout */
      std::nullopt, /* device */
      true, /* pin_memory */
      std::nullopt /* memory format */
  );
  at::cuda::memcpy_and_sync(
      (void*)pinned_num_nonzeros_h.const_data_ptr<int>(),
      num_nonzeros.get(),
      sizeof(int) * num_chunks,
      cudaMemcpyDeviceToHost,
      stream);
  int64_t num_nonzeros_h = 0;

  for (int64_t idx = 0; idx < num_chunks; idx++) {
    num_nonzeros_h += (int)*(pinned_num_nonzeros_h.const_data_ptr<int>() + idx);
  }
  // num_nonzeros_h = (int)*(pinned_num_nonzeros_h.const_data_ptr<int>());
  // expected output size is num_nonzeros x ndim
  // we are producing output with size {num_nonzeros, ndim} and strides {1,
  // num_nonzeros} (that is, transposed ndim x num_nonzeros output) we are able
  // to directly use passed output with this size and strides, and we can also
  // (per contract) resize passed output with incorrect sizes anyway we want.
  // However, out with correct sizes and incorrect strides will have to be
  // copied to from the intermediate we've produced.
  bool need_to_copy = out.dim() == 2 && out.sizes()[0] == num_nonzeros_h &&
      out.sizes()[1] == self.dim() && !out.t().is_contiguous();
  at::Tensor out_temp = need_to_copy
      ? Tensor(
            at::detail::empty_cuda({self.dim(), num_nonzeros_h}, out.options()))
      : out.resize_({self.dim(), num_nonzeros_h});
  // Scalars are expected to produce output of size (1,0), so we can't write to
  // it
  int64_t curr_nonzeros = 0;
  if (self.dim() > 0) {
    for (int64_t idx = 0; idx < num_chunks; idx++) {
      int remaining = std::min(chunk_size, self.numel() - idx * chunk_size);

      cub::CountingInputIterator<int64_t> counting_itr(idx * chunk_size);
      cub::TransformInputIterator<bool, NonZeroOp<scalar_t>, const scalar_t*>
          itr(self_.const_data_ptr<scalar_t>() + idx * chunk_size,
              NonZeroOp<scalar_t>());
      temp_storage_bytes = 0;
      cub::DeviceSelect::Flagged(
          nullptr,
          temp_storage_bytes,
          counting_itr,
          itr,
          out_temp.mutable_data_ptr<int64_t>(),
          ((int*)num_nonzeros.get()) + idx,
          remaining,
          stream);
      auto temp_storage = allocator.allocate(temp_storage_bytes);
      cub::DeviceSelect::Flagged(
          temp_storage.get(),
          temp_storage_bytes,
          counting_itr,
          itr,
          out_temp.mutable_data_ptr<int64_t>() + curr_nonzeros,
          ((int*)num_nonzeros.get()) + idx,
          remaining,
          stream);
      curr_nonzeros +=
          (int)*(pinned_num_nonzeros_h.const_data_ptr<int>() + idx);
    }
    if (num_nonzeros_h > 0 && self.dim() > 1) {
      TensorDims<int64_t> dims;
      for (int i = 0; i < self.dim(); i++) {
        dims.sizes[i] = self.sizes()[i];
      }
      const int nthreads = 256;
      const int nblocks = (num_nonzeros_h + nthreads - 1) / nthreads;
      write_indices<<<nblocks, nthreads, 0, stream>>>(
          out_temp.mutable_data_ptr<int64_t>(),
          dims,
          self.dim(),
          num_nonzeros_h);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  if (need_to_copy) {
    out.copy_(out_temp.t());
  } else {
    // transpose out so it is correct size
    Tensor out_ = out_temp.t();
    out.set_(out_);
  }
}

template <typename scalar_t>
void nonzero_static_cuda_out_impl(
    const Tensor& self,
    int64_t size,
    int64_t fill_value,
    Tensor& out) {
# if (defined(CUDA_VERSION) && CUDA_VERSION > 11040) || defined(USE_ROCM)

  Tensor self_contiguous_ = self.contiguous();
  // see comment in nonzero_cuda_out_impl on reqs for out
  bool out_correct_size =
      out.dim() == 2 && out.sizes()[0] == size && out.sizes()[1] == self.dim();
  bool need_to_copy = out_correct_size && !out.t().is_contiguous();
  if (!out_correct_size) {
    out.resize_({self.dim(), size}).t();
  }
  if (out.numel() == 0) return;
  // we need to allocate temporary out to then copy to user provided out
  at::Tensor out_temp;
  if (need_to_copy) {
    out_temp =
        Tensor(at::detail::empty_cuda({self.dim(), size}, out.options())).t();
  }
  int64_t* out_data_ptr = need_to_copy ? out_temp.mutable_data_ptr<int64_t>()
                                       : out.mutable_data_ptr<int64_t>();

  const scalar_t * in_data_ptr = self_contiguous_.const_data_ptr<scalar_t>();
  constexpr int BLOCK_THREADS = 512; //block_threads<sizeof(scalar_t)>();
  constexpr int ITEMS_PER_THREAD = 16;
  auto grid_size = (self.numel() + BLOCK_THREADS * ITEMS_PER_THREAD - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);
  const int64_t num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t target_blocks = sizeof(scalar_t) == 1 ? 2 * num_sms : num_sms;
  const int iters_per_cta = (grid_size + target_blocks - 1)/target_blocks;
  grid_size = (self.numel() + iters_per_cta * BLOCK_THREADS * ITEMS_PER_THREAD - 1) / (iters_per_cta * BLOCK_THREADS * ITEMS_PER_THREAD);
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto agg = allocator.allocate(grid_size * sizeof(int));
  at::cuda::cub::calc_block_sums<BLOCK_THREADS, ITEMS_PER_THREAD, true>
  <<<grid_size, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
    in_data_ptr, (int*)agg.get(), self.numel(), iters_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto agg_cum = allocator.allocate(grid_size * sizeof(int64_t));
  // computing partial sums in int64 in the flag kernel
  // leads to 20-30% slowdown, so compute them in a separate 2 us kernel
  compute_agg<BLOCK_THREADS><<<1, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
   (int*)agg.get(), (int64_t*)agg_cum.get(), grid_size
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  flag_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
  <<<grid_size, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
    in_data_ptr, out_data_ptr, (int64_t*)agg_cum.get(), self.numel(), size, iters_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int64_t out_grid = std::min(num_sms, (size + BLOCK_THREADS - 1)/BLOCK_THREADS);
  write_fill_value<<<out_grid, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(out_data_ptr, (int64_t *)agg_cum.get() + grid_size - 1, fill_value, size);
  if (self.dim() > 1) {
    TensorDims<int64_t> dims;
    for (int i = 0; i < self.dim(); i++) {
      dims.sizes[i] = self.sizes()[i];
    }
    const int nthreads = 256;
    const int nblocks = (size + nthreads - 1) / nthreads;
    write_indices<<<nblocks, nthreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        out_data_ptr,
        dims,
        self.dim(),
        size,
        (int64_t *)agg_cum.get() + grid_size - 1,
        fill_value);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (need_to_copy) {
    out.copy_(out_temp);
  }
#else
  TORCH_CHECK(false, "Nonzero_static is not supported for cuda <= 11.4");
#endif
}

Tensor& nonzero_out_cuda(const Tensor& self, Tensor& out) {
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "Expected object of scalar type ",
      at::kLong,
      " as out, but got ",
      out.dtype());
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= MAX_DIMS,
      "nonzero is not supported for tensor with more than ",
      MAX_DIMS,
      " dimensions");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "nonzero_cuda",
      [&] { nonzero_cuda_out_impl<scalar_t>(self, out); });
  return out;
}

Tensor nonzero_cuda(const Tensor& self) {
  Tensor out = at::detail::empty_cuda({0}, self.options().dtype(kLong));
  return at::native::nonzero_out_cuda(self, out);
}

Tensor& nonzero_static_out_cuda(
    const Tensor& self,
    int64_t size,
    int64_t fill_value,
    Tensor& out) {
  TORCH_CHECK(
      out.dtype() == at::kLong,
      "nonzero_static: Expected out tensor to have scalar type ",
      at::kLong,
      " but got ",
      out.dtype());
  TORCH_CHECK(
      self.device() == out.device(),
      "expected self and out to be on the same device, but got out on ",
      out.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      self.dim() <= MAX_DIMS,
      "nonzero_static is not supported for tensor with more than ",
      MAX_DIMS,
      " dimensions");
  TORCH_CHECK(
      size >= 0, "nonzero_static: 'size' must be an non-negative integer"
  )
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      at::ScalarType::ComplexHalf,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      self.scalar_type(),
      "nonzero_cuda",
      [&] {
        nonzero_static_cuda_out_impl<scalar_t>(self, size, fill_value, out);
      });
  return out;
}

Tensor nonzero_static_cuda(
    const Tensor& self,
    int64_t size,
    int64_t fill_value) {
  TORCH_CHECK(
      size >= 0, "nonzero_static: 'size' must be an non-negative integer"
  )
  Tensor out = Tensor(at::detail::empty_cuda(
                          {self.dim(), size}, self.options().dtype(kLong)))
                   .t();
  return at::native::nonzero_static_out_cuda(self, size, fill_value, out);
}

} // namespace at::native
