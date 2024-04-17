#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh> //for MAX_DIMS
#include <ATen/cuda/cub.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#include <ATen/ops/nonzero_native.h>
#endif


namespace at::native {

namespace{
template<typename T>
struct NonZeroOp
{
    __host__ __device__ __forceinline__ bool operator()(const T& a) const {
      return (a!=T(0));
    }
};

//TODO: actually support int64_t index_t
template<typename index_t>
struct TensorDims {
  index_t sizes[MAX_DIMS];
};

template <typename index_t>
__global__ void write_indices(
    int64_t* inp,
    TensorDims<index_t> dims,
    int ndim,
    index_t n) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
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
  }
}

} //anonymous namespace

template<typename scalar_t>
void nonzero_cuda_out_impl(const Tensor& self, Tensor& out){
  Tensor self_ = self.contiguous();
  int N = self_.numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
// compute number of nonzero elements
  size_t temp_storage_bytes=0;
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto num_nonzeros = allocator.allocate(sizeof(int));
  cub::TransformInputIterator<bool, NonZeroOp<scalar_t>, const scalar_t*> itr(self_.const_data_ptr<scalar_t>(), NonZeroOp<scalar_t>());
  cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, itr, (int*)num_nonzeros.get(), N, stream);
  auto temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DeviceReduce::Sum(temp_storage.get(), temp_storage_bytes, itr, (int*)num_nonzeros.get(), N, stream);
  int num_nonzeros_h;
  at::cuda::memcpy_and_sync(&num_nonzeros_h, num_nonzeros.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
  //expected output size is num_nonzeros x ndim
  //we are producing output with size {num_nonzeros, ndim} and strides {1, num_nonzeros} (that is, transposed ndim x num_nonzeros output)
  //we are able to directly use passed output with this size and strides, and we can also (per contract)
  //resize passed output with incorrect sizes anyway we want.
  //However, out with correct sizes and incorrect strides will have to be copied to from the intermediate we've produced.
  bool need_to_copy = out.dim() == 2 && out.sizes()[0] == num_nonzeros_h && out.sizes()[1] == self.dim() && !out.t().is_contiguous();
  at::Tensor out_temp = need_to_copy ?
      Tensor(at::detail::empty_cuda({self.dim(), num_nonzeros_h}, out.options())) :
      out.resize_({self.dim(), num_nonzeros_h});
  //Scalars are expected to produce output of size (1,0), so we can't write to it
  if (self.dim() > 0) {
    cub::CountingInputIterator<int64_t> counting_itr(0);
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_storage_bytes, counting_itr, itr,
      out_temp.mutable_data_ptr<int64_t>(), (int*)num_nonzeros.get(), N, stream);
    temp_storage = allocator.allocate(temp_storage_bytes);
    cub::DeviceSelect::Flagged(temp_storage.get(), temp_storage_bytes, counting_itr, itr,
      out_temp.mutable_data_ptr<int64_t>(), (int*)num_nonzeros.get(), N, stream);
    if (num_nonzeros_h > 0 && self.dim() > 1){
        TensorDims<int> dims;
        for (int i=0; i<self.dim(); i++){
            dims.sizes[i] = self.sizes()[i];
        }
        const int nthreads = 256;
        const int nblocks = (num_nonzeros_h + nthreads -1)/nthreads;
        write_indices<<<nblocks, nthreads, 0, stream>>>(out_temp.mutable_data_ptr<int64_t>(),
        dims, self.dim(), num_nonzeros_h);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  if (need_to_copy) {
    out.copy_(out_temp.t());
  } else {
    //transpose out so it is correct size
    Tensor out_ = out_temp.t();
    out.set_(out_);
  }
}

Tensor& nonzero_out_cuda(const Tensor& self, Tensor& out){
  TORCH_CHECK(self.numel() < std::numeric_limits<int>::max(), "nonzero is not supported for tensors with more than INT_MAX elements, \
  See https://github.com/pytorch/pytorch/issues/51871");
  TORCH_CHECK(out.dtype() == at::kLong, "Expected object of scalar type ", at::kLong, " as out, but got ", out.dtype());
  TORCH_CHECK(self.device() == out.device(), "expected self and out to be on the same device, but got out on ",
  out.device(), " and self on ", self.device());
  TORCH_CHECK(self.dim() <= MAX_DIMS, "nonzero is not supported for tensor with more than ", MAX_DIMS, " dimensions");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(at::ScalarType::ComplexHalf, at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half,
    self.scalar_type(), "nonzero_cuda",
    [&] {nonzero_cuda_out_impl<scalar_t>(self, out);});
  return out;
}

Tensor nonzero_cuda(const Tensor& self){
  Tensor out = at::detail::empty_cuda({0}, self.options().dtype(kLong));
  return at::native::nonzero_out_cuda(self, out);
}
} //namespace at::native
