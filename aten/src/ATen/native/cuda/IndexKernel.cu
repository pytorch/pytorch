#include <ATen/native/TensorAdvancedIndexing.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/core/Array.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <THC/THCTensorInfo.cuh>

namespace at { namespace native {

static constexpr int launch_bound2 = 4;

static constexpr int launch_size_nd = 128;

template <int Dims, typename T, typename IndexType>
__device__ __forceinline__ IndexType indexToOffset(
    const cuda::detail::TensorInfo<T, IndexType>& info,
    int64_t index,
    IndexType size) {
  IndexType linearIndex = static_cast<IndexType>(index);
  CUDA_KERNEL_ASSERT(linearIndex < size && linearIndex >= -size);
  if (linearIndex < 0) {
    linearIndex += size;
  }
  return cuda::detail::IndexToOffset<T, IndexType, Dims>::get(linearIndex, info);
}

template<typename IndexType, typename T>
void dispatchTakePutImpl(const Tensor& input, Tensor& output, const Tensor& index) {
  auto inputInfo = cuda::detail::getTensorInfo<T, IndexType>(input);
  inputInfo.collapseDims();
  auto numel = input.numel();
  if (inputInfo.isContiguous()) {
    cuda::CUDA_tensor_apply2<T, int64_t>(
        output,
        index,
        [inputInfo, numel] __device__ (
            T & out, const int64_t& idx) {
            auto offset = indexToOffset<-2, T, IndexType>(inputInfo, idx, numel);
            out = inputInfo.data[offset];
        });
  } else {
    cuda::CUDA_tensor_apply2<T, int64_t>(
        output,
        index,
        [inputInfo, numel] __device__ (
            T & out, const int64_t& idx) {
            auto offset = indexToOffset<-1, T, IndexType>(inputInfo, idx, numel);
            out = inputInfo.data[offset];
        });
  }
}

template<typename T>
void dispatchTakePut(const Tensor& input, Tensor& output, const Tensor& index) {
  if (cuda::detail::canUse32BitIndexMath(input)) {
    dispatchTakePutImpl<int32_t, T>(input, output, index);
  } else {
    dispatchTakePutImpl<int64_t, T>(input, output, index);
  }
}

template<int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, launch_bound2)
__global__ void index_elementwise_kernel(int N, func_t f) {
  int tid = threadIdx.x;
  int nv = nt * vt;
  int idx = nv * blockIdx.x + tid;
  #pragma unroll
  for (int i = 0; i < vt; i++) {
    if (idx < N) {
      f(idx);
      idx += nt;
    }
  }
}

template<int nt, int vt, typename func_t>
static void launch_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }
  dim3 block(nt);
  dim3 grid((N + block.x * vt - 1) / (block.x * vt));
  auto stream = at::cuda::getCurrentCUDAStream();
  index_elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, const func_t& f) {
  int num_indices = index_size.size();
  AT_ASSERT(num_indices == index_stride.size());
  AT_ASSERT(num_indices == iter.ntensors() - 2);

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_index_kernel(sub_iter, index_size, index_stride, f);
    }
    return;
  }

  auto sizes = at::detail::Array<int64_t, 25>(0);
  auto strides = at::detail::Array<int64_t, 25>(0);
  auto index_ptrs = at::detail::Array<char*, 25>(nullptr);
  for (int i = 0; i < num_indices; i++) {
    sizes[i] = index_size[i];
    strides[i] = index_stride[i];
    index_ptrs[i] = (char*)iter.data_ptr(i + 2);
  }

  char* out_ptr = (char*)iter.data_ptr(0);
  char* in_ptr = (char*)iter.data_ptr(1);

  auto offset_calc = make_offset_calculator<3>(iter);
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int idx) {
    auto offsets = offset_calc.get(idx);
    char* out_data = out_ptr + offsets[0];
    char* in_data = in_ptr + offsets[1];

    int64_t offset = 0;
    #pragma unroll
    for (int i = 0; i < num_indices; i++) {
      int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);
      CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
      if (index < 0) {
        index += sizes[i];
      }
      offset += index * strides[i];
    }

    f(out_data, in_data, offset);
  });
}

// The kernels are templated on an opaque, self-aligned type of the correct
// size to avoid redundant kernels for different types of the same size.
template <int N> struct alignas(N) OpaqueType { char data[N]; };


template <typename scalar_t>
void index_kernel_impl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* out_data, char* in_data, int64_t offset) {
    *(scalar_t*)out_data = *(scalar_t*)(in_data + offset);
  });
}

template <typename scalar_t>
void index_put_kernel_impl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  gpu_index_kernel(iter, index_size, index_stride, []C10_DEVICE(char* out_data, char* in_data, int64_t offset) {
    *(scalar_t*)(out_data + offset) = *(scalar_t*)in_data;
  });
}

static void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "index_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}


static void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  TORCH_CHECK(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, iter.dtype(), "index_put", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_put_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

static Tensor & masked_select_out_cuda_impl(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor or ByteTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  Tensor _mask = (mask.dim() == 0) ? mask.unsqueeze(0) : mask;
  Tensor _self = (self.dim() == 0) ? self.unsqueeze(0) : self;
  std::tie(_mask, _self) = expand_outplace(_mask, _self);
  at::native::index_out(result, _self, _mask);

  return result;
}

Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_cuda_impl(result, self, mask);
}

Tensor & masked_select_out_cuda(Tensor & result, const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_cuda_impl(result, self, mask);
}

void take_out_cuda_template(Tensor& output, const Tensor& input, const Tensor& index) {
  TORCH_CHECK(output.device().type() == at::kCUDA, "device type of output (", output.device().type(), ") is not GPU");
  TORCH_CHECK(input.device().type() == at::kCUDA, "device type of input (", input.device().type(), ") is not GPU");
  TORCH_CHECK(index.device().type() == at::kCUDA, "device type of index (", index.device().type(), ") is not GPU");

  TORCH_CHECK(output.layout() == Layout::Strided, "take() only supports strided layout, got layout: ", output.layout(), " on output tensor");
  TORCH_CHECK(input.layout() == Layout::Strided, "take() only supports strided layout, got layout: ", input.layout(), " on input tensor");
  TORCH_CHECK(index.layout() == Layout::Strided, "take() only supports strided layout, got layout: ", index.layout(), " on index tensor");

  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
          "output and input scalar type must match. but got different types: ", output.scalar_type(), " and ", input.scalar_type());
  TORCH_CHECK(index.scalar_type() == kLong, "index must be an int64 tensor");

  TensorArg output_arg{ output, "output", 1 };
  TensorArg input_arg{ input, "input", 2 };
  TensorArg index_arg{ index, "index", 3 };
  checkAllSameGPU("take", {output_arg, input_arg, index_arg});

  TORCH_CHECK(input.dim() < MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);
  TORCH_CHECK(output.dim() < MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);
  TORCH_CHECK(index.dim() < MAX_CUTORCH_DIMS, CUTORCH_DIM_WARNING);

  TORCH_CHECK(!(input.numel() == 0 && index.numel() != 0), "tried to take from an empty tensor");

  at::assert_no_internal_overlap(output);
  at::assert_no_partial_overlap(output, index);
  at::assert_no_overlap(output, input);

  output.resize_(index.sizes());

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Bool, at::ScalarType::Half, input.scalar_type(), "take_cuda", [&] {
    dispatchTakePut<scalar_t>(input, output, index);
  });
}

Tensor take_cuda(const Tensor& self, const Tensor& index) {
    auto out = at::empty(index.sizes(), self.options());
    take_out_cuda_template(out, self, index);
    return out;
}

Tensor& take_out_cuda(Tensor& out, const Tensor& self, const Tensor& index) {
    take_out_cuda_template(out, self, index);
    return out;
}

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);

}} // namespace at::native
