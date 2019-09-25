#include <ATen/native/Indexing.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/core/Array.h>

namespace at { namespace native {

template <int N>
static OffsetCalculator<N> index_make_offset_calculator(const TensorIterator& iter) {
  AT_ASSERT(N <= iter.ntensors());
  std::array<const int64_t*, N> strides;
  for (int i = 0; i < N; i++) {
    strides[i] = iter.strides(i).data();
  }
  return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data());
}

__device__ char* device_strcpy(char* dst, const char* src, int n){
  int i = 0;
  while (i+1 < n && src[i]) {
    dst[i] = src[i];
    ++i;
  }
  if (i < n) {
    dst[i] = '\0';
  }
  return dst;
}

__device__ bool _graceful_assert(c10::cuda::CUDAAssert* assert_state, bool condition, const char* message, uint32_t line, const char* file) {
  if (!condition) { 
    if (atomicCAS(&assert_state->error, 0, 1) == 0) {

        // copy message
        device_strcpy(assert_state->message, message, c10::cuda::MAX_ASSERT_MSG_LENGTH);
        device_strcpy(assert_state->file, file, c10::cuda::MAX_ASSERT_MSG_LENGTH);
        assert_state->line = line;
        assert_state->type = 1;

        // fill details
        c10::cuda::CUDAAssertDetailIndexKernel* details = reinterpret_cast<c10::cuda::CUDAAssertDetailIndexKernel*>(assert_state->details);
        details->index = 1234;  // write some dummy details
    }
  }

  return !assert_state->error;    // return false if we are in error state, signals kernel to quit
}

#define graceful_assert(assert_state, exp, msg) \
  _graceful_assert(assert_state, exp, msg, static_cast<uint32_t>(__LINE__), __FILE__)

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

  auto offset_calc = index_make_offset_calculator<3>(iter);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto assert_state = stream.assert_state();

  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), [=]__device__(int idx) {
    auto offsets = offset_calc.get(idx);
    char* out_data = out_ptr + offsets[0];
    char* in_data = in_ptr + offsets[1];

    int64_t offset = 0;
    #pragma unroll
    for (int i = 0; i < num_indices; i++) {
      int64_t index = *(int64_t*)(index_ptrs[i] + offsets[2]);

      if (!graceful_assert(assert_state, index >= -sizes[i] && index < sizes[i], "index out of bounds")) {
        return;
      }

      /*if (index < -sizes[i] || index >= sizes[i]) {
        if (atomicExch(&ds->err, 1) == 0) {
          // Only the first thread that encounters an error records the set of values
          // for the error message.
          ds->index = index;
          ds->axis = i;
          ds->size = sizes[i];
          __threadfence_system();
          assert(0);
        }
        return;
      }*/
      
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
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, iter.dtype(), "index_cuda", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}


static void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  AT_ASSERTM(!accumulate, "index_put does not support accumulate=true");
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, iter.dtype(), "index_put", [&] {
    using dtype = OpaqueType<sizeof(scalar_t)>;
    index_put_kernel_impl<dtype>(iter, index_size, index_stride);
  });
}

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);

}} // namespace at::native
