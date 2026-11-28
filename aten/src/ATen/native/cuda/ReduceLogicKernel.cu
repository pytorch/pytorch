#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/Dispatch.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/CallOnce.h>

#include <array>

namespace at::native {

namespace {

#ifdef USE_ROCM
bool is_gfx1250_device(c10::DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(
      device_index >= 0 && device_index < C10_COMPILE_TIME_MAX_GPUS);
  static std::array<c10::once_flag, C10_COMPILE_TIME_MAX_GPUS> once_flags;
  static std::array<bool, C10_COMPILE_TIME_MAX_GPUS> results{};
  c10::call_once(once_flags[device_index], [device_index] {
    results[device_index] =
        at::detail::getCUDAHooks().isGPUArch({"gfx1250"}, device_index);
  });
  return results[device_index];
}

// Bytes per vectorized input load for any/all along the contiguous dim on
// gfx1250, the widest vector load; gpu_reduce_kernel's default is 4 elements.
constexpr int kLogicReduceInputVectorBytes = 16;

// Elements per input vector for any/all of a 1- or 2-byte scalar_t along one
// dim that is unit-stride in the input (TensorIterator places the reduced dim
// at dim 0) with more than one output, or 0 to keep the default.
// setReduceConfig vectorizes the input only for rows of at least
// max(128, vec * vec) elements, so this takes the widest vector that still
// vectorizes the row: 1-byte rows of 128 to 255 take 8 elements. The gate is
// the arch, not the warp size: gfx1250 is the only wave32 GPU this was
// measured on, so other wave32 GPUs keep the default.
template <typename scalar_t>
int contiguous_logic_reduce_vec_size(const TensorIterator& iter) {
  const int64_t in = iter.ntensors() - 1;
  const bool contiguous_inner_reduction = iter.num_reduce_dims() == 1 &&
      iter.ndim() >= 1 && iter.strides(in)[0] == iter.element_size(in) &&
      (iter.ndim() == 1 || iter.strides(in)[0] < iter.strides(in)[1]);
  if (!contiguous_inner_reduction || iter.num_output_elements() <= 1 ||
      iter.shape()[0] < 128) {
    return 0;
  }
  int vec = kLogicReduceInputVectorBytes / static_cast<int>(sizeof(scalar_t));
  while (static_cast<int64_t>(vec) * vec > iter.shape()[0]) {
    vec /= 2;
  }
  return is_gfx1250_device(iter.device(0).index()) ? vec : 0;
}
#endif

template <typename scalar_t, typename func_t>
void launch_logic_reduce_kernel(
    TensorIterator& iter,
    const func_t& f,
    bool ident) {
#ifdef USE_ROCM
  if constexpr (sizeof(scalar_t) <= 2) {
    const int vec_size = contiguous_logic_reduce_vec_size<scalar_t>(iter);
    if constexpr (sizeof(scalar_t) == 1) {
      if (vec_size == 16) {
        gpu_reduce_kernel<scalar_t, bool, /*vt0=*/4, /*input_vec_size=*/16>(
            iter, f, ident);
        return;
      }
    }
    if (vec_size == 8) {
      gpu_reduce_kernel<scalar_t, bool, /*vt0=*/4, /*input_vec_size=*/8>(
          iter, f, ident);
      return;
    }
  }
#endif
  gpu_reduce_kernel<scalar_t, bool>(iter, f, ident);
}

} // namespace

void and_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_cuda", [&]() {
        launch_logic_reduce_kernel<scalar_t>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc && static_cast<bool>(val));
            }),
            true);
      });
}

void or_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_cuda", [&]() {
        launch_logic_reduce_kernel<scalar_t>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(bool acc, scalar_t val) -> bool {
              return (acc || static_cast<bool>(val));
            }),
            false);
      });
}

REGISTER_DISPATCH(and_stub, &and_kernel_cuda)
REGISTER_DISPATCH(or_stub, &or_kernel_cuda)

} // namespace at::native
