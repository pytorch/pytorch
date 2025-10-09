#pragma once

#include <torch/csrc/stable/stableivalue_conversions.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/csrc/stable/parallel_utils.h>
#include <torch/headeronly/core/ScalarType.h>

namespace torch::stable {

// We expect this to be the stable version of the empty_like op that takes in
// no kwargs (device, dtype, layout, memory_format). We will add kwargs
// support in the future.
inline torch::stable::Tensor empty_like(const torch::stable::Tensor& self) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::empty_like", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the fill_.Scalar op
// with identical semantics to the existing fill_.Scalar op.
// A subtle nuance is that `value` is typed as a double, but it is
// actually a Scalar. This is because Scalar.h is currently not
// header-only.
inline torch::stable::Tensor fill_(
    const torch::stable::Tensor& self,
    double value) {
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_fill__Scalar(self.get(), value));
  return self;
}

// We expect this to be the stable version of the narrow.default op.
// narrow takes in a SymInt for start and length, but these are typed as
// int64_t as SymInt is not yet header-only.
inline torch::stable::Tensor narrow(
    torch::stable::Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_narrow(self.get(), dim, start, length, &ret0));
  return torch::stable::Tensor(ret0);
}

// We expect this to be a stable version of the new_empty op that takes in
// only dtype information.
inline torch::stable::Tensor new_empty(
    const torch::stable::Tensor& self,
    std::vector<int64_t> size,
    std::optional<c10::ScalarType> dtype = std::nullopt) {
  int32_t device_type;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(self.get(), &device_type));

  int32_t device_index;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(self.get(), &device_index));

  int32_t target_dtype;
  if (dtype.has_value()) {
    target_dtype = to<int32_t>(from(dtype.value()));
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &target_dtype));
  }

  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));

  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_empty(
      self.get(),
      size.data(),
      static_cast<int64_t>(size.size()),
      &target_dtype,
      &layout,
      &device_type,
      device_index,
      nullptr, // pin_memory (nullptr for default)
      &ret0));

  return torch::stable::Tensor(ret0);
}

// We expect this to be a stable version of the new_zeros op that takes in
// only dtype information.
inline torch::stable::Tensor new_zeros(
    const torch::stable::Tensor& self,
    std::vector<int64_t> size,
    std::optional<c10::ScalarType> dtype = std::nullopt) {
  int32_t device_type;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(self.get(), &device_type));

  int32_t device_index;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(self.get(), &device_index));

  int32_t target_dtype;
  if (dtype.has_value()) {
    target_dtype = to<int32_t>(from(dtype.value()));
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &target_dtype));
  }

  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));

  AtenTensorHandle ath;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_zeros(
      self.get(),
      size.data(),
      static_cast<int64_t>(size.size()),
      &target_dtype,
      &layout,
      &device_type,
      device_index,
      nullptr, // pin_memory (nullptr for default)
      &ath));

  return torch::stable::Tensor(ath);
}

// We expect this to be the stable version of the pad.default op.
// pad.default takes in a SymInt[] as the pad argument however pad is typed as
// use std::vector<int64_t> because
// (1) IntArrayRef is not yet header-only
// (2) SymInt is not yet header-only
inline torch::stable::Tensor pad(
    const torch::stable::Tensor& self,
    std::vector<int64_t> pad,
    const std::string& mode = "constant",
    double value = 0.0) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_pad(
      self.get(), pad.data(), pad.size(), mode.c_str(), &value, &ret0));
  return torch::stable::Tensor(ret0);
}

// We expect the following two functions to be stable versions of the
// amax.default op with identical semantics to the existing amax.default op. If
// `keepdim` is true, the result will have the same number of dimensions as
// `self`, with the specified dimension having size 1. Otherwise, the result
// will have one fewer dimension than `self`, with the specified dimension
// removed.

// This function is an overload to compute the maximum value along each slice of
// `self` along a single dimension `dim`.
inline torch::stable::Tensor amax(
    const torch::stable::Tensor& self,
    int64_t dim,
    bool keepdim = false) {
  AtenTensorHandle ret = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_amax(self.get(), &dim, 1, keepdim, &ret));
  return torch::stable::Tensor(ret);
}

// This function is an overload to compute the maximum value along each slice of
// `self` reducing over all the dimensions in the vector `dims`. The
// amax.default op takes in a SymInt[] as the dims argument, however dims is
// typed as use std::vector<int64_t> here because (1) IntArrayRef is not yet
// header-only (2) SymInt is not yet header-only
inline torch::stable::Tensor amax(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims,
    bool keepdim = false) {
  AtenTensorHandle ret = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_amax(
      self.get(),
      dims.data(),
      static_cast<int64_t>(dims.size()),
      keepdim,
      &ret));
  return torch::stable::Tensor(ret);
}

// We expect this to be the stable version of the transpose op with identical
// semantics to the existing transpose.int op.
inline torch::stable::Tensor transpose(
    const torch::stable::Tensor& self,
    int64_t dim0,
    int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{from(self), from(dim0), from(dim1)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::transpose", "int", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the zero_ op with identical
// semantics to the existing zero_ op (except that it will not be called as
// a tensor method but only as a function i.e. zero_(t) not t.zero_()).
inline torch::stable::Tensor zero_(torch::stable::Tensor& self) {
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{from(self)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::zero_", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the copy_ op with
// identical semantics to the existing copy_ op.
inline torch::stable::Tensor copy_(
    torch::stable::Tensor& self,
    const torch::stable::Tensor& src,
    std::optional<bool> non_blocking = std::nullopt) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      from(self), from(src), from(non_blocking.value_or(false))};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::copy_", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the clone op. We will
// add optional memory_format kwarg support in the future.
inline torch::stable::Tensor clone(const torch::stable::Tensor& self) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{from(self), from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::clone", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

namespace internal {

// Copied from aten/src/ATen/Parallel.h
inline int64_t divup(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// We expect this to be the stable version of the OpenMP variant of the
// invoke_parallel op with identical semantics to the existing invoke_parallel.
// This is copy pasted from aten/src/ATen/ParallelOpenMP.h except that we
// replace ThreadIdGuard with the shim-ed version.
// Requiring the extension to have _OPENMP defined to use invoke_parallel
// matches the existing semantic.
#ifdef _OPENMP
template <typename F>
inline void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f) {
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;

#pragma omp parallel
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See
    // #32008)
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        ThreadIdGuard tid_guard(tid);
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
}
#else
template <typename F>
inline void invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const F& f) {
  STD_TORCH_CHECK(
      false,
      "Attempting to call torch::stable::invoke_parallel "
      "without _OPENMP. Internal error, should not have gotten here");
}
#endif // _OPENMP

// For the ParallelNative path, this helps with converting C++ lambdas
// etc. to a C-style function pointer expected by the C-shim
template <typename F>
struct Trampoline {
  static void invoke(int64_t begin, int64_t end, void* ctx) {
    ParallelGuard guard(true);
    F* fn = static_cast<F*>(ctx);
    (*fn)(begin, end);
  }
};

} // namespace internal

#ifdef _OPENMP
#define EXTENSION_HAS_OPENMP 1
#else
#define EXTENSION_HAS_OPENMP 0
#endif

// We expect this to be the ABI stable version of parallel_for with identical
// semantics to at::parallel_for
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  STD_TORCH_CHECK(grain_size >= 0);
  if (begin >= end) {
    return;
  }

  // INTRA_OP_PARALLEL = 1 --> either AT_PARALLEL_OPENMP or AT_PARALLEL_NATIVE
  // For the first case, we additionally need to make sure EXTENSION_HAS_OPENMP
  // in order to use the parallel path (otherwise the extension doesn't know
  // how to compile invoke_parallel). This is consistent with the existing
  // semantic.
  if (aoti_torch_get_intra_op_parallel_enabled() &&
      ((!aoti_torch_get_parallel_openmp_enabled()) || EXTENSION_HAS_OPENMP)) {
    aoti_torch_lazy_init_num_threads();
    const auto numiter = end - begin;
    const bool use_parallel =
        (numiter > grain_size && numiter > 1 &&
         !aoti_torch_in_parallel_region() && aoti_torch_get_num_threads() > 1);
    if (!use_parallel) {
      ThreadIdGuard tid_guard(0);
      ParallelGuard guard(true);
      f(begin, end);
      return;
    }

    if (aoti_torch_get_parallel_openmp_enabled()) {
      // From above check we know EXTENSION_HAS_OPENMP == 1
      // For parallel openmp path (default), we call internal::invoke_parallel
      // defined in this header so inlining of f still happens
      internal::invoke_parallel(
          begin, end, grain_size, [&](int64_t begin, int64_t end) {
            ParallelGuard guard(true);
            f(begin, end);
          });
    } else {
      // For parallel native path, we call the shim-ed invoke_parallel, the
      // native invoke_parallel takes in std::function (not templated F) so
      // there's no inlining anyway.
      TORCH_ERROR_CODE_CHECK(aoti_torch_invoke_parallel(
          begin,
          end,
          grain_size,
          &internal::Trampoline<F>::invoke,
          const_cast<void*>(static_cast<const void*>(&f))));
    }
  } else {
    ThreadIdGuard tid_guard(0);
    ParallelGuard guard(true);
    f(begin, end);
  }
}

} // namespace torch::stable
