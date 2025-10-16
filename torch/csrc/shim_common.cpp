#include <torch/csrc/stable/c/shim.h>

#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/ParallelGuard.h>

#define TORCH_SHIM_CONVERT_EXCEPTION_TO_ERROR_CODE(...) \
  try {                                                 \
    __VA_ARGS__                                         \
  } catch (const std::exception& e) {                   \
    C10_LOG_API_USAGE_ONCE("torch_shim_exception");     \
    return TORCH_SHIM_FAILURE;                          \
  }                                                     \
  return TORCH_SHIM_SUCCESS;

// ABI stable parallel utilities implementations
void lazy_init_num_threads() {
  at::internal::lazy_init_num_threads();
}

bool in_parallel_region() {
  return at::in_parallel_region();
}

uint64_t get_num_threads() {
  return static_cast<uint64_t>(at::get_num_threads());
}

uint64_t get_thread_idx() {
  return static_cast<uint64_t>(at::get_thread_num());
}

TorchShimError create_thread_id_guard(
    uint64_t thread_id,
    ThreadIdGuardHandle* ret_guard) {
  TORCH_SHIM_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::internal::ThreadIdGuard* guard =
        new at::internal::ThreadIdGuard(thread_id);
    *ret_guard = reinterpret_cast<ThreadIdGuardHandle>(guard);
  });
}

TorchShimError delete_thread_id_guard(ThreadIdGuardHandle guard) {
  TORCH_SHIM_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::internal::ThreadIdGuard* tid_guard =
        reinterpret_cast<at::internal::ThreadIdGuard*>(guard);
    delete tid_guard;
  });
}

TorchShimError create_parallel_guard(
    bool state,
    ParallelGuardHandle* ret_guard) {
  TORCH_SHIM_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::ParallelGuard* guard = new c10::ParallelGuard(state);
    *ret_guard = reinterpret_cast<ParallelGuardHandle>(guard);
  });
}

TorchShimError delete_parallel_guard(ParallelGuardHandle guard) {
  TORCH_SHIM_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::ParallelGuard* parallel_guard =
        reinterpret_cast<c10::ParallelGuard*>(guard);
    delete parallel_guard;
  });
}

bool parallel_guard_is_enabled() {
  return c10::ParallelGuard::is_enabled();
}

bool intra_op_parallel_enabled() {
#ifdef INTRA_OP_PARALLEL
  return true;
#else
  return false;
#endif
}

bool openmp_is_available() {
  return AT_PARALLEL_OPENMP;
}

TorchShimError invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    ParallelFunc lambda,
    void* ctx) {
#if !AT_PARALLEL_NATIVE
  TORCH_SHIM_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        false,
        "Only use invoke_parallel if libtorch built with AT_PARALLEL_NATIVE=1");
  });
#else
  TORCH_SHIM_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto wrapper = [lambda, ctx](int64_t chunk_begin, int64_t chunk_end) {
      lambda(chunk_begin, chunk_end, ctx);
    };

    at::internal::invoke_parallel(begin, end, grain_size, wrapper);
  });
#endif
}
