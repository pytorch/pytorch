#ifndef TORCH_SHIM_H
#define TORCH_SHIM_H

#include <stdint.h>

#ifdef __GNUC__
#define TORCH_SHIM_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
#ifdef EXPORT_TORCH_SHIM_FUNCTIONS
#define TORCH_SHIM_EXPORT __declspec(dllexport)
#else
#define TORCH_SHIM_EXPORT __declspec(dllimport)
#endif
#else // !_WIN32
#define TORCH_SHIM_EXPORT
#endif // _WIN32
#endif // __GNUC__

#ifdef __cplusplus
extern "C" {
#endif

// Error handling
typedef int32_t TorchShimError;
#define TORCH_SHIM_SUCCESS 0
#define TORCH_SHIM_FAILURE 1

// parallel utilities
// `lazy_init_num_threads` is only intended for use in the torch/csrc/stable/
// implementation of parallel_for and should NOT be used in user code.
TORCH_SHIM_EXPORT void lazy_init_num_threads();
TORCH_SHIM_EXPORT bool in_parallel_region();
TORCH_SHIM_EXPORT uint64_t get_num_threads();
TORCH_SHIM_EXPORT uint64_t get_thread_idx();

struct ThreadIdGuardOpaque;
using ThreadIdGuardHandle = ThreadIdGuardOpaque*;

TORCH_SHIM_EXPORT TorchShimError
create_thread_id_guard(uint64_t thread_id, ThreadIdGuardHandle* ret_guard);

TORCH_SHIM_EXPORT TorchShimError
delete_thread_id_guard(ThreadIdGuardHandle guard);

struct ParallelGuardOpaque;
using ParallelGuardHandle = ParallelGuardOpaque*;

TORCH_SHIM_EXPORT TorchShimError
create_parallel_guard(bool state, ParallelGuardHandle* ret_guard);

TORCH_SHIM_EXPORT TorchShimError
delete_parallel_guard(ParallelGuardHandle guard);

TORCH_SHIM_EXPORT bool parallel_guard_is_enabled();

// Check if INTRA_OP_PARALLEL is defined
TORCH_SHIM_EXPORT bool intra_op_parallel_enabled();

// Value of AT_PARALLEL_OPENMP
TORCH_SHIM_EXPORT bool openmp_is_available();

// invoke_parallel function, only intended for use for the
// AT_PARALLEL_NATIVE path, where the function is not inlined.
typedef void (*ParallelFunc)(int64_t begin, int64_t end, void* ctx);

TORCH_SHIM_EXPORT TorchShimError invoke_parallel(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    ParallelFunc lambda,
    void* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TORCH_SHIM_H
