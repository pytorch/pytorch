#ifndef STABLE_TORCH_SHIM
#define STABLE_TORCH_SHIM

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/csrc/stable/version.h>

// This header defines stable C API extensions for backward/forward
// compatibility when calling ATen operations through the dispatcher.
//
// This is separate from the main AOTI shim to provide versioning capabilities
// for schema changes in native ATen functions.

#ifdef __cplusplus
extern "C" {
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
using StableIValue = uint64_t;

// Has the same semantic as aoti_torch_call_dispatcher, but takes an
// additional argument for the extension build version. This is
// needed for backward compatibility when calling native functions via
// the dispatcher. The caller should pass in the libtorch version the
// extension is building with (NOT target version).
AOTI_TORCH_EXPORT AOTITorchError torch_call_dispatcher(
    const char* opName,
    const char* overloadName,
    StableIValue* stack,
    uint64_t extension_build_version);

// Version-aware variant of aoti_torch_library_impl that takes an
// extension_build_version parameter for backward compatibility
AOTI_TORCH_EXPORT AOTITorchError torch_library_impl(
    TorchLibraryHandle self,
    const char* name,
    void (*fn)(StableIValue*, uint64_t, uint64_t),
    uint64_t extension_build_version);

struct StableListOpaque;
using StableListHandle = StableListOpaque*;

// returns an owning reference of a StableList. callee is responsible for
// freeing memory.
AOTI_TORCH_EXPORT AOTITorchError
torch_new_list_reserve_size(size_t size, StableListHandle* ret);

AOTI_TORCH_EXPORT AOTITorchError
torch_list_size(StableListHandle list_handle, size_t* size);

AOTI_TORCH_EXPORT AOTITorchError torch_list_get_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue* element);

AOTI_TORCH_EXPORT AOTITorchError torch_list_set_item(
    StableListHandle list_handle,
    size_t index,
    StableIValue element);

AOTI_TORCH_EXPORT AOTITorchError
torch_list_push_back(StableListHandle list_handle, StableIValue element);

// deletes the underlying list referenced by list_handle
AOTI_TORCH_EXPORT AOTITorchError
torch_delete_list(StableListHandle list_handle);

// Helper function to parse device string using c10::Device
// Returns device type and index via output parameters
AOTI_TORCH_EXPORT AOTITorchError torch_parse_device_string(
    const char* device_string,
    uint32_t* out_device_type,
    int32_t* out_device_index);

// Parallel utility APIs for stable ABI
// Function pointer type for parallel_for callback
// The callback receives begin and end indices for a range to process
typedef void (*ParallelFunc)(int64_t begin, int64_t end, void* ctx);

AOTI_TORCH_EXPORT AOTITorchError torch_parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    ParallelFunc func,
    void* ctx);

// Get the current thread index in a parallel region
// Returns 0 if not in a parallel region
AOTI_TORCH_EXPORT uint32_t torch_get_thread_idx();

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_TORCH_SHIM
