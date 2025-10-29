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

// Device APIs for stable ABI
struct DeviceOpaque;
using DeviceHandle = DeviceOpaque*;

AOTI_TORCH_EXPORT AOTITorchError torch_create_device(
    int32_t device_type,
    int32_t device_index,
    DeviceHandle* ret_device);

AOTI_TORCH_EXPORT AOTITorchError torch_create_device_from_string(
    const char* device_string,
    DeviceHandle* ret_device);

AOTI_TORCH_EXPORT AOTITorchError
torch_new_device_handle(DeviceHandle orig_handle, DeviceHandle* new_handle);

AOTI_TORCH_EXPORT AOTITorchError torch_delete_device(DeviceHandle device);

AOTI_TORCH_EXPORT AOTITorchError
torch_device_type(DeviceHandle device, int32_t* ret_device_type);

AOTI_TORCH_EXPORT AOTITorchError
torch_device_index(DeviceHandle device, int32_t* ret_device_index);

AOTI_TORCH_EXPORT AOTITorchError
torch_device_set_index(DeviceHandle device, int32_t device_index);

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STABLE_TORCH_SHIM
