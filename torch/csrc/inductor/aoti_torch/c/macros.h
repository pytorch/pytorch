#ifndef AOTI_TORCH_MACRO_H
#define AOTI_TORCH_MACRO_H

#include <stddef.h>
#include <stdint.h>
#ifdef __GNUC__
#define AOTI_TORCH_EXPORT __attribute__((__visibility__("default")))
#else // !__GNUC__
#ifdef _WIN32
// PyTorch2 doesn't currently work on Windows. Exporting these APIs can lead
// to symbol clashes at link time if libtorch is included in a DLL and binary
// that depends on the DLL. As a short term fix, we don't export the symbols.
// In the long term, this will need to be addressed when Windows is supported.
#ifdef OVRSOURCE
// Do not export AOTI on Windows for internal builds
#define AOTI_TORCH_EXPORT
#else /* OVRSOURCE */
#ifdef EXPORT_AOTI_FUNCTIONS
#define AOTI_TORCH_EXPORT __declspec(dllexport)
#else
#define AOTI_TORCH_EXPORT __declspec(dllimport)
#endif
#endif /* OVRSOURCE */
#else // !_WIN32
#define AOTI_TORCH_EXPORT
#endif // _WIN32
#endif // __GNUC__

#ifdef __cplusplus
extern "C" {
#endif
// AtenTensorHandle represents an abstract notion of Tensor that can be passed
// between model.so and libtorch.so.  The contents of the structure itself
// are private; model.so is not allowed to access any fields directly, it must
// go through functions defined in this ABI.  Under the hood, this is
// represented as at::Tensor*, but we reserve the right to change this (and in
// fact, we probably should change it to at::TensorImpl* at least).
//
// An AtenTensorHandle can be owning (please check the API reference for exact
// ownership/borrow semantics).  If you have an owning AtenTensorHandle
// in model.so, you are obligated to aoti_torch_delete_tensor_object when you
// are done.  You can use the helper C++ class RAIIAtenTensorHandle
// (see aot_runtime/model.h) to ensure the deallocator is called in RAII style
// (note that RAIIAtenTensorHandle is private to model.so, and never crosses
// the ABI boundary.)
struct AtenTensorOpaque;
using AtenTensorHandle = AtenTensorOpaque*;

struct AtenGeneratorOpaque;
using AtenGeneratorHandle = AtenGeneratorOpaque*;

struct AOTIProxyExecutorOpaque;
using AOTIProxyExecutorHandle = AOTIProxyExecutorOpaque*;

struct C10IValueOpaque;
using C10IValueHandle = C10IValueOpaque*;

using AOTITorchError = int32_t;
#define AOTI_TORCH_SUCCESS 0
#define AOTI_TORCH_FAILURE 1

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOTI_TORCH_MACRO_H
