#ifndef SAMPLE_SHIM_H
#define SAMPLE_SHIM_H

// This is a sample shim file for testing get_shim_functions

#ifdef __cplusplus
extern "C" {
#endif

// This function is NOT versioned - should be ignored
AOTI_TORCH_EXPORT int unversioned_function(int arg);

// Simple versioned function (version 2.10)
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
AOTI_TORCH_EXPORT int simple_versioned_func(int arg);
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

// Multiple functions with version 2.9
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
AOTI_TORCH_EXPORT int old_function_1(int arg);
AOTI_TORCH_EXPORT void old_function_2(void* ptr);
#endif

// Typedef function pointer (version 2.10)
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
typedef int (*callback_function_ptr)(int, int);
#endif

// Nested version blocks with platform ifdef
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
#ifdef SOME_PLATFORM
AOTI_TORCH_EXPORT int platform_specific_func(int arg);
#endif
AOTI_TORCH_EXPORT int always_available_func(int arg);
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0

// Functions in #else branch should NOT be versioned
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
AOTI_TORCH_EXPORT int modern_implementation(int arg);
#else
AOTI_TORCH_EXPORT int legacy_fallback(int arg);
#endif

// Commented out function - should be ignored
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
// AOTI_TORCH_EXPORT int commented_out_func(int arg);
AOTI_TORCH_EXPORT int actual_function(int arg);
#endif

// Complex nested conditionals
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_12_0
#ifdef DEBUG_MODE
#if PLATFORM_VERSION >= 100
AOTI_TORCH_EXPORT int deeply_nested_func(int arg);
#endif
#endif
AOTI_TORCH_EXPORT int outer_block_func(int arg);
#endif

// Multiple typedefs with different versions
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
typedef void (*legacy_callback)(int);
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
typedef int (*modern_callback)(int, void*);
#endif

// Using declarations (type aliases)
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
struct OpaqueHandle {
  void* data;
  size_t size;
};
using HandleType = OpaqueHandle*;
#endif

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
struct NewOpaqueStruct {
  int32_t type;
  void* buffer;
  size_t capacity;
};

class NewOpaqueClass {
 public:
  virtual ~NewOpaqueClass() = default;
  virtual void process() = 0;
};

using NewHandleType = NewOpaqueStruct*;
#endif

// Function after #elif should not be versioned
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
AOTI_TORCH_EXPORT int primary_path(int arg);
#elif TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
AOTI_TORCH_EXPORT int secondary_path(int arg);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SAMPLE_SHIM_H
