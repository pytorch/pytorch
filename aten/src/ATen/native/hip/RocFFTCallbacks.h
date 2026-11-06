#pragma once

// Host side of the rocFFT JIT callback path. The callbacks themselves are
// device code in rocfft/Callbacks.hip, compiled to a SPIR-V module that rocFFT
// links into its kernels; see cmake/RocFFTCallbacks.cmake.
//
// This file is hand-written for ROCm and is not produced by hipify: everything
// under */hip/* is in the ignore list of tools/amd_build/build_amd.py.

#include <ATen/native/hip/rocfft/CallbackData.h>

#include <rocfft/rocfft.h>

#include <cstdint>

namespace at::native::detail {

// Which callback a plan is built with. rocFFT compiles the callback into the
// plan's kernels, so two plans that differ only by this are different plans.
enum class RocFFTCallbackKind : int8_t {
  None,
  LoadGather,       // time2col gather + analysis window, read from the raw signal
  LoadGatherPow2,   // same, for a power-of-two transform length
  StoreWindow,      // synthesis window applied as the output is written
  StoreOverlapAdd,  // synthesis window plus atomic overlap-add into the signal
};

// Attaches `kind` to `desc` so that rocFFT links it into the kernels it compiles
// for the plan. Only meaningful once rocfft_callbacks_available() has said yes.
void rocfft_register_callback(rocfft_plan_description desc, RocFFTCallbackKind kind);

// Points the callback of a plan built with `kind` at `data`, which must be a
// device pointer to a RocFFTCallbackData.
void rocfft_set_callback_data(rocfft_execution_info info, RocFFTCallbackKind kind, void* data);

// Whether rocFFT can actually JIT a callback into its kernels here. Doing so
// goes through hiprtc targeting amdgcnspirv, which older ROCm runtimes cannot
// compile and which no build-time check can see, so this plans a throwaway
// transform once and reports whether that worked.
bool rocfft_callbacks_available();

} // namespace at::native::detail
