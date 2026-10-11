#pragma once

// Layout description passed to the rocFFT JIT callbacks in Callbacks.hip.
// Included by both the device source and the host code that fills it in, so the
// two cannot drift apart.

#include <cstdint>

namespace at::native::detail {

// Describes the framing that the callbacks synthesize on the fly: the FFT sees
// a (channels, frames, n_fft) buffer that is never materialized, backed by a
// raw (channels, signal) one.
//
// Single precision only. rocFFT picks the callback's element type from the
// plan, so a double-precision transform would need a separate set of symbols.
struct RocFFTCallbackData {
  const float* window;
  uint32_t n_fft;
  uint32_t n_frames;
  uint32_t hop;
  uint32_t signal_stride;  // elements between channels in the raw signal
  uint32_t n_fft_log2;     // valid only when n_fft is a power of two
};

} // namespace at::native::detail
