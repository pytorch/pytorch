// Device-side rocFFT JIT callbacks for the stft/istft fusion path.
//
// This file is not part of any normal HIP translation unit. It is compiled on
// its own to SPIR-V (see cmake/RocFFTCallbacks.cmake) and the resulting module
// is handed to rocfft_plan_description_set_{load,store}_callback, which links
// the named symbol into rocFFT's own kernels at plan creation time. Nothing
// here is reachable from host code, so it lives outside the native/hip glob.
//
// Load callbacks have the signature
//   T load_cb(T* data, size_t offset, void* cbdata, void* sharedMem)
// and store callbacks
//   void store_cb(T* data, size_t offset, T element, void* cbdata, void* sharedMem)
// where T is one element of the buffer rocFFT is reading or writing, and offset
// is an index into the layout declared on the plan.
//
// The gather callbacks exploit the fact that rocFFT only ever uses `offset` to
// index the pointer it is given: pass the raw signal instead of a framed buffer
// and the frames are read straight out of it, so time2col never runs.

#include <ATen/native/hip/rocfft/CallbackData.h>

#include <hip/hip_runtime.h>

using at::native::detail::RocFFTCallbackData;

// Gather one element of frame `f` of channel `c` out of the raw signal and
// apply the analysis window.
extern "C" __device__ float at_rocfft_load_gather(
    float* data, size_t offset, void* cbdata, void*) {
  const RocFFTCallbackData* d = static_cast<const RocFFTCallbackData*>(cbdata);
  const unsigned off = static_cast<unsigned>(offset);
  const unsigned k = off / d->n_fft;  // frame index, flattened over channels
  const unsigned i = off - k * d->n_fft;
  const unsigned c = k / d->n_frames;
  const unsigned f = k - c * d->n_frames;
  return data[c * d->signal_stride + f * d->hop + i] * d->window[i];
}

// Same, for a power-of-two transform length, where the division collapses to a
// shift and a mask.
extern "C" __device__ float at_rocfft_load_gather_pow2(
    float* data, size_t offset, void* cbdata, void*) {
  const RocFFTCallbackData* d = static_cast<const RocFFTCallbackData*>(cbdata);
  const unsigned off = static_cast<unsigned>(offset);
  const unsigned k = off >> d->n_fft_log2;
  const unsigned i = off & (d->n_fft - 1);
  const unsigned c = k / d->n_frames;
  const unsigned f = k - c * d->n_frames;
  return data[c * d->signal_stride + f * d->hop + i] * d->window[i];
}

// istft: apply the synthesis window as the C2R output is written, instead of in
// a separate pass over the frames. Each output element is written once, so this
// stays bit-exact and deterministic.
extern "C" __device__ void at_rocfft_store_window(
    float* data, size_t offset, float element, void* cbdata, void*) {
  const RocFFTCallbackData* d = static_cast<const RocFFTCallbackData*>(cbdata);
  data[offset] = element * d->window[static_cast<unsigned>(offset) % d->n_fft];
}

// istft: window and accumulate straight into a signal-shaped buffer, collapsing
// framing, windowing and overlap-add into the transform's store. Frames overlap,
// so the additions race and the result is not bitwise reproducible; callers must
// gate this on deterministic algorithms being off.
extern "C" __device__ void at_rocfft_store_overlap_add(
    float* data, size_t offset, float element, void* cbdata, void*) {
  const RocFFTCallbackData* d = static_cast<const RocFFTCallbackData*>(cbdata);
  const unsigned off = static_cast<unsigned>(offset);
  const unsigned k = off / d->n_fft;
  const unsigned i = off - k * d->n_fft;
  const unsigned c = k / d->n_frames;
  const unsigned f = k - c * d->n_frames;
  atomicAdd(&data[c * d->signal_stride + f * d->hop + i], element * d->window[i]);
}
