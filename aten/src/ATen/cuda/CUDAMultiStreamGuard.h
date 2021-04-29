#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

namespace at { namespace cuda {

// TODO: Implement this generically in c10.  You'll need some way to get
// the number of GPUs from the GuardImpl, in that case.
class CUDAMultiStreamGuard final {
public:
  /// Calls `set_stream` on each of the streams in the list.
  /// This may be useful if you need to set different streams
  /// for different devices.
  explicit CUDAMultiStreamGuard(ArrayRef<CUDAStream> streams) {
    original_streams_.reserve(streams.size());
    for (const auto& s : streams) {
      original_streams_.push_back(getCurrentCUDAStream(s.device_index()));
      setCurrentCUDAStream(s);
    }
  }

  CUDAMultiStreamGuard(const CUDAMultiStreamGuard&) = delete;
  CUDAMultiStreamGuard& operator=(const CUDAMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  CUDAMultiStreamGuard(CUDAMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  CUDAMultiStreamGuard& operator=(CUDAMultiStreamGuard&& other) = delete;

  /// Resets the CUDA stream on each device to the one that was active upon
  /// construction.
  ~CUDAMultiStreamGuard() {
    for (const auto& s : original_streams_) {
      setCurrentCUDAStream(s);
    }
  }

private:
  /// The original streams that were active on the affected devices.
  std::vector<CUDAStream> original_streams_;
};

}} // namespace at::cuda
