#pragma once

#include <c10/hip/HIPStream.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// See Note [Masquerading as CUDA] for motivation

class HIPStreamMasqueradingAsCUDA final : public c10::cuda::CUDAStream {
public:
  using c10::cuda::CUDAStream::CUDAStream;

  static HIPStreamMasqueradingAsCUDA unpack3(StreamId stream_id,
                                             DeviceIndex device_index,
                                             c10::DeviceType device_type) {
    // NB: constructor manages CUDA->HIP translation for us
    return HIPStreamMasqueradingAsCUDA(Stream::unpack3(
        stream_id, device_index, device_type));
  }

  // New method, gets the underlying "HIPStream" [CUDAStream]
  c10::cuda::CUDAStream hip_stream() const { return *this; }
};

HIPStreamMasqueradingAsCUDA
inline getStreamFromPoolMasqueradingAsCUDA(const bool isHighPriority = false, DeviceIndex device = -1) {
  return HIPStreamMasqueradingAsCUDA(c10::cuda::getStreamFromPool(isHighPriority, device));
}

HIPStreamMasqueradingAsCUDA
inline getStreamFromPoolMasqueradingAsCUDA(const int priority, DeviceIndex device = -1) {
  return HIPStreamMasqueradingAsCUDA(c10::cuda::getStreamFromPool(priority, device));
}

HIPStreamMasqueradingAsCUDA
inline getStreamFromExternalMasqueradingAsCUDA(hipStream_t ext_stream, DeviceIndex device) {
  return HIPStreamMasqueradingAsCUDA(c10::cuda::getStreamFromExternal(ext_stream, device));
}

inline HIPStreamMasqueradingAsCUDA getDefaultHIPStreamMasqueradingAsCUDA(DeviceIndex device_index = -1) {
  return HIPStreamMasqueradingAsCUDA(c10::cuda::getDefaultCUDAStream(device_index));
}

inline HIPStreamMasqueradingAsCUDA getCurrentHIPStreamMasqueradingAsCUDA(DeviceIndex device_index = -1) {
  return HIPStreamMasqueradingAsCUDA(c10::cuda::getCurrentCUDAStream(device_index));
}

inline void setCurrentHIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA stream) {
  c10::cuda::setCurrentCUDAStream(stream.hip_stream());
}

inline std::ostream& operator<<(std::ostream& stream, const HIPStreamMasqueradingAsCUDA& s) {
  stream << s.hip_stream() << " (masquerading as CUDA)";
  return stream;
}

}} // namespace c10::hip

namespace std {
  template <>
  struct hash<c10::hip::HIPStreamMasqueradingAsCUDA> {
    size_t operator()(c10::hip::HIPStreamMasqueradingAsCUDA s) const noexcept {
      return std::hash<c10::Stream>{}(s.unwrap());
    }
  };
} // namespace std
