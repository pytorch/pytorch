#pragma once

#include <c10/hip/HIPStream.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// See Note [Masquerading as CUDA] for motivation

class HIPStreamMasqueradingAsCUDA {
public:

  enum Unchecked { UNCHECKED };

  explicit HIPStreamMasqueradingAsCUDA(Stream stream) {
    AT_CHECK(stream_.device_type() == DeviceType::CUDA /* !!! */);
    // Unsafely coerce the "CUDA" stream into a HIP stream
    stream_ = HIPStream(Stream(Stream::UNSAFE, Device(DeviceType::HIP, stream.device_index()), stream.id()));
  }

  explicit HIPStreamMasqueradingAsCUDA(Unchecked, Stream stream) : stream_(stream) {
    // Unsafely coerce the "CUDA" stream into a HIP stream
    stream_ = HIPStream(Stream(Stream::UNSAFE, Device(DeviceType::HIP, stream.device_index()), stream.id()));
  }

  // New constructor, just for this.  Does NOT coerce.
  explicit HIPStreamMasqueradingAsCUDA(HIPStream stream) : stream_(stream) {}

  bool operator==(const HIPStreamMasqueradingAsCUDA& other) const noexcept {
    return stream_ == other.stream_;
  }

  bool operator!=(const HIPStreamMasqueradingAsCUDA& other) const noexcept {
    return stream_ != other.stream_;
  }

  operator hipStream_t() const { return stream_.stream(); }

  operator Stream() const {
    // Unsafely coerce HIP stream into a "CUDA" stream
    return Stream(Stream::UNSAFE, device(), id());
  }

  DeviceIndex device_index() const { return stream_.device_index(); }

  Device device() const {
    // Unsafely coerce HIP device into CUDA device
    return Device(DeviceType::HIP, stream_.device_index());
  }

  StreamId id() const        { return stream_.id(); }
  bool query() const         { return stream_.query(); }
  void synchronize() const   { stream_.synchronize(); }
  int priority() const       { return stream_.priority(); }
  hipStream_t stream() const { return stream_.stream(); }

  Stream unwrap() const {
    // Unsafely coerce HIP stream into "CUDA" stream
    return Stream(Stream::UNSAFE, device(), id());
  }

  uint64_t pack() const noexcept {
    // Unsafely coerce HIP stream into "CUDA" stream before packing
    return unwrap().pack();
  }

  static HIPStreamMasqueradingAsCUDA unpack(uint64_t bits) {
    // NB: constructor manages CUDA->HIP translation for us
    return HIPStreamMasqueradingAsCUDA(Stream::unpack(bits));
  }

  static std::tuple<int, int> priority_range() { return HIPStream::priority_range(); }

  // New method, gets the underlying HIPStream
  HIPStream hip_stream() const { return stream_; }

private:
  HIPStream stream_;
};

HIPStreamMasqueradingAsCUDA
getStreamFromPoolMasqueradingAsCUDA(const bool isHighPriority = false, DeviceIndex device = -1) {
  return HIPStreamMasqueradingAsCUDA(getStreamFromPool(isHighPriority, device));
}

HIPStreamMasqueradingAsCUDA getDefaultHIPStreamMasqueardingAsCUDA(DeviceIndex device_index = -1) {
  return HIPStreamMasqueradingAsCUDA(getDefaultHIPStream(device_index));
}

HIPStreamMasqueradingAsCUDA getCurrentHIPStreamMasqueradingAsCUDA(DeviceIndex device_index = -1) {
  return HIPStreamMasqueradingAsCUDA(getCurrentHIPStream(device_index));
}

void setCurrentHIPStreamMasqueradingAsCUDA(HIPStreamMasqueradingAsCUDA stream) {
  setCurrentHIPStream(stream.hip_stream());
}

std::ostream& operator<<(std::ostream& stream, const HIPStreamMasqueradingAsCUDA& s) {
  stream << s.hip_stream() << " (masquerading as CUDA)";
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
