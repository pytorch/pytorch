#pragma once

#include <ATen/hip/HIPConfig.h>

// The includes of HIPGuard.h
#include <c10/hip/impl/HIPGuardImpl.h>
#include <c10/hip/HIPMacros.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include <c10/hip/impl/HIPGuardImpl.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10 { namespace hip {

// Note [Masquerading as CUDA]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// HIPGuardImplMasqueradingAsCUDA is like a HIPGuardImpl, but
// it reports its DeviceType as CUDA (e.g., type() returns CUDA,
// getDevice() reports the current HIP device as a CUDA device.)
// We can't directly use HIPGuardImpl, since it (piously) requires
// the DeviceType to be HIP.
//
// This is necessary for PyTorch at the moment, which is implemented
// by pretending that CUDA is actually HIP.  Eventually, we want
// to make PyTorch treat HIP as a separate DeviceType, and then we
// can delete this class.
//
// Also, note that the cpp file associated with this also *overwrites*
// the entry in the DeviceGuardImpl registry for CUDA with this HIP
// implementation.
struct HIPGuardImplMasqueradingAsCUDA final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::CUDA;
  HIPGuardImplMasqueradingAsCUDA() {}
  HIPGuardImplMasqueradingAsCUDA(DeviceType t) {
    AT_ASSERT(t == DeviceType::CUDA);
  }
  DeviceType type() const override {
    return DeviceType::CUDA;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      C10_HIP_CHECK(hipSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    int device;
    C10_HIP_CHECK(hipGetDevice(&device));
    return Device(DeviceType::CUDA, device);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::CUDA);
    C10_HIP_CHECK(hipSetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    hipSetDevice(d.index());
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentHIPStream().unwrap();
  }
  Stream exchangeStream(Stream s) const noexcept override {
    HIPStream cs(s);
    auto old_stream = getCurrentHIPStream(s.device().index());
    setCurrentHIPStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const override {
    int deviceCnt;
    C10_HIP_CHECK(hipGetDeviceCount(&deviceCnt));
    return deviceCnt;
  }
};

// All of the guards which have HIPGuardImpl burned in need to also have
// variants using HIPGuardImplMasqueradingAsCUDA.

/// This code is all a direct copy from c10/cuda/HIPGuardMasqueradingAsCUDA.h, but with
/// the correct InlineDeviceGuard burned in.  Sorry about the
/// copy-pasting.

struct HIPGuardMasqueradingAsCUDA {
  explicit HIPGuardMasqueradingAsCUDA() = delete;
  explicit HIPGuardMasqueradingAsCUDA(DeviceIndex device_index) : guard_(device_index) {}
  explicit HIPGuardMasqueradingAsCUDA(Device device) : guard_(device) {}

  HIPGuardMasqueradingAsCUDA(const HIPGuardMasqueradingAsCUDA&) = delete;
  HIPGuardMasqueradingAsCUDA& operator=(const HIPGuardMasqueradingAsCUDA&) = delete;
  HIPGuardMasqueradingAsCUDA(HIPGuardMasqueradingAsCUDA&& other) = delete;
  HIPGuardMasqueradingAsCUDA& operator=(HIPGuardMasqueradingAsCUDA&& other) = delete;

  void set_device(Device device) { guard_.set_device(device); }
  void reset_device(Device device) { guard_.reset_device(device); }
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }
  Device original_device() const { return guard_.original_device(); }
  Device current_device() const { return guard_.current_device(); }

 private:
  c10::impl::InlineDeviceGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct OptionalHIPGuardMasqueradingAsCUDA {
  explicit OptionalHIPGuardMasqueradingAsCUDA() : guard_() {}
  explicit OptionalHIPGuardMasqueradingAsCUDA(optional<Device> device_opt) : guard_(device_opt) {}
  explicit OptionalHIPGuardMasqueradingAsCUDA(optional<DeviceIndex> device_index_opt) : guard_(device_index_opt) {}

  OptionalHIPGuardMasqueradingAsCUDA(const OptionalHIPGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPGuardMasqueradingAsCUDA& operator=(const OptionalHIPGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPGuardMasqueradingAsCUDA(OptionalHIPGuardMasqueradingAsCUDA&& other) = delete;
  OptionalHIPGuardMasqueradingAsCUDA& operator=(OptionalHIPGuardMasqueradingAsCUDA&& other) = delete;

  void set_device(Device device) { guard_.set_device(device); }
  void reset_device(Device device) { guard_.reset_device(device); }
  void set_index(DeviceIndex device_index) { guard_.set_index(device_index); }
  optional<Device> original_device() const { return guard_.original_device(); }
  optional<Device> current_device() const { return guard_.current_device(); }
  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalDeviceGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct HIPStreamGuardMasqueradingAsCUDA {
  explicit HIPStreamGuardMasqueradingAsCUDA() = delete;
  explicit HIPStreamGuardMasqueradingAsCUDA(Stream stream) : guard_(stream) {}
  HIPStreamGuardMasqueradingAsCUDA(const HIPStreamGuardMasqueradingAsCUDA&) = delete;
  HIPStreamGuardMasqueradingAsCUDA& operator=(const HIPStreamGuardMasqueradingAsCUDA&) = delete;
  HIPStreamGuardMasqueradingAsCUDA(HIPStreamGuardMasqueradingAsCUDA&& other) = delete;
  HIPStreamGuardMasqueradingAsCUDA& operator=(HIPStreamGuardMasqueradingAsCUDA&& other) = delete;

  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  HIPStream original_stream() const {
    return HIPStream(HIPStream::UNCHECKED, guard_.original_stream());
  }
  HIPStream current_stream() const {
    return HIPStream(HIPStream::UNCHECKED, guard_.current_stream());
  }

  Device current_device() const { return guard_.current_device(); }
  Device original_device() const { return guard_.original_device(); }

private:
  c10::impl::InlineStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

struct OptionalHIPStreamGuardMasqueradingAsCUDA {
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA() : guard_() {}
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA(Stream stream) : guard_(stream) {}
  explicit OptionalHIPStreamGuardMasqueradingAsCUDA(optional<Stream> stream_opt) : guard_(stream_opt) {}

  OptionalHIPStreamGuardMasqueradingAsCUDA(const OptionalHIPStreamGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPStreamGuardMasqueradingAsCUDA& operator=(const OptionalHIPStreamGuardMasqueradingAsCUDA&) = delete;
  OptionalHIPStreamGuardMasqueradingAsCUDA(OptionalHIPStreamGuardMasqueradingAsCUDA&& other) = delete;
  OptionalHIPStreamGuardMasqueradingAsCUDA& operator=(OptionalHIPStreamGuardMasqueradingAsCUDA&& other) = delete;

  void reset_stream(Stream stream) { guard_.reset_stream(stream); }

  optional<HIPStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(HIPStream(HIPStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  optional<HIPStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(HIPStream(HIPStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  void reset() { guard_.reset(); }

private:
  c10::impl::InlineOptionalStreamGuard<HIPGuardImplMasqueradingAsCUDA> guard_;
};

}} // namespace c10::hip
