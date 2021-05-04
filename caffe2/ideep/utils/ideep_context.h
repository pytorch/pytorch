#pragma once

#include <cstdlib>
#include <ctime>
#include <random>

#include <caffe2/core/context.h>

namespace caffe2 {

class IDEEPContext final : public BaseContext {
 public:
  typedef std::mt19937 rand_gen_type;
  IDEEPContext() : random_seed_(RandomNumberSeed()) {}
  explicit IDEEPContext(const DeviceOption& option)
      : random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : RandomNumberSeed()) {
    CAFFE_ENFORCE_EQ(option.device_type(), PROTO_IDEEP);
  }
  explicit IDEEPContext(const at::Device& device)
      : IDEEPContext(DeviceToOption(device)) {}

  ~IDEEPContext() noexcept override {}

  inline void SwitchToDevice(int /*stream_id*/) {}
  using BaseContext::SwitchToDevice;

  inline void WaitEvent(const Event& ev) {
    ev.Wait(IDEEP, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(IDEEP, this, err_msg);
  }


  inline void FinishDeviceComputation() {}

  inline rand_gen_type& RandGenerator() {
    if (!random_generator_.get()) {
      random_generator_.reset(new rand_gen_type(random_seed_));
    }
    return *random_generator_.get();
  }

  inline static at::DataPtr New(size_t nbytes) {
    return GetAllocator(CPU)->allocate(nbytes);
  }

  void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) override {
    if (nbytes == 0) {
      return;
    }
    CAFFE_ENFORCE(src);
    CAFFE_ENFORCE(dst);
    memcpy(dst, src, nbytes);
  }

  void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytesSameDevice(nbytes, src, dst);
  }

  void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytesSameDevice(nbytes, src, dst);
  }

  bool SupportsNonFundamentalTypes() const override {
    // IDEEP meta copy is OK
    return true;
  }

  // Two copy functions that deals with cross-device copies.
  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst);

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(size_t n, const T* src, T* dst) {
    if (c10::guts::is_fundamental<T>::value) {
      CopyBytes<SrcContext, DstContext>(
          n * sizeof(T),
          static_cast<const void*>(src),
          static_cast<void*>(dst));
    } else {
      for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      meta.copy()(src, dst, n);
    } else {
      CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
    }
  }

  static bool HasAsyncPartDefault() {
    return false;
  }

  static bool SupportsAsyncScheduling() {
    return false;
  }

  static bool IsStreamFree(const DeviceOption& /* unused */, int /* unused */) {
    return true;
  }

  at::Device device() const override {
    return at::Device(IDEEP);
  }

  DeviceType device_type() const override {
    return IDEEP;
  }

  static constexpr DeviceType GetDeviceType() {
    return IDEEP;
  }

 protected:
  // TODO(jiayq): instead of hard-coding a generator, make it more flexible.
  int random_seed_{1701};
  std::unique_ptr<rand_gen_type> random_generator_;
};

template <>
inline void IDEEPContext::CopyBytes<IDEEPContext, IDEEPContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  if (nbytes == 0) {
    return;
  }
  CAFFE_ENFORCE(src);
  CAFFE_ENFORCE(dst);
  memcpy(dst, src, nbytes);
}

template <>
inline void IDEEPContext::CopyBytes<CPUContext, IDEEPContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  if (nbytes == 0) {
    return;
  }
  CAFFE_ENFORCE(src);
  CAFFE_ENFORCE(dst);
  memcpy(dst, src, nbytes);
}

template <>
inline void IDEEPContext::CopyBytes<IDEEPContext, CPUContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  if (nbytes == 0) {
    return;
  }
  CAFFE_ENFORCE(src);
  CAFFE_ENFORCE(dst);
  memcpy(dst, src, nbytes);
}

} // namespace caffe2
