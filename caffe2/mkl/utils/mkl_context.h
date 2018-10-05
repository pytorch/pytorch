#ifndef CAFFE2_UTILS_MKL_CONTEXT_H_
#define CAFFE2_UTILS_MKL_CONTEXT_H_

#include <cstdlib>
#include <ctime>
#include <random>

#include "caffe2/core/context.h"
#include "caffe2/core/context_base.h"

namespace caffe2 {

BaseStaticContext* GetMKLStaticContext();

/**
 * The MKL Context, which is largely the same as the CPUContext. We instantiate
 * this mainly in order to have a first-class MKL device.
 *
 * Note that although New() and Delete() are implemented, we expect MKLContext
 * operators to mainly perform input and output via MKLMemory. As a result,
 * most likely MKLContext::New and ::Delete won't be used as often.
 */
class MKLContext : public BaseContext {
 public:
  MKLContext() : random_seed_(RandomNumberSeed()) {}
  explicit MKLContext(const DeviceOption& option)
      : random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : RandomNumberSeed()) {
    CAFFE_ENFORCE_EQ(option.device_type(), PROTO_MKLDNN);
  }
  explicit MKLContext(const at::Device& device)
      : MKLContext(DeviceToOption(device)) {}

  ~MKLContext() override {}

  BaseStaticContext* GetStaticContext() const override {
    return GetMKLStaticContext();
  }

  static BaseStaticContext* StaticContext() {
    return GetMKLStaticContext();
  }

  inline void SwitchToDevice(int /*stream_id*/ = 0) override {}

  inline void WaitEvent(const Event& ev) override {
    ev.Wait(MKLDNN, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const override {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(MKLDNN, this, err_msg);
  }

  inline void FinishDeviceComputation() override {}

  inline std::mt19937& RandGenerator() {
    if (!random_generator_.get()) {
      random_generator_.reset(new std::mt19937(random_seed_));
    }
    return *random_generator_.get();
  }

  inline static at::DataPtr New(size_t nbytes) {
    return StaticContext()->New(nbytes);
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
    // MKL meta copy is OK
    return true;
  }

  // Two copy functions that deals with cross-device copies.
  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst);

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(size_t n, const T* src, T* dst) {
    if (std::is_fundamental<T>::value) {
      CopyBytes<SrcContext, DstContext>(
          n * sizeof(T),
          static_cast<const void*>(src),
          static_cast<void*>(dst));
    } else {
      for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
      }
    }
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      meta.copy()(src, dst, n);
    } else {
      CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
    }
  }

  // By default MKL operators don't have async device parts
  static bool HasAsyncPartDefault() {
    return false;
  }

  static bool SupportsAsyncScheduling() {
    return false;
  }

  static bool IsStreamFree(const DeviceOption& option, int stream_id) {
    return true;
  }

  DeviceType device_type() const override {
    return MKLDNN;
  }

  static constexpr DeviceType GetDeviceType() {
    return MKLDNN;
  }

 protected:
  // TODO(jiayq): instead of hard-coding a generator, make it more flexible.
  int random_seed_{1701};
  std::unique_ptr<std::mt19937> random_generator_;
};

template <>
inline void MKLContext::CopyBytes<MKLContext, MKLContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  memcpy(dst, src, nbytes);
}

class MKLStaticContext : public BaseStaticContext {
 public:
  inline at::DataPtr New(size_t nbytes) const override {
    return GetCPUAllocator()->allocate(nbytes);
  }

  DeviceType GetDeviceType() override {
    return MKLDNN;
  }

  void ExtractDeviceOption(DeviceOption* device, const void* /*data*/)
      override {
    device->set_device_type(TypeToProto(GetDeviceType()));
  }
};

} // namespace caffe2

#endif // CAFFE2_UTILS_MKL_CONTEXT_H_
