#ifndef CAFFE2_CORE_CONTEXT_H_
#define CAFFE2_CORE_CONTEXT_H_

#include <cstdlib>
#include <ctime>
#include <random>
#include <unordered_map>

#include <c10/util/typeid.h>
#include "caffe2/core/allocator.h"
#include "caffe2/core/context_base.h"
#include "caffe2/core/event.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"

#include <c10/util/ArrayRef.h>

C10_DECLARE_bool(caffe2_report_cpu_memory_usage);

namespace caffe2 {

/**
 * A function to generate a random number seed that is unique in a best-effort
 * basis, using an ever-incrementing seed and the current time.
 */
CAFFE2_API uint32_t RandomNumberSeed();

/**
 * The CPU Context, representing the bare minimum of what a Context class in
 * Caffe2 should implement.
 *
 * // TODO modify docs
 * See operator.h, especially Operator<Context>, for how Context are used in
 * actual operator implementations that are associated with specific devices.
 * In general, the Context class is passed in as a template argument, and
 * the operator can use the functions defined in the context to execute whatever
 * computation it has.
 *
 */
class CAFFE2_API CPUContext final : public BaseContext {
 public:
  typedef std::mt19937 rand_gen_type;
  CPUContext() {}
  explicit CPUContext(const DeviceOption& option)
      : random_seed_(option.has_random_seed() ? option.random_seed() : 1701),
        random_seed_set_(option.has_random_seed() ? true : false) {
    CAFFE_ENFORCE_EQ(option.device_type(), PROTO_CPU);
  }
  explicit CPUContext(const at::Device& device)
      : CPUContext(DeviceToOption(device)) {}

  ~CPUContext() noexcept override {}

  inline void SwitchToDevice(int /*stream_id*/) override {}

  using BaseContext::SwitchToDevice;

  inline void WaitEvent(const Event& ev) override {
    ev.Wait(CPU, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const override {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(CPU, this, err_msg);
  }

  inline void FinishDeviceComputation() override {}

  inline rand_gen_type& RandGenerator() {
    if (!random_generator_.get()) {
      random_generator_.reset(new rand_gen_type(RandSeed()));
    }
    return *random_generator_.get();
  }

  inline uint32_t RandSeed() {
    if (!random_seed_set_) {
      random_seed_ = RandomNumberSeed();
      random_seed_set_ = true;
    }
    return static_cast<uint32_t>(random_seed_);
  }

  inline static at::DataPtr New(size_t nbytes) {
    return GetCPUAllocator()->allocate(nbytes);
  }

  void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) override;

  void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytesSameDevice(nbytes, src, dst);
  }

  void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytesSameDevice(nbytes, src, dst);
  }

  bool SupportsNonFundamentalTypes() const override {
    // CPU non fumdamental type copy OK
    return true;
  }

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
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    if (meta.copy()) {
      meta.copy()(src, dst, n);
    } else {
      CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
    }
  }

  // By default CPU operators don't have async device parts
  static bool HasAsyncPartDefault() {
    return false;
  }

  static bool SupportsAsyncScheduling() {
    return false;
  }

  // CPU streams are not implemented and are silently ignored by CPU ops,
  // return true to signal executor to schedule a CPU op
  static bool IsStreamFree(
      const DeviceOption& /* option */,
      int /* stream_id */) {
    return true;
  }

  at::Device device() const override {
    // TODO: numa?
    return at::Device(CPU);
  }

  DeviceType device_type() const override {
    return CPU;
  }

  static constexpr DeviceType GetDeviceType() {
    return CPU;
  }

 protected:
  // TODO(jiayq): instead of hard-coding a generator, make it more flexible.
  int random_seed_{1701};
  bool random_seed_set_{false};
  std::unique_ptr<rand_gen_type> random_generator_;
};

template <>
inline void CPUContext::CopyBytes<CPUContext, CPUContext>(
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

#endif // CAFFE2_CORE_CONTEXT_H_
