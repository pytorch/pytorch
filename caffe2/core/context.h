#ifndef CAFFE2_CORE_CONTEXT_H_
#define CAFFE2_CORE_CONTEXT_H_

#include <random>

#include "caffe2/proto/caffe2.pb.h"
#include "glog/logging.h"

namespace caffe2 {

class CPUContext {
 public:
  CPUContext() : random_generator_(0) {}
  explicit CPUContext(const DeviceOption& device_option)
      : random_generator_(device_option.random_seed()) {
    DCHECK_EQ(device_option.device_type(), CPU);
  }
  virtual ~CPUContext() {}
  inline void SwitchToDevice() {}
  inline bool FinishDeviceComputation() { return true; }

  inline std::mt19937& RandGenerator() { return random_generator_; }

  static void* New(size_t nbytes) {
    void* data = new char[nbytes];
    // memset(data, 0, nbytes);
    return data;
  }
  static void Delete(void* data) { delete[] static_cast<char*>(data); }

  // Two copy functions that deals with cross-device copies.
  template <class SrcContext, class DstContext>
  inline void Memcpy(size_t nbytes, const void* src, void* dst);
  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    Memcpy<SrcContext, DstContext>(n * sizeof(T),
                                   static_cast<const void*>(src),
                                   static_cast<void*>(dst));
  }

 protected:
  std::mt19937 random_generator_;
};

template<>
inline void CPUContext::Memcpy<CPUContext, CPUContext>(
    size_t nbytes, const void* src, void* dst) {
  memcpy(dst, src, nbytes);
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_H_
