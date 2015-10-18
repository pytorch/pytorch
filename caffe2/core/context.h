#ifndef CAFFE2_CORE_CONTEXT_H_
#define CAFFE2_CORE_CONTEXT_H_

#include <ctime>
#include <random>

#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/core/logging.h"

namespace caffe2 {

/**
 * The CPU Context, representing the bare minimum of what a Context class in
 * Caffe2 should implement.
 *
 * See opeartor.h, especially Operator<Context>, for how Context are used in
 * actual operator implementations that are associated with specific devices.
 * In general, the Context class is passed in as a template argument, and
 * the operator can use the functions defined in the context to execute whatever
 * computation it has.
 *
 * A Context defines all the necessities to run an operator on a specific
 * device. Specific Context classes have the freedom to choose what functions it
 * implements, but there are a few functions that you should consider
 * implementing if you want to write your own context class:
 * - void SwitchToDevice(): any necessary code to switch to the device before
 *     running anything.
 * - bool FinishDeviceComputation(): any wrapping-up work after all the
 *     computation of the operator is done. If there are errors during the
 *     execution, return false. For example, in a CUDAContext, this function
 *     carries out a stream synchronization and spots potential errors for
 *     the cuda kernel calls.
 * - static void* New(size_t nbytes): allocates memory.
 * - static void Delete(void* data): deletes memory.
 * - template <class SrcContext, class DstContext> void Memcpy(...): does cross
 *     context memory copy.
 * - template <typename T, class SrcContext, class DstContext> void Copy(...):
 *     usually a simple wrapper around the above Memcpy function.
 *
 * We intentionally did not create a base class for the various possible Context
 * classes there might be, since they are intended to be specified during
 * compile time rather than via polymorphism.
 */
class CPUContext {
 public:
  CPUContext() : random_generator_(0) {}
  explicit CPUContext(const DeviceOption& option)
      : random_generator_(
            option.has_random_seed() ? option.random_seed() : time(NULL)) {
    CAFFE_CHECK_EQ(option.device_type(), CPU);
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

// For simplicity, we will typedef Tensor<CPUContext> to TensorCPU.
template <class Context> class Tensor;
typedef Tensor<CPUContext> TensorCPU;

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_H_
