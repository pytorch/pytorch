#ifndef CAFFE2_CORE_CONTEXT_H_
#define CAFFE2_CORE_CONTEXT_H_

#include <cstdlib>
#include <ctime>
#include <random>
#include <unordered_map>

#include "caffe2/core/logging.h"
#include "caffe2/core/typeid.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/math.h"

CAFFE2_DECLARE_bool(caffe2_report_cpu_memory_usage);

namespace caffe2 {

// Use 32-byte alignment should be enough for computation up to AVX512.
constexpr size_t gCaffe2Alignment = 32;

using MemoryDeleter = std::function<void(void* ptr)>;

// A virtual allocator class to do memory allocation and deallocation.
struct CPUAllocator {
  CPUAllocator() {}
  virtual ~CPUAllocator() noexcept {}
  virtual std::pair<void*, MemoryDeleter> New(size_t nbytes) = 0;
};

// A virtual struct that is used to report Caffe2's memory allocation and
// deallocation status
class MemoryAllocationReporter {
 public:
  MemoryAllocationReporter() : allocated_(0) {}
  void New(void* ptr, size_t nbytes);
  void Delete(void* ptr);

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_;
};

struct DefaultCPUAllocator final : CPUAllocator {
  DefaultCPUAllocator() {}
  ~DefaultCPUAllocator() override {}
  std::pair<void*, MemoryDeleter> New(size_t nbytes) override {
    void* data = nullptr;
#ifdef __ANDROID__
    data = memalign(gCaffe2Alignment, nbytes);
#elif defined(_MSC_VER)
    data = _aligned_malloc(nbytes, gCaffe2Alignment);
#else
    CAFFE_ENFORCE_EQ(posix_memalign(&data, gCaffe2Alignment, nbytes), 0);
#endif
    CAFFE_ENFORCE(data);
    memset(data, 0, nbytes);
    return {data, Delete};
  }
#ifdef _MSC_VER
  static void Delete(void* data) {
    _aligned_free(data);
  }
#else
  static void Delete(void* data) {
    free(data);
  }
#endif
};

typedef std::mt19937 rand_gen_type;

// Get the CPU Alloctor.
CPUAllocator* GetCPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
void SetCPUAllocator(CPUAllocator* alloc);

/**
 * The CPU Context, representing the bare minimum of what a Context class in
 * Caffe2 should implement.
 *
 * See operator.h, especially Operator<Context>, for how Context are used in
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
 * - void FinishDeviceComputation(): any wrapping-up work after all the
 *     computation of the operator is done. If there are errors during the
 *     execution, throw exception. For example, in a CUDAContext, this function
 *     carries out a stream synchronization and spots potential errors for
 *     the cuda kernel calls.
 * - static std::pair<void*, MemoryDeleter> New(size_t nbytes): allocates
       memory and returns a deleter.
 * - template <class SrcContext, class DstContext> void CopyBytes(...): does
 *     cross context memory copy.
 * - template <typename T, class SrcContext, class DstContext> void Copy(...):
 *     usually a simple wrapper around the above CopyBytes function.
 *
 * We intentionally did not create a base class for the various possible Context
 * classes there might be, since they are intended to be specified during
 * compile time using templates rather than via polymorphism. You should also
 * not have classes derived from existing context classes.
 */
class CPUContext final {
 public:
  CPUContext() : random_seed_(math::randomNumberSeed()) {}
  explicit CPUContext(const DeviceOption& option)
      : random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : math::randomNumberSeed()) {
    CAFFE_ENFORCE_EQ(option.device_type(), CPU);
  }

  ~CPUContext() noexcept {}

  inline void SwitchToDevice(int /*stream_id*/) {}
  inline void SwitchToDevice() {
    SwitchToDevice(0);
  }

  inline void FinishDeviceComputation() {}

  inline rand_gen_type& RandGenerator() {
    if (!random_generator_.get()) {
      random_generator_.reset(new rand_gen_type(random_seed_));
    }
    return *random_generator_.get();
  }

  static std::pair<void*, MemoryDeleter> New(size_t nbytes) {
    auto data_and_deleter = GetCPUAllocator()->New(nbytes);
    if (FLAGS_caffe2_report_cpu_memory_usage) {
      reporter_.New(data_and_deleter.first, nbytes);
      auto original_deleter = data_and_deleter.second;
      data_and_deleter.second = [original_deleter](void* data) {
        reporter_.Delete(data);
        original_deleter(data);
      };
    }
    return data_and_deleter;
  }

  // Two copy functions that deals with cross-device copies.
  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst);

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(size_t n, const T* src, T* dst) {
    if (std::is_fundamental<T>::value) {
      CopyBytes<SrcContext, DstContext>(n * sizeof(T),
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

 protected:
  // TODO(jiayq): instead of hard-coding a generator, make it more flexible.
  int random_seed_{1701};
  std::unique_ptr<rand_gen_type> random_generator_;
  static MemoryAllocationReporter reporter_;
};

template<>
inline void CPUContext::CopyBytes<CPUContext, CPUContext>(
    size_t nbytes, const void* src, void* dst) {
  if (nbytes == 0) {
    return;
  }
  CAFFE_ENFORCE(src);
  CAFFE_ENFORCE(dst);
  memcpy(dst, src, nbytes);
}

}  // namespace caffe2

#endif  // CAFFE2_CORE_CONTEXT_H_
