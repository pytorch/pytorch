#ifndef CAFFE2_CORE_CONTEXT_HIP_H_
#define CAFFE2_CORE_CONTEXT_HIP_H_

#include <hiprand.h>
#include <ctime>
#include <mutex>

#include "caffe2/core/context.h"
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/common_miopen.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/numa.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

enum class HipMemoryPoolType {
  NONE = 0,
  CUB = 1,
  THC = 2,
};

/**
 * Gets the current memory pool type used by Caffe2.
 *
 * The memory pool is set up during caffe2's global initialization time.
 */
HipMemoryPoolType GetHipMemoryPoolType();

/**
 * A struct to host thread-local cuda objects.
 *
 * In Caffe2, each thread has its own non-default cuda stream as well as
 * related objects such as cublas and curand handles. This is achieved by
 * having the ThreadLocalCUDAObjects wrapper that takes care of allocating
 * and deallocating these objects at the thread scope. This class is solely
 * used inside CUDAContext and should not be used externally.
 */
class ThreadLocalHIPObjects {
  friend class HIPContext;

 private:
  ThreadLocalHIPObjects() {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_HIP_GPUS; ++i) {
      hip_streams_[i] = vector<hipStream_t>();
      rocblas_handles_[i] = vector<rocblas_handle>();
      miopen_handles_[i] = vector<miopenHandle_t>();
    }
  }

  hipStream_t GetStream(int gpu, int stream_id) {
    vector<hipStream_t>& gpu_streams = hip_streams_[gpu];
    if (gpu_streams.size() <= (unsigned)stream_id) {
      gpu_streams.resize(stream_id + 1, nullptr);
    }
    if (!gpu_streams[stream_id]) {
      DeviceGuard guard(gpu);
      HIP_ENFORCE(hipStreamCreateWithFlags(
          &gpu_streams[stream_id], hipStreamNonBlocking));
    }
    return gpu_streams[stream_id];
  }

  rocblas_handle GetHandle(int gpu, int stream_id) {
    DeviceGuard guard(gpu);
    vector<rocblas_handle>& gpu_handles = rocblas_handles_[gpu];
    if (gpu_handles.size() <= (unsigned)stream_id) {
      gpu_handles.resize(stream_id + 1, nullptr);
    }
    if (!gpu_handles[stream_id]) {
      ROCBLAS_ENFORCE(rocblas_create_handle(&gpu_handles[stream_id]));
      // The default is ROCBLAS_POINTER_MODE_HOST. You can override
      // it after obtaining the rocblas handle, but do that with
      // caution.
      ROCBLAS_ENFORCE(rocblas_set_pointer_mode(
          gpu_handles[stream_id], rocblas_pointer_mode_host));
      ROCBLAS_ENFORCE(rocblas_set_stream(
          gpu_handles[stream_id], GetStream(gpu, stream_id)));
    }
    return gpu_handles[stream_id];
  }

  miopenHandle_t GetMiopenHandle(int gpu, int stream_id) {
    DeviceGuard guard(gpu);
    vector<miopenHandle_t>& gpu_handles = miopen_handles_[gpu];
    if (gpu_handles.size() <= (unsigned)stream_id) {
      gpu_handles.resize(stream_id + 1, nullptr);
    }
    if (!gpu_handles[stream_id]) {
      MIOPEN_ENFORCE(miopenCreate(&gpu_handles[stream_id]));
      MIOPEN_ENFORCE(
          miopenSetStream(gpu_handles[stream_id], GetStream(gpu, stream_id)));
    }
    return gpu_handles[stream_id];
  }

  ~ThreadLocalHIPObjects() noexcept {
    for (int i = 0; i < CAFFE2_COMPILE_TIME_MAX_HIP_GPUS; ++i) {
      for (auto& handle : rocblas_handles_[i]) {
        if (handle) {
          ROCBLAS_CHECK(rocblas_destroy_handle(handle));
        }
      }
      for (auto& stream : hip_streams_[i]) {
        if (stream) {
          HIP_CHECK(hipStreamDestroy(stream));
        }
      }
      for (auto& handle : miopen_handles_[i]) {
        if (handle) {
          MIOPEN_CHECK(miopenDestroy(handle));
        }
      }
    }
  }
  vector<hipStream_t> hip_streams_[CAFFE2_COMPILE_TIME_MAX_HIP_GPUS];
  vector<rocblas_handle> rocblas_handles_[CAFFE2_COMPILE_TIME_MAX_HIP_GPUS];
  vector<miopenHandle_t> miopen_handles_[CAFFE2_COMPILE_TIME_MAX_HIP_GPUS];
};

BaseStaticContext* GetHIPStaticContext();

class HIPContext final : public BaseContext {
 public:
  // The default HIP context constructor.
  explicit HIPContext(const int gpu_id = -1);
  explicit HIPContext(const DeviceOption& option);
  explicit HIPContext(const at::Device& device)
      : HIPContext(DeviceToOption(device)) {}

  ~HIPContext() override {
    if (hiprand_generator_) {
      HIPRAND_CHECK(hiprandDestroyGenerator(hiprand_generator_));
    }
    FinishDeviceComputation();
  }

  BaseStaticContext* GetStaticContext() const override {
    return GetHIPStaticContext();
  }

  static BaseStaticContext* StaticContext() {
    return GetHIPStaticContext();
  }

  inline void SwitchToDevice(int stream_id) override {
    set_stream_id(stream_id);
    CaffeHipSetDevice(gpu_id_);
  }

  using BaseContext::SwitchToDevice;

  inline void WaitEvent(const Event& ev) override {
    ev.Wait(HIP, this);
  }

  inline void Record(Event* ev, const char* err_msg = nullptr) const override {
    CAFFE_ENFORCE(ev, "Event must not be null.");
    ev->Record(HIP, this, err_msg);
  }

  void FinishDeviceComputation() override {
    hipStreamSynchronize(hip_objects_.GetStream(gpu_id_, stream_id_));
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
      CAFFE_THROW("Encountered HIP error: ", hipGetErrorString(error));
    }
  }

  inline int hip_gpu_id() const {
    return gpu_id_;
  }

  inline hipStream_t hip_stream() {
    return hip_stream(gpu_id_, stream_id_);
  }

  inline hipStream_t hip_stream() const {
    return hip_stream(gpu_id_, stream_id_);
  }

  static hipStream_t hip_stream(int gpu_id, int stream_id) {
    return hip_objects_.GetStream(gpu_id, stream_id);
  }

  rocblas_handle rocblas_handle() {
    return hip_objects_.GetHandle(gpu_id_, stream_id_);
  }

  miopenHandle_t miopen_handle() {
    return hip_objects_.GetMiopenHandle(gpu_id_, stream_id_);
  }

  hiprandGenerator_t& hiprand_generator() {
    if (!hiprand_generator_) {
      DeviceGuard guard(gpu_id_);
      HIPRAND_ENFORCE(hiprandCreateGenerator(
          &hiprand_generator_, HIPRAND_RNG_PSEUDO_DEFAULT));
      HIPRAND_ENFORCE(hiprandSetPseudoRandomGeneratorSeed(
          hiprand_generator_, random_seed_));
      CHECK_NOTNULL(hiprand_generator_);
    }
    HIPRAND_ENFORCE(hiprandSetStream(hiprand_generator_, hip_stream()));
    return hiprand_generator_;
  }

  static at::DataPtr New(size_t nbytes) {
    return StaticContext()->New(nbytes);
  }

  // Get a mutex to lock out hipMalloc / hipFree calls when
  // NCCL kernels are being launched. Should remove threat of
  // deadlocks
  static std::mutex& mutex();

  // Functions to query memory stats. Only available if flag
  // --caffe2_gpu_memory_tracking is enabled.
  static std::vector<long> TotalMemoryByGpu();
  static std::vector<long> MaxMemoryByGpu();

  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst) {
    if (nbytes == 0)
      return;
    HIP_ENFORCE(hipMemcpyAsync(
        dst,
        src,
        nbytes,
        hipMemcpyDefault,
        hip_objects_.GetStream(gpu_id_, stream_id_)));
  }

  void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) override {
    CopyBytes<HIPContext, HIPContext>(nbytes, src, dst);
  }

  void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytes<HIPContext, CPUContext>(nbytes, src, dst);
  }

  void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) override {
    CopyBytes<CPUContext, HIPContext>(nbytes, src, dst);
  }

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    CopyBytes<SrcContext, DstContext>(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "HIPContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  // By default HIP operators have async device parts
  static bool HasAsyncPartDefault() {
    return true;
  }

  static bool SupportsAsyncScheduling() {
    return true;
  }

  static bool IsStreamFree(const DeviceOption& option, int stream_id) {
    auto stream = HIPContext::hip_stream(option.hip_gpu_id(), stream_id);
    return hipStreamQuery(stream) == hipSuccess;
  }

  DeviceType device_type() const override {
    return HIP;
  }

  static constexpr DeviceType GetDeviceType() {
    return HIP;
  }

 protected:
  static void Delete(void* data);
  void set_stream_id(int stream_id) {
    stream_id_ = stream_id;
  }

  int gpu_id_;
  int stream_id_ = 0;
  int random_seed_;
  hiprandGenerator_t hiprand_generator_{nullptr};
  static thread_local ThreadLocalHIPObjects hip_objects_;
};

// For the CPU context, we also allow a (probably expensive) function
// to copy the data from a HIP context. Inside the function, we create
// a temporary CUDAContext object to carry out the copy. From the caller's
// side, these functions are synchronous with respect to the host, similar
// to a normal CPUContext::CopyBytes<CPUContext, CPUContext> call.
template <>
inline void CPUContext::CopyBytes<HIPContext, CPUContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  HIPContext context(GetGPUIDForPointer(src));
  context.CopyBytes<HIPContext, CPUContext>(nbytes, src, dst);
}
template <>
inline void CPUContext::CopyBytes<CPUContext, HIPContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  HIPContext context(GetGPUIDForPointer(dst));
  context.CopyBytes<CPUContext, HIPContext>(nbytes, src, dst);
}

/**
 * An allocator that does the CPU memory allocation with pinned memory.
 *
 * This is needed because if we want to do any asynchronous HIP memcpy,
 * the underlying CPU memory also needs to be allocated into pinned memory
 * space. As a result, whenever Caffe2 is built with GPU and there is
 * GPU present during runtime, at global initialization time we will set
 * the CPU memory allocator to allocate pinned memory.
 */
struct PinnedCPUAllocator final : public at::Allocator {
  PinnedCPUAllocator() {}
  ~PinnedCPUAllocator() override {}
  at::DataPtr allocate(size_t nbytes) const override {
    void* data;
    at::DataPtr data_ptr;
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    if (IsNUMAEnabled()) {
      data_ptr = baseAllocator_.allocate(nbytes);
      data = data_ptr.get();
      CAFFE_ENFORCE(data);
      HIP_ENFORCE(hipHostRegister(data, nbytes, hipHostRegisterDefault));
    } else {
      HIP_ENFORCE(hipHostMalloc(&data, nbytes));
      data_ptr = {data, data, &Delete, at::Device(CPU)};
    }
    memset(data, 0, nbytes);
    return data_ptr;
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

 private:
  static void Delete(void* data) {
    // Caffe2 uses a lazy way to figure out if one is actually going to use GPUs
    // or not. If a HIPContext::New() call is made, inside the CUDAContext
    // function we will switch the cpu side allocator to a PinnedCPUAllocator.
    // But, if one calls CPUContext::New() before any HIP allocations,
    // PinnedCPUAllocator can still delete the corresponding memory.
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    if (IsNUMAEnabled()) {
      HIP_ENFORCE(hipHostUnregister(data));
      DefaultCPUAllocator::Delete(data);
    } else {
      hipError_t err = hipHostFree(data);
      if (err == hipErrorInvalidValue) {
        free(data);
        // Calling hipGetLastError will reset the cuda error.
        hipGetLastError();
      } else {
        // For all other errors, still do a hip check.
        HIP_ENFORCE(err);
      }
    }
  }

  DefaultCPUAllocator baseAllocator_;
};

class HIPStaticContext final : public BaseStaticContext {
 public:
  at::DataPtr New(size_t nbytes) const override;

  DeviceType GetDeviceType() override {
    return HIP;
  }

  void ExtractDeviceOption(DeviceOption* device, const void* data) override {
    device->set_device_type(TypeToProto(GetDeviceType()));
    device->set_hip_gpu_id(GetGPUIDForPointer(data));
  }

};

// Get the HIP Alloctor.
CAFFE2_API at::Allocator* GetHIPAllocator();

typedef Tensor TensorHIP;

} // namespace caffe2

#endif // CAFFE2_CORE_CONTEXT_HIP_H_
