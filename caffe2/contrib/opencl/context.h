#ifndef CAFFE2_OPENCL_CONTEXT_H_
#define CAFFE2_OPENCL_CONTEXT_H_

#include "caffe2/core/context.h"

#define CL_HPP_ENABLE_EXCEPTIONS 1
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#include "libopencl.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define OPENCL_CHECK(expr) (void)expr

namespace caffe2 {

struct OpenCLContextSingleton {
 private:
  OpenCLContextSingleton();
  OpenCLContextSingleton(const OpenCLContextSingleton &) = delete;
  OpenCLContextSingleton(OpenCLContextSingleton&&) = delete;
 public:
  static OpenCLContextSingleton& getInstance();
  cl::Platform platform;
  cl::Device device;
  std::vector<cl::Device> devices;
  cl::Context context;
  cl::CommandQueue queue;
};

class OpenCLContext final {
 public:
  explicit OpenCLContext();
  explicit OpenCLContext(const DeviceOption& option) {
    DCHECK_EQ(option.device_type(), PROTO_OPENCL);
    OpenCLContext();
  }
  ~OpenCLContext() {}

  /*
   * Everything below is basically boiler plate for Context classes
   */
  static std::pair<void*, MemoryDeleter> New(size_t nbytes);

  static void Delete(void* data);

  template <class SrcContext, class DstContext>
  inline void CopyBytes(size_t nbytes, const void* src, void* dst) {}

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    CopyBytes<SrcContext, DstContext>(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "OpenCLContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  void SwitchToDevice(int a, ...) {
    auto& ctx = GetSingleton();
    CAFFE_ENFORCE(a < ctx.devices.size());
    ctx.device = ctx.devices[a];
  }
  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  inline void WaitEvent(const Event& ev) { /* TODO */
  }
  void FinishDeviceComputation() {
    auto& ctx = GetSingleton();
    ctx.queue.finish();
  }

  inline void Record(Event* ev, const char*&) const { /* TODO */
  }
  static bool IsStreamFree(const DeviceOption& /* unused */, int /* unused */) {
    return true;
  }
  bool HasAsyncPartDefault() const {
    return false;
  }
  bool SupportsAsyncScheduling() const {
    return false;
  }

  // OpenCL specific helper functions
  cl::Kernel BuildKernel(const char* src, std::string additional_options = "", const char* fn_name = "K");
  static struct OpenCLContextSingleton& GetSingleton();
  static std::string BuildArgumentList(std::vector<std::pair<std::string, std::string>> args);
};


} // namespace caffe2

#endif /* CAFFE2_OPENCL_CONTEXT_H_ */
