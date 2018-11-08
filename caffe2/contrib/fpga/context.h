#ifndef CAFFE2_OPENCL_CONTEXT_H_
#define CAFFE2_OPENCL_CONTEXT_H_

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"

#define CL_HPP_ENABLE_EXCEPTIONS 1
#define CL_HPP_CL_1_2_DEFAULT_BUILD 1
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define OPENCL_CHECK(expr) (void)expr

namespace caffe2 {

class BaseSingletonContext {
 public:
  BaseSingletonContext() {}
  static BaseSingletonContext* getInstance(const string& engine);

 public:
  cl::Platform platform;
  cl::Context context;
  std::vector<cl::Device> devices;
  cl::Device device;
  std::vector<cl::CommandQueue> queues;
  string engine;
};

class OpenCLContextSingleton : public BaseSingletonContext {
 public:
  OpenCLContextSingleton() : BaseSingletonContext() {}
  static OpenCLContextSingleton& getInstance();

 public:
};

struct OpenCLEventWrapper {
  explicit OpenCLEventWrapper(const DeviceOption& option)
      : ocl_gpu_id_(option.device_id()),
        status_(EventStatus::EVENT_INITIALIZED) {
    CAFFE_ENFORCE(option.device_type(), PROTO_OPENCL);
  }
  ~OpenCLEventWrapper() {}

  int ocl_gpu_id_;
  std::mutex mutex_;
  std::atomic<int> status_;
  std::string err_msg_;
};

class OpenCLContext final : public BaseContext {
 public:
  explicit OpenCLContext() {}
  explicit OpenCLContext(const DeviceOption& option) {
    DCHECK_EQ(option.device_type(), PROTO_OPENCL);
    // TODO: have a better way to pass the engine
    if (option.extra_info().size() > 0) {
      engine_ = option.extra_info(0);
    };
  }
  explicit OpenCLContext(const at::Device& device)
      : OpenCLContext(DeviceToOption(device)) {}
  ~OpenCLContext() {}

  at::Device device() const override {
    return at::Device(OPENCL);
  }

  DeviceType device_type() const override {
    return OPENCL;
  }

  static constexpr DeviceType GetDeviceType() {
    return OPENCL;
  }

  /*
   * Everything below is basically boiler plate for Context classes
   */
  static at::DataPtr New(size_t nbytes, int mode = CL_MEM_READ_WRITE);

  static void Delete(void* data);

  template <class SrcContext, class DstContext>
  void CopyBytes(size_t, const void*, void*);

  void CopyBytesSameDevice(size_t nbytes, const void* src, void* dst) override {
    CAFFE_ENFORCE(0, "Not implemented");
  }

  void CopyBytesToCPU(size_t nbytes, const void* src, void* dst) override;

  void CopyBytesFromCPU(size_t nbytes, const void* src, void* dst) override;

  template <typename T, class SrcContext, class DstContext>
  inline void Copy(int n, const T* src, T* dst) {
    CopyBytes<SrcContext, DstContext>(
        n * sizeof(T), static_cast<const void*>(src), static_cast<void*>(dst));
  }

  template <typename T_in, typename T_out, class SrcContext, class DstContext>
  inline void Copy(const Tensor& src, Tensor& dst) {
    dst.Resize(src.sizes());
    size_t n = src.numel();
    if (std::is_same<T_in, T_out>::value) {
      if (std::is_fundamental<T_in>::value) {
        dst.template mutable_data<T_out>();
        src.template data<T_in>();
        CopyBytes<SrcContext, DstContext>(
            n * sizeof(T_in),
            static_cast<const void*>(src.template data<T_in>()),
            static_cast<void*>(dst.template mutable_data<T_out>()));
      } else {
        for (int i = 0; i < n; ++i) {
          dst.template mutable_data<T_out>()[i] = src.template data<T_in>()[i];
        }
      }
    } else {
      CAFFE_THROW("This Copy requires specialization.");
    }
  }

  template <class SrcContext, class DstContext>
  inline void
  CopyItems(const TypeMeta& meta, size_t n, const void* src, void* dst) {
    CAFFE_ENFORCE(!meta.copy(), "OpenCLContext requires fundamental types.");
    CopyBytes<SrcContext, DstContext>(n * meta.itemsize(), src, dst);
  }

  void SwitchToDevice(int id) override {
    auto& ctx = *GetSingleton(engine_);
    CAFFE_ENFORCE(
        id < ctx.devices.size(),
        "id:",
        id,
        " size:",
        ctx.devices.size(),
        " engine:",
        engine_);
    ctx.device = ctx.devices[id];
  }
  void SwitchToDevice() {
    SwitchToDevice(0);
  }

  inline void WaitEvent(const Event&) { /* TODO */
  }
  void FinishDeviceComputation() {
    auto& ctx = *GetSingleton(engine_);
    for (int i = 0; i < ctx.queues.size(); i++) {
      ctx.queues[i].finish();
    }
  }

  inline void Record(Event*, const char* err_msg = nullptr) const override {
    /* TODO */
  }

  static bool IsStreamFree(const DeviceOption&, int) {
    return true;
  }
  bool HasAsyncPartDefault() const {
    return false;
  }
  bool SupportsAsyncScheduling() const {
    return false;
  }

  // OpenCL specific helper functions
  cl::Kernel BuildKernel(
      const char* src,
      std::string additional_options = "",
      const char* fn_name = "K");

  cl::Kernel BuildKernel(const std::string& binaries);

  cl::Program LoadProgram(const std::string& filename);

  static BaseSingletonContext* GetSingleton(const std::string& engine);
  static std::string BuildArgumentList(
      std::vector<std::pair<std::string, std::string>> args);
  static std::mutex& mutex();

 private:
  std::string engine_;
};

// Get the OpenCL Alloctor.
// We need to keep this to expose the allocate function
// with an extra argument
struct DefaultOpenCLAllocator;
CAFFE2_API DefaultOpenCLAllocator* GetOpenCLAllocator();

using TensorOpenCL = Tensor;

} // namespace caffe2

#endif /* CAFFE2_OPENCL_CONTEXT_H_ */
