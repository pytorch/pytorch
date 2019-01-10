#include "context.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

CAFFE_KNOWN_TYPE(Tensor<OpenCLContext>);

OpenCLContextSingleton::OpenCLContextSingleton() {
  const auto platform_id = 0;
  const auto device_id = 0;

  auto platforms = std::vector<cl::Platform>();
  cl::Platform::get(&platforms);
  if (platforms.size() == 0 || platform_id >= platforms.size()) {
    CAFFE_THROW("Cannot find platform for OpenCL.");
  }
  platform = platforms[platform_id];

  devices = std::vector<cl::Device>();
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  if (devices.size() == 0 || device_id >= devices.size()) {
    CAFFE_THROW("Cannot find OpenCL compatible device.");
  }
  device = devices[device_id];

  context = cl::Context({device});
  queue = cl::CommandQueue(context, device);
}

OpenCLContextSingleton& OpenCLContextSingleton::getInstance() {
  static OpenCLContextSingleton* instance;
  if (instance == nullptr) {
    instance = new OpenCLContextSingleton();
  }
  return *instance;
}

std::pair<void*, MemoryDeleter> OpenCLContext::New(size_t nbytes) {
  auto& ctx = GetSingleton();
  cl_int err = 0;

  cl::Buffer* buffer = new cl::Buffer(ctx.context, CL_MEM_READ_WRITE,
      nbytes, nullptr, &err);
  OPENCL_CHECK(err);
  // TODO(bwasti): use host ptr if possible to make CopyBytes free
  return std::make_pair((void *)buffer, OpenCLContext::Delete);
}

void OpenCLContext::Delete(void *ptr) {
  delete (cl::Buffer *)ptr;
}

struct OpenCLContextSingleton& OpenCLContext::GetSingleton() {
  return OpenCLContextSingleton::getInstance();
}

cl::Kernel OpenCLContext::BuildKernel(const char* src, std::string additional_options, const char* fn_name) {
  auto& ctx = GetSingleton();

  cl::Program::Sources source(1,
      std::make_pair(src, strlen(src)));

  cl::Program p = cl::Program(ctx.context, source);
  cl_int err = CL_SUCCESS;
  std::string options = "-cl-std=CL1.1 -cl-fast-relaxed-math -cl-single-precision-constant";
  options += additional_options;
  err = p.build(ctx.devices, options.c_str());
  cl_build_status build_status = p.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(ctx.device);
  if (err != CL_SUCCESS || build_status != CL_BUILD_SUCCESS) {
    auto str = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device);
    LOG(ERROR) << str;
    CAFFE_THROW(str);
  }

  auto kernel = cl::Kernel(p, fn_name, &err);
  OPENCL_CHECK(err);
  return kernel;
}

std::string OpenCLContext::BuildArgumentList(std::vector<std::pair<std::string, std::string>> args) {
  std::string out = " "; // There may be args before this
  for (auto arg : args) {
    out += "-D " + arg.first + "=" + arg.second + " ";
  }
  return out;
}

void EventCreateOPENCL(const DeviceOption& /* unused */, Event* /* unused */) {}
void EventRecordOPENCL(
    Event* /* unused */,
    const void* /* unused */,
    const char* /* unused */) {}
void EventWaitOPENCL(const Event* /* unused */, void* /* unused */) {}
void EventFinishOPENCL(const Event* /* unused */) {}
void EventResetOPENCL(Event* /* unused */) {}

REGISTER_EVENT_CREATE_FUNCTION(OPENCL, EventCreateOPENCL);
REGISTER_EVENT_RECORD_FUNCTION(OPENCL, EventRecordOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(OPENCL, OPENCL, EventWaitOPENCL);
REGISTER_EVENT_FINISH_FUNCTION(OPENCL, EventFinishOPENCL);
REGISTER_EVENT_RESET_FUNCTION(OPENCL, EventResetOPENCL);

} // namespace caffe2
