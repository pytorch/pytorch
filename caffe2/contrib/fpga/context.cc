#include "context.h"
#include "context_intel_fpga.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/event_cpu.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

using namespace std;

namespace at {

REGISTER_CONTEXT(DeviceType::OPENCL, caffe2::OpenCLContext);
} // namespace at

namespace caffe2 {

struct DefaultOpenCLAllocator final : public at::Allocator {
  DefaultOpenCLAllocator() {}
  ~DefaultOpenCLAllocator() override {}
  at::DataPtr allocate(size_t nbytes, int mode) const {
    // Lock the mutex
    std::lock_guard<std::mutex> lock(OpenCLContext::mutex());

    FPGAContextSingleton ctx = *static_cast<FPGAContextSingleton*>(
        OpenCLContext::GetSingleton("FPGA"));
    cl_int err = 0;

    CAFFE_ENFORCE(nbytes % 2 == 0);
    cl::Buffer* buffer =
        new cl::Buffer(ctx.context, mode, nbytes / 2, nullptr, &err);
    OPENCL_CHECK(err);
    return {buffer, buffer, &Delete, at::Device(OPENCL)};
  }
  at::DataPtr allocate(size_t nbytes) const override {
    return allocate(nbytes, CL_MEM_READ_WRITE);
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

 private:
  static void Delete(void* ptr) {
    std::lock_guard<std::mutex> lock(OpenCLContext::mutex());
    delete (cl::Buffer*)ptr;
  }
};

static std::unique_ptr<DefaultOpenCLAllocator> g_opencl_allocator(
    new DefaultOpenCLAllocator());
DefaultOpenCLAllocator* GetOpenCLAllocator() {
  return g_opencl_allocator.get();
}

// Implementation of generic OpenCL methods as well as OpenCLContext
BaseSingletonContext* BaseSingletonContext::getInstance(const string& engine) {
  std::lock_guard<std::mutex> guard(FPGAContextSingleton::mutex());
  if (engine == "") {
    return &OpenCLContextSingleton::getInstance();
  } else if (engine == "FPGA") {
    return &FPGAContextSingleton::getInstance();
  } else {
    CAFFE_THROW("Unsupported engine: " + engine);
  }
}

OpenCLContextSingleton& OpenCLContextSingleton::getInstance() {
  static OpenCLContextSingleton* instance;
  if (instance == nullptr) {
    instance = new OpenCLContextSingleton();
  }
  return *instance;
}

BaseSingletonContext* OpenCLContext::GetSingleton(const std::string& engine) {
  return BaseSingletonContext::getInstance(engine);
}

// OpenCLContext
at::DataPtr OpenCLContext::New(size_t nbytes, int mode) {
  // Can't use registry because we need to call the allocate function
  // with both arguments
  return GetOpenCLAllocator()->allocate(nbytes, mode);
}

template <>
void OpenCLContext::CopyBytes<OpenCLContext, CPUContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  auto& ctx = *GetSingleton(engine_);

  ctx.queues.begin()->enqueueReadBuffer(
      *((const cl::Buffer*)src), true, 0, nbytes, dst);
}

template <>
void OpenCLContext::CopyBytes<CPUContext, OpenCLContext>(
    size_t nbytes,
    const void* src,
    void* dst) {
  auto& ctx = *GetSingleton(engine_);

  // use queues 0-2 for inputs, 3 for outputs only
  ctx.queues.back().enqueueWriteBuffer(
      *((const cl::Buffer*)dst), true, 0, nbytes, src);
}

void OpenCLContext::CopyBytesToCPU(size_t nbytes, const void* src, void* dst) {
  CopyBytes<OpenCLContext, CPUContext>(nbytes, src, dst);
}

void OpenCLContext::CopyBytesFromCPU(
    size_t nbytes,
    const void* src,
    void* dst) {
  CopyBytes<CPUContext, OpenCLContext>(nbytes, src, dst);
}

void OpenCLContext::Delete(void* ptr) {
  delete (cl::Buffer*)ptr;
}

cl::Kernel OpenCLContext::BuildKernel(
    const char* src,
    std::string additional_options,
    const char* fn_name) {
  auto& ctx = *GetSingleton(engine_);

  cl::Program::Sources source(1, std::make_pair(src, strlen(src)));

  cl::Program p = cl::Program(ctx.context, source);
  cl_int err = CL_SUCCESS;
  std::string options =
      "-cl-std=CL1.1 -cl-fast-relaxed-math -cl-single-precision-constant";
  options += additional_options;
  err = p.build(ctx.devices, options.c_str());
  cl_build_status build_status =
      p.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(ctx.device);
  if (err != CL_SUCCESS || build_status != CL_BUILD_SUCCESS) {
    auto str = p.getBuildInfo<CL_PROGRAM_BUILD_LOG>(ctx.device);
    LOG(ERROR) << str;
    CAFFE_THROW(str);
  }

  auto kernel = cl::Kernel(p, fn_name, &err);
  OPENCL_CHECK(err);
  return kernel;
}

cl::Program OpenCLContext::LoadProgram(const std::string& filename) {
  auto& ctx = *GetSingleton(engine_);

  cl::Program::Binaries code;

  FILE* fp = fopen(filename.c_str(), "rb");

  if (fp == nullptr) {
    LOG(ERROR) << "Failed to open program file " << filename;
    CAFFE_THROW("no program");
  }

  fseek(fp, 0, SEEK_END);
  int binary_length = ftell(fp);
  const unsigned char* binary =
      (unsigned char*)malloc(sizeof(unsigned char) * binary_length);
  assert(binary && "Malloc failed");
  rewind(fp);

  if (fread((void*)binary, binary_length, 1, fp) == 0) {
    LOG(ERROR) << "Failed to read from the AOCX file (fread)";
    CAFFE_THROW("no program");
  }

  fclose(fp);

  code.push_back(std::make_pair<const void*, size_t>(binary, binary_length));

  cl::Program p = cl::Program(ctx.context, ctx.devices, code);
  return p;
}

std::string OpenCLContext::BuildArgumentList(
    std::vector<std::pair<std::string, std::string>> args) {
  std::string out = " "; // There may be args before this
  for (auto arg : args) {
    out += "-D " + arg.first + "=" + arg.second + " ";
  }
  return out;
}

std::mutex& OpenCLContext::mutex() {
  static std::mutex m;
  return m;
}

// Event Registration
void EventCreateOPENCL(const DeviceOption& devopt, Event* event) {
  CAFFE_ENFORCE_EQ(devopt.device_type(), PROTO_OPENCL);
  event->event_ = std::make_shared<OpenCLEventWrapper>(devopt);
  LOG(INFO) << "event create " << event->event_;
}

void EventRecordOPENCL(
    Event* event,
    const void* /* context */,
    const char* err_msg) {
  LOG(INFO) << "event record " << err_msg;
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  std::unique_lock<std::mutex> lock(wrapper->mutex_);

  // Possible state changes:
  //  INITIALIZED -> SCHEDULED or SUCCESS/FAILED
  //  SCHEDULED -> SUCCESS/FAILED
  //  SUCCESS/FAILED - terminal, no further changes to status_/err_msg_

  CAFFE_ENFORCE_EQ(
      wrapper->status_,
      EventStatus::EVENT_INITIALIZED,
      "Calling Record multiple times");

  if (!err_msg) {
    wrapper->status_ = EventStatus::EVENT_SCHEDULED;
  } else {
    wrapper->err_msg_ = err_msg;
    wrapper->status_ = EventStatus::EVENT_FAILED;
  }
}

void EventFinishOPENCL(const Event* event) {
  LOG(INFO) << "event finish";
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_);
    LOG(INFO) << "event finish " << wrapper->status_;
    while (wrapper->status_ == EventStatus::EVENT_INITIALIZED) {
      usleep(100000);
    }
  }
  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    // ok, even if event is already completed and status was not yet updated
    std::unique_lock<std::mutex> lock(wrapper->mutex_);
    wrapper->status_ = EventStatus::EVENT_SUCCESS;
  }
  LOG(INFO) << "returning finish";
}

void EventWaitOPENCL(const Event* /* event */, void* /* unused */) {
  LOG(INFO) << "===event wait";
}

void EventWaitOPENCLCPU(const Event* event, void* /* unused */) {
  LOG(INFO) << "===event wait opencl cpu";
  event->Finish();
}

void EventWaitCPUOPENCL(const Event* event, void* /* unused */) {
  LOG(INFO) << "event wait cpu opencl";
  EventFinishOPENCL(event);
}

void EventSetFinishedOPENCL(const Event* event, const char* err_msg) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  {
    LOG(INFO) << "event finished " << wrapper << " " << err_msg;
    std::unique_lock<std::mutex> lock(wrapper->mutex_);
    CAFFE_ENFORCE_EQ(
        wrapper->status_,
        EventStatus::EVENT_INITIALIZED,
        "Calling SetFinished on recorded CUDA event");

    if (!err_msg) {
      wrapper->status_ = EventStatus::EVENT_SUCCESS;
    } else {
      wrapper->err_msg_ = err_msg;
      wrapper->status_ = EventStatus::EVENT_FAILED;
    }
  }
  LOG(INFO) << "finished " << wrapper;
}

void EventResetOPENCL(Event* event) {
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  {
    std::unique_lock<std::mutex> lock(wrapper->mutex_);

    LOG(INFO) << "event reset " << wrapper << "  " << wrapper->status_;
    wrapper->status_ = EventStatus::EVENT_INITIALIZED;
    wrapper->err_msg_ = "";
  }
}

const std::string kNoError = "No error";

const std::string& EventErrorMessageOPENCL(const Event* event) {
  LOG(INFO) << "event message";
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  // supposed to be called after EventQueryCUDA to update status first
  {
    if (wrapper->status_ == EventStatus::EVENT_FAILED) {
      return wrapper->err_msg_;
    } else {
      return kNoError;
    }
  }
}

EventStatus EventQueryOPENCL(const Event* event) {
  // This works because all the calls are synchronous at the moment
  auto* wrapper = static_cast<OpenCLEventWrapper*>(event->event_.get());
  LOG(INFO) << "event query " << wrapper;

  std::unique_lock<std::mutex> lock(wrapper->mutex_);
  if (wrapper->status_ == EventStatus::EVENT_SCHEDULED) {
    LOG(INFO) << "after enforcing";
    wrapper->status_ = EventStatus::EVENT_SUCCESS;
    // Lock only to set error
  }
  LOG(INFO) << "returning query with " << wrapper->status_;
  return static_cast<EventStatus>(wrapper->status_.load());
}

REGISTER_EVENT_CREATE_FUNCTION(OPENCL, EventCreateOPENCL);
REGISTER_EVENT_RECORD_FUNCTION(OPENCL, EventRecordOPENCL);

REGISTER_EVENT_WAIT_FUNCTION(OPENCL, OPENCL, EventWaitOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(CPU, OPENCL, EventWaitCPUOPENCL);
REGISTER_EVENT_WAIT_FUNCTION(OPENCL, CPU, EventWaitOPENCLCPU);
REGISTER_EVENT_FINISH_FUNCTION(OPENCL, EventFinishOPENCL);

REGISTER_EVENT_QUERY_FUNCTION(OPENCL, EventQueryOPENCL);
REGISTER_EVENT_ERROR_MESSAGE_FUNCTION(OPENCL, EventErrorMessageOPENCL);
REGISTER_EVENT_SET_FINISHED_FUNCTION(OPENCL, EventSetFinishedOPENCL);
REGISTER_EVENT_RESET_FUNCTION(OPENCL, EventResetOPENCL);

REGISTER_ALLOCATOR(OPENCL, new DefaultOpenCLAllocator());

} // namespace caffe2
