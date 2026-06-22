#include <ATen/cuda/CUDAGreenContext.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080 && !defined(USE_ROCM) && defined(PYTORCH_C10_DRIVER_API_SUPPORTED)
#include <c10/cuda/driver_api.h>
#include <stdexcept>
#include <vector>
#define HAS_CUDA_GREEN_CONTEXT() 1
#else
#define HAS_CUDA_GREEN_CONTEXT() 0
// Suppress unused private field warnings as this class is not supposed to be called
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wunused-private-field")
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13010 && HAS_CUDA_GREEN_CONTEXT()
#define HAS_CUDA_WORKQUEUE_SUPPORT() 1
#else
#define HAS_CUDA_WORKQUEUE_SUPPORT() 0
#endif

namespace at::cuda {

GreenContext::GreenContext(
    uint32_t device_id,
    std::optional<uint32_t> num_sms,
    std::optional<int32_t> workqueue_scope,
    std::optional<uint32_t> workqueue_concurrency_limit) {
#if HAS_CUDA_GREEN_CONTEXT()
  TORCH_CHECK(
      num_sms.has_value() || workqueue_scope.has_value(),
      "At least one of num_sms or workqueue_scope must be specified");
  TORCH_CHECK(
      !workqueue_concurrency_limit.has_value() || workqueue_scope.has_value(),
      "workqueue_concurrency_limit requires workqueue_scope to be set");

  int driver_version;
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  TORCH_CHECK(
      driver_version >= 12080, "cuda driver too old to use green context!");
  CUcontext pctx = nullptr;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&pctx));
  if (C10_UNLIKELY(!pctx)) {
    TORCH_WARN(
        "Attempted to create a green context but"
        " there was no primary context! Creating a primary context...");

    cudaFree(nullptr);
  }

  CUdevice device;
  device_id_ = device_id;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id));

  std::vector<CUdevResource> resources;

  // --- SM resource ---
  if (num_sms.has_value()) {
    CUdevResource sm_resource;
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
        device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM));

    TORCH_CHECK(
        *num_sms > 0 && *num_sms <= sm_resource.sm.smCount,
        "Invalid number of SMs requested for green context: ",
        *num_sms,
        " (device has ",
        sm_resource.sm.smCount,
        " SMs)");

    // Split resources
    std::vector<CUdevResource> split_result(1);
    unsigned int nb_groups = 1;
    CUdevResource remaining;

    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuDevSmResourceSplitByCount_(
            split_result.data(),
            &nb_groups,
            &sm_resource,
            &remaining,
            0, // default flags
            *num_sms));
    TORCH_CHECK(nb_groups == 1, "Failed to create single SM resource group");
    resources.push_back(split_result[0]);
  }

  // --- Workqueue config resource ---
  if (workqueue_scope.has_value()) {
#if HAS_CUDA_WORKQUEUE_SUPPORT()
    TORCH_CHECK(
        driver_version >= 13010, "cuda driver too old to use workqueue configuration!");
    CUdevResource wq_resource{};
    C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
        device, &wq_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));

    wq_resource.wqConfig.sharingScope =
        static_cast<CUdevWorkqueueConfigScope>(*workqueue_scope);
    if (workqueue_concurrency_limit.has_value()) {
      wq_resource.wqConfig.wqConcurrencyLimit = *workqueue_concurrency_limit;
    }
    resources.push_back(wq_resource);
#else
    TORCH_CHECK(
        false,
        "Workqueue configuration for green contexts requires CUDA 13.1+!");
#endif
  }

  // Generate resource descriptor
  CUdevResourceDesc desc;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDevResourceGenerateDesc_(
          &desc,
          resources.data(),
          static_cast<unsigned int>(resources.size())));

  // Create green context
  // CU_GREEN_CTX_DEFAULT_STREAM is required per docs:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxCreate_(
      &green_ctx_, desc, device, CU_GREEN_CTX_DEFAULT_STREAM));

  // Convert to regular context
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuCtxFromGreenCtx_(&context_, green_ctx_));
  TORCH_CHECK(context_, "Green ctx conversion to regular ctx failed!");
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

std::unique_ptr<GreenContext> GreenContext::create(
    std::optional<uint32_t> device_id,
    std::optional<uint32_t> num_sms,
    std::optional<int32_t> workqueue_scope,
    std::optional<uint32_t> workqueue_concurrency_limit) {
#if HAS_CUDA_GREEN_CONTEXT()
  if (!device_id.has_value()) {
    device_id = at::cuda::current_device();
  }
  return std::unique_ptr<GreenContext>(new GreenContext(
      device_id.value(), num_sms, workqueue_scope, workqueue_concurrency_limit));
#else
  TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
}

uint32_t GreenContext::max_workqueue_concurrency(
    std::optional<uint32_t> device_id) {
#if HAS_CUDA_WORKQUEUE_SUPPORT()
  int driver_version;
  C10_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  TORCH_CHECK(
      driver_version >= 13010, "cuda driver too old to use workqueue configuration!");
  if (!device_id.has_value()) {
    device_id = at::cuda::current_device();
  }
  CUdevice device;
  C10_CUDA_DRIVER_CHECK(
      c10::cuda::DriverAPI::get()->cuDeviceGet_(&device, device_id.value()));
  CUdevResource wq_resource;
  C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuDeviceGetDevResource_(
      device, &wq_resource, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));
  return wq_resource.wqConfig.wqConcurrencyLimit;
#else
  TORCH_CHECK(false, "Workqueue configuration requires CUDA 13.1+!");
#endif
}

  // Implement move operations
#if HAS_CUDA_GREEN_CONTEXT()
  GreenContext::GreenContext(GreenContext&& other) noexcept
      : device_id_(std::exchange(other.device_id_, -1)),
        green_ctx_(std::exchange(other.green_ctx_, nullptr)),
        context_(std::exchange(other.context_, nullptr)),
        parent_stream_(std::exchange(other.parent_stream_, nullptr)) {
        curr_stream_idx_.exchange(other.curr_stream_idx_);
        std::swap(this->green_ctx_streams_, other.green_ctx_streams_);
  }
#else
  GreenContext::GreenContext(GreenContext&& other) noexcept {
    TORCH_CHECK(false, "Green Context move constructor is only supported on CUDA 12.8+!");
  }
#endif

  GreenContext& GreenContext::operator=(GreenContext&& other) noexcept {
#if HAS_CUDA_GREEN_CONTEXT()
    if (this != &other) {
      // Clean up current resources
      if (green_ctx_) {
        CUcontext current = nullptr;
        C10_CUDA_DRIVER_CHECK(
            c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&current));
        if (current == context_) {
          TORCH_CHECK(
              false,
              "attempting to overwrite current green ctx "
              "when it is active!");
        }
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxDestroy_(green_ctx_));
      }

      // Take ownership of other's resources
      device_id_ = std::exchange(other.device_id_, -1);
      green_ctx_ = std::exchange(other.green_ctx_, nullptr);
      context_ = std::exchange(other.context_, nullptr);
      parent_stream_ = std::exchange(other.parent_stream_, nullptr);
      curr_stream_idx_.exchange(other.curr_stream_idx_);
      std::swap(this->green_ctx_streams_, other.green_ctx_streams_);
    }
    return *this;
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  GreenContext::~GreenContext() noexcept{
#if HAS_CUDA_GREEN_CONTEXT()
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuGreenCtxDestroy_(green_ctx_));
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  // Make this context current
  void GreenContext::setContext() {
#if HAS_CUDA_GREEN_CONTEXT()
    auto current_stream = c10::cuda::getCurrentCUDAStream();
    parent_stream_ = current_stream.stream();

    at::cuda::CUDAEvent ev;
    ev.record(current_stream);

    CUcontext current = nullptr;
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuCtxGetCurrent_(&current));
    if (!current) {
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuCtxSetCurrent_(context_));
    } else {
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuCtxPushCurrent_(context_));
    }
    // setContext API uses default stream
    // see GreenContext::Stream() for side-stream creation
    auto green_ctx_stream = c10::cuda::getDefaultCUDAStream();
    ev.block(green_ctx_stream);
    c10::cuda::setCurrentCUDAStream(c10::cuda::CUDAStream(green_ctx_stream));
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  void GreenContext::popContext() {
#if HAS_CUDA_GREEN_CONTEXT()
    // see above note about stream being hardcoded to the default stream
    at::cuda::CUDAEvent ev;
    ev.record(c10::cuda::getCurrentCUDAStream());
    CUcontext popped;
    C10_CUDA_DRIVER_CHECK(
        c10::cuda::DriverAPI::get()->cuCtxPopCurrent_(&popped));
    TORCH_INTERNAL_ASSERT(
        popped == context_, "expected popped context to be the current ctx");
    auto parent_stream = c10::cuda::getStreamFromExternal(parent_stream_, device_id_);
    ev.block(parent_stream);
    c10::cuda::setCurrentCUDAStream(parent_stream);
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }

  CUDAStream GreenContext::Stream() {
#if HAS_CUDA_GREEN_CONTEXT()
    curr_stream_idx_++;
    auto idx = curr_stream_idx_ % kStreamPerGreenContextPool;
    if (curr_stream_idx_ < kStreamPerGreenContextPool) {
       CUstream green_ctx_side_stream;
       C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuGreenCtxStreamCreate_(
         &green_ctx_side_stream, green_ctx_, CU_STREAM_NON_BLOCKING, 0));
       // implies we leak side-streams, but this has precedent in e.g., c10/cuda/CUDAStream.cpp
       // if we do not have any statically allocated GreenContexts, would it be safe to
       // destroy these streams in a destructor?
       green_ctx_streams_[idx] = green_ctx_side_stream;
       return c10::cuda::getStreamFromExternal(green_ctx_side_stream, device_id_);
    }
    return c10::cuda::getStreamFromExternal(green_ctx_streams_[idx], device_id_);
#else
    TORCH_CHECK(false, "Green Context is only supported on CUDA 12.8+!");
#endif
  }
} // namespace at::cuda
