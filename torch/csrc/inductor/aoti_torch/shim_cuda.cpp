
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/MemPool.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/ScopeExit.h>

#include <memory>
#include <optional>
#include <thread>
#include <utility>

namespace {

// The caching allocators fail, and leak a pool reference, on a second capture
// into a pool that is already capturing. Only graphs hold the host caching
// allocator's reference on a pool, and it asserts on a capture into a pool
// whose graphs are all gone. The shim checks both before capturing; a pool
// and its graphs are used by one thread at a time.
struct CUDAMemPoolUsage {
  bool capturing = false;
  int64_t graphs = 0;
  bool captured = false;
};

struct CUDAMemPoolState {
  at::cuda::MemPool pool;
  std::shared_ptr<CUDAMemPoolUsage> usage =
      std::make_shared<CUDAMemPoolUsage>();
};

// A CUDA graph, and while it is being captured, the stream and thread of the
// capture, since capture_end must run on both.
struct CUDAGraphState {
  at::cuda::CUDAGraph graph;
  std::optional<at::cuda::CUDAStream> capture_stream;
  std::thread::id capture_thread;
  c10::DeviceIndex device = -1;
  // Set once a capture into the pool began; counts this graph until deleted.
  std::shared_ptr<CUDAMemPoolUsage> pool_usage;

  CUDAGraphState() = default;
  ~CUDAGraphState() {
    if (pool_usage) {
      pool_usage->graphs--;
    }
  }
  CUDAGraphState(const CUDAGraphState&) = delete;
  CUDAGraphState& operator=(const CUDAGraphState&) = delete;
  CUDAGraphState(CUDAGraphState&&) = delete;
  CUDAGraphState& operator=(CUDAGraphState&&) = delete;
};

CUDAGraphState* to_graph_state(CUDAGraphHandle graph) {
  TORCH_CHECK(graph != nullptr, "CUDA graph handle is null");
  return reinterpret_cast<CUDAGraphState*>(graph);
}

// Passes on every warning but the one about an empty graph, which discarding
// a capture that recorded nothing would raise.
class IgnoreEmptyGraphWarning : public c10::WarningHandler {
 public:
  explicit IgnoreEmptyGraphWarning(c10::WarningHandler* next) : next_(next) {}
  void process(const c10::Warning& warning) override {
    if (warning.msg().find("The CUDA Graph is empty") == std::string::npos) {
      next_->process(warning);
    }
  }

 private:
  c10::WarningHandler* next_;
};

void check_capture_thread(const CUDAGraphState& state) {
  TORCH_CHECK(
      std::this_thread::get_id() == state.capture_thread,
      "A CUDA graph capture must end on the thread that began it");
}

// Ends a capture whose graph is being deleted, without instantiating it. The
// stream stops capturing and the allocator stops routing to the pool before
// any error is thrown, e.g. for an invalidated capture.
void discard_capture(CUDAGraphState& state) {
  auto stream = std::exchange(state.capture_stream, std::nullopt);
  if (!stream.has_value()) {
    return;
  }
  auto pool_free =
      c10::make_scope_exit([&] { state.pool_usage->capturing = false; });
  at::cuda::CUDAStreamGuard stream_guard(*stream);
  IgnoreEmptyGraphWarning warning_handler(
      c10::WarningUtils::get_warning_handler());
  c10::WarningUtils::WarningHandlerGuard warning_guard(&warning_handler);
  state.graph.capture_end_pre();
}

} // namespace

AOTITorchError aoti_torch_create_cuda_guard(
    int32_t device_index,
    CUDAGuardHandle* ret_guard // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::CUDAGuard* guard = new at::cuda::CUDAGuard(device_index);
    *ret_guard = reinterpret_cast<CUDAGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_guard(CUDAGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAGuard*>(guard); });
}

AOTITorchError aoti_torch_cuda_guard_set_index(
    CUDAGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    reinterpret_cast<at::cuda::CUDAGuard*>(guard)->set_index(device_index);
  });
}

AOTITorchError aoti_torch_create_cuda_stream_guard(
    void* stream,
    int32_t device_index,
    CUDAStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::cuda::CUDAStreamGuard* guard =
        new at::cuda::CUDAStreamGuard(at::cuda::getStreamFromExternal(
            static_cast<cudaStream_t>(stream), device_index));
    *ret_guard = reinterpret_cast<CUDAStreamGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_cuda_stream_guard(
    CUDAStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAStreamGuard*>(guard); });
}

AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *(cudaStream_t*)(ret_stream) = at::cuda::getCurrentCUDAStream(device_index);
  });
}

AOTITorchError aoti_torch_create_cuda_tunableop_dynamic_dims_guard(
    uint8_t dynamic_dims_mask,
    CUDATunableOpDynamicDimsGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(
        dynamic_dims_mask <= 0xF,
        "TunableOp dynamic-dimension mask must fit in four bits, but got ",
        static_cast<int>(dynamic_dims_mask));
    *ret_guard = nullptr;
    auto* tunable_context = at::cuda::tunable::getTuningContext();
    if (dynamic_dims_mask != 0 && tunable_context->IsTuningEnabled() &&
        tunable_context->IsTunableOpEnabled()) {
      auto* guard = new at::cuda::tunable::TunableDynamicDimsGuard(
          at::cuda::tunable::DynamicDimsMask{dynamic_dims_mask});
      *ret_guard = reinterpret_cast<CUDATunableOpDynamicDimsGuardHandle>(guard);
    }
  });
}

AOTITorchError aoti_torch_delete_cuda_tunableop_dynamic_dims_guard(
    CUDATunableOpDynamicDimsGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    delete reinterpret_cast<at::cuda::tunable::TunableDynamicDimsGuard*>(guard);
  });
}

AOTITorchError aoti_torch_cuda_caching_allocator_raw_alloc(
    uint64_t nbytes,
    void** ret_ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (nbytes == 0) {
      *ret_ptr = nullptr;
      return AOTI_TORCH_SUCCESS;
    }

    *ret_ptr = c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);

    if (*ret_ptr == nullptr) {
      TORCH_CHECK(
          false,
          "Failed to allocate ",
          nbytes,
          " bytes from CUDA caching allocator");
    }
  });
}

AOTITorchError aoti_torch_cuda_caching_allocator_raw_delete(void* ptr) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (ptr != nullptr) {
      c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    }
  });
}

AOTITorchError aoti_torch_create_cuda_mem_pool(
    int32_t device_index,
    CUDAMemPoolHandle* ret_pool) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // MemPool needs an initialized caching allocator on the current device.
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    auto device = static_cast<c10::DeviceIndex>(device_index);
    at::cuda::CUDAGuard device_guard(device);
    *ret_pool = reinterpret_cast<CUDAMemPoolHandle>(new CUDAMemPoolState());
  });
}

AOTITorchError aoti_torch_delete_cuda_mem_pool(CUDAMemPoolHandle pool) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<CUDAMemPoolState*>(pool); });
}

AOTITorchError aoti_torch_create_cuda_graph(CUDAGraphHandle* ret_graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret_graph = reinterpret_cast<CUDAGraphHandle>(new CUDAGraphState());
  });
}

AOTITorchError aoti_torch_delete_cuda_graph(CUDAGraphHandle graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (graph == nullptr) {
      return AOTI_TORCH_SUCCESS;
    }
    auto* state = to_graph_state(graph);
    if (state->capture_stream.has_value()) {
      check_capture_thread(*state);
    }
    std::unique_ptr<CUDAGraphState> owned_state(state);
    discard_capture(*owned_state);
  });
}

AOTITorchError aoti_torch_cuda_graph_capture_begin(
    CUDAGraphHandle graph,
    CUDAMemPoolHandle pool,
    void* stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* state = to_graph_state(graph);
    TORCH_CHECK(pool != nullptr, "CUDA memory pool handle is null");
    TORCH_CHECK(state->pool_usage == nullptr, "A CUDA graph captures once");
    auto* pool_state = reinterpret_cast<CUDAMemPoolState*>(pool);
    auto* mem_pool = &pool_state->pool;
    auto& usage = pool_state->usage;
    TORCH_CHECK(
        !usage->capturing, "A CUDA memory pool takes one capture at a time");
    TORCH_CHECK(
        !usage->captured || usage->graphs > 0,
        "A CUDA memory pool can't take a capture once every graph that began "
        "a capture into it is deleted; use a new pool");
    // Checked first: querying a capturing stream's flags would invalidate its
    // capture.
    auto raw_stream = static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    C10_CUDA_CHECK(cudaStreamIsCapturing(raw_stream, &capture_status));
    TORCH_CHECK(
        capture_status == cudaStreamCaptureStatusNone,
        "The CUDA graph capture stream is already capturing");
    // While a blocking stream captures, work that other threads launch on the
    // legacy default stream would invalidate the capture.
    unsigned int stream_flags = 0;
    C10_CUDA_CHECK(cudaStreamGetFlags(raw_stream, &stream_flags));
    TORCH_CHECK(
        stream_flags & cudaStreamNonBlocking,
        "A CUDA graph capture stream must be created with cudaStreamNonBlocking");
    auto capture_stream =
        at::cuda::getStreamFromExternal(raw_stream, mem_pool->device());
    // Python creates the default generators when it initializes CUDA; other
    // hosts would create one lazily inside the capture, which fails.
    at::cuda::detail::getDefaultCUDAGenerator(mem_pool->device());
    at::cuda::CUDAStreamGuard stream_guard(capture_stream);
    // Even a capture_begin that fails may have taken, and then dropped, the
    // host caching allocator's reference on the pool.
    usage->captured = true;
    try {
      state->graph.capture_begin(
          mem_pool->id(), cudaStreamCaptureModeThreadLocal);
    } catch (...) {
      // Undoes the pool routing a failed capture_begin may have set up.
      state->graph.reset();
      throw;
    }
    usage->capturing = true;
    usage->graphs++;
    state->pool_usage = usage;
    state->capture_stream = capture_stream;
    state->capture_thread = std::this_thread::get_id();
    state->device = mem_pool->device();
  });
}

AOTITorchError aoti_torch_cuda_graph_capture_end(CUDAGraphHandle graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* state = to_graph_state(graph);
    TORCH_CHECK(
        state->capture_stream.has_value(), "CUDA graph is not being captured");
    check_capture_thread(*state);
    // The stream stops capturing even if ending the capture fails.
    at::cuda::CUDAStreamGuard stream_guard(*state->capture_stream);
    state->capture_stream.reset();
    auto pool_free =
        c10::make_scope_exit([&] { state->pool_usage->capturing = false; });
    state->graph.capture_end();
  });
}

AOTITorchError aoti_torch_cuda_graph_replay(
    CUDAGraphHandle graph,
    void* stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* state = to_graph_state(graph);
    TORCH_CHECK(state->device >= 0, "CUDA graph was never captured");
    at::cuda::CUDAStreamGuard stream_guard(at::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(stream), state->device));
    state->graph.replay();
  });
}
