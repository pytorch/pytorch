
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/MemPool.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/env.h>

#include <memory>
#include <optional>
#include <utility>

namespace {

// A CUDA graph and the stream it is being captured on, since capture_end
// must run on that stream.
struct CUDAGraphState {
  at::cuda::CUDAGraph graph;
  std::optional<at::cuda::CUDAStream> capture_stream;
  c10::DeviceIndex device = -1;
};

CUDAGraphState* to_graph_state(CUDAGraphHandle graph) {
  TORCH_CHECK(graph != nullptr, "CUDA graph handle is null");
  return reinterpret_cast<CUDAGraphState*>(graph);
}

class IgnoreWarnings : public c10::WarningHandler {
 public:
  void process(const c10::Warning& /*warning*/) override {}
};

bool strict_cuda_graph_env() {
  auto value = c10::utils::get_env("AOTI_RUNTIME_STRICT_CUDA_GRAPH");
  return value.has_value() && value->c_str()[0] != '0';
}

// Ends a capture whose graph won't be used. The stream stops capturing and the
// allocator stops routing to the pool even when the capture was invalidated,
// so errors are ignored, as is the warning about an empty graph, unless
// AOTI_RUNTIME_STRICT_CUDA_GRAPH is set.
void discard_capture(CUDAGraphState& state) {
  auto stream = std::exchange(state.capture_stream, std::nullopt);
  if (!stream.has_value()) {
    return;
  }
  at::cuda::CUDAStreamGuard stream_guard(*stream);
  if (strict_cuda_graph_env()) {
    state.graph.capture_end_pre();
    return;
  }
  IgnoreWarnings ignore_warnings;
  c10::WarningUtils::WarningHandlerGuard warning_guard(&ignore_warnings);
  try {
    state.graph.capture_end_pre();
    // NOLINTNEXTLINE(bugprone-empty-catch)
  } catch (...) {
  }
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
    *ret_pool = reinterpret_cast<CUDAMemPoolHandle>(new at::cuda::MemPool());
  });
}

AOTITorchError aoti_torch_delete_cuda_mem_pool(CUDAMemPoolHandle pool) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::MemPool*>(pool); });
}

AOTITorchError aoti_torch_create_cuda_graph(CUDAGraphHandle* ret_graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret_graph = reinterpret_cast<CUDAGraphHandle>(new CUDAGraphState());
  });
}

AOTITorchError aoti_torch_delete_cuda_graph(CUDAGraphHandle graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::unique_ptr<CUDAGraphState> state(
        reinterpret_cast<CUDAGraphState*>(graph));
    if (state != nullptr) {
      discard_capture(*state);
    }
  });
}

AOTITorchError aoti_torch_cuda_graph_capture_begin(
    CUDAGraphHandle graph,
    CUDAMemPoolHandle pool,
    void* stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* state = to_graph_state(graph);
    TORCH_CHECK(pool != nullptr, "CUDA memory pool handle is null");
    auto* mem_pool = reinterpret_cast<at::cuda::MemPool*>(pool);
    auto capture_stream = at::cuda::getStreamFromExternal(
        static_cast<cudaStream_t>(stream), mem_pool->device());
    // Python creates the default generators when it initializes CUDA; other
    // hosts would create one lazily inside the capture, which fails.
    at::cuda::detail::getDefaultCUDAGenerator(mem_pool->device());
    at::cuda::CUDAStreamGuard stream_guard(capture_stream);
    state->graph.capture_begin(
        mem_pool->id(), cudaStreamCaptureModeThreadLocal);
    state->capture_stream = capture_stream;
    state->device = mem_pool->device();
  });
}

AOTITorchError aoti_torch_cuda_graph_capture_end(
    CUDAGraphHandle graph,
    bool instantiate) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* state = to_graph_state(graph);
    TORCH_CHECK(
        state->capture_stream.has_value(), "CUDA graph is not being captured");
    if (!instantiate) {
      discard_capture(*state);
      return AOTI_TORCH_SUCCESS;
    }
    // The stream stops capturing even if ending the capture fails.
    at::cuda::CUDAStreamGuard stream_guard(*state->capture_stream);
    state->capture_stream.reset();
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
