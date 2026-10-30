
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <ATen/Context.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/MemPool.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

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
    at::cuda::CUDAGuard device_guard(
        static_cast<c10::DeviceIndex>(device_index));
    *ret_pool = reinterpret_cast<CUDAMemPoolHandle>(new at::cuda::MemPool());
  });
}

AOTITorchError aoti_torch_delete_cuda_mem_pool(CUDAMemPoolHandle pool) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* mem_pool = reinterpret_cast<at::cuda::MemPool*>(pool);
    if (mem_pool != nullptr) {
      // ~MemPool would throw, and so abort, while this thread captures.
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      C10_CUDA_CHECK(cudaStreamIsCapturing(
          c10::cuda::getCurrentCUDAStream(mem_pool->device()).stream(),
          &status));
      TORCH_CHECK(
          status == cudaStreamCaptureStatusNone,
          "Can't delete a CUDA memory pool while the current stream on its "
          "device is capturing");
    }
    delete mem_pool;
  });
}

AOTITorchError aoti_torch_create_cuda_graph(
    int32_t keep_graph,
    int64_t num_rng_states,
    CUDAGraphHandle* ret_graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret_graph = reinterpret_cast<CUDAGraphHandle>(
        new at::cuda::CUDAGraph(keep_graph != 0, num_rng_states));
  });
}

AOTITorchError aoti_torch_delete_cuda_graph(CUDAGraphHandle graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::cuda::CUDAGraph*>(graph); });
}

AOTITorchError aoti_torch_cuda_graph_capture_begin(
    CUDAGraphHandle graph,
    CUDAMemPoolHandle pool,
    int32_t capture_mode) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(graph != nullptr, "Null CUDA graph handle");
    TORCH_CHECK(pool != nullptr, "Null CUDA memory pool handle");
    TORCH_CHECK(
        capture_mode == cudaStreamCaptureModeGlobal ||
            capture_mode == cudaStreamCaptureModeThreadLocal ||
            capture_mode == cudaStreamCaptureModeRelaxed,
        "Invalid CUDA graph capture mode ",
        capture_mode);
    // Split so hipify keeps the name, which ROCm's allocator reports too.
    TORCH_CHECK(
        c10::cuda::CUDACachingAllocator::name() !=
            "c"
            "udaMallocAsync",
        "CUDA graph capture isn't supported with "
        "PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync");
    auto* mem_pool = reinterpret_cast<at::cuda::MemPool*>(pool);
    // On another device, the allocator would quietly route the capture to a
    // new pool on that device.
    TORCH_CHECK(
        c10::cuda::current_device() == mem_pool->device(),
        "A CUDA graph capture must run on its pool's device ",
        static_cast<int>(mem_pool->device()),
        ", not the current device ",
        static_cast<int>(c10::cuda::current_device()));
    // Python creates the default generators when it initializes CUDA; other
    // hosts would create one lazily inside the capture, which fails.
    at::cuda::detail::getDefaultCUDAGenerator(mem_pool->device());
    reinterpret_cast<at::cuda::CUDAGraph*>(graph)->capture_begin(
        mem_pool->id(), static_cast<cudaStreamCaptureMode>(capture_mode));
  });
}

AOTITorchError aoti_torch_cuda_graph_capture_end(CUDAGraphHandle graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(graph != nullptr, "Null CUDA graph handle");
    reinterpret_cast<at::cuda::CUDAGraph*>(graph)->capture_end();
  });
}

AOTITorchError aoti_torch_cuda_graph_replay(CUDAGraphHandle graph) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    TORCH_CHECK(graph != nullptr, "Null CUDA graph handle");
    auto* cuda_graph = reinterpret_cast<at::cuda::CUDAGraph*>(graph);
    // As torch.cuda.CUDAGraph.replay does for keep_graph.
    if (!cuda_graph->has_graph_exec()) {
      cuda_graph->instantiate();
    }
    cuda_graph->replay();
  });
}
