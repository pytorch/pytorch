#include <ATen/Functions.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/core/CPUAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace at::cuda {

static bool _cuda_graphs_debug = false;
constexpr int kSynchronizeBusyWaitMillis = 10;

MempoolId_t graph_pool_handle() {
  // Sets just the second value, to distinguish it from MempoolId_ts created from
  // cudaStreamGetCaptureInfo id_s in capture_begin.
  auto new_pool = c10::cuda::MemPool();
  return new_pool.id();
}

/**
 * Note [CUDA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native CUDA ops (like RNG ops with
 *     CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [CUDA Graph-safe RNG states] in CUDAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write capture
 *     bindings naively in an extension, they likely won't interact with the native
 *     ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with CUDA graph capture] in CUDACachingAllocator.cpp
 * describes memory management for captures.
 */

std::atomic<int> CUDAGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL watchdog so that they
// can be resolved before the capture begins. Note that event queries are not allowed during a
// graph capture in the default capture mode.
void CUDAGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

void CUDAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(pending_event_queries > 0,
    "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int CUDAGraph::num_pending_event_queries() {
  return pending_event_queries;
}

CUDAGraph::CUDAGraph()
  // CUDAStreams may not be default-constructed.
  : capture_stream_(at::cuda::getCurrentCUDAStream()) {
}

void CUDAGraph::register_generator_state(
    c10::intrusive_ptr<at::CUDAGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

void CUDAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<CUDAGeneratorImpl> cuda_gen =
      dynamic_intrusive_pointer_cast<CUDAGeneratorImpl>(
          generator.getIntrusivePtr());
  cuda_gen->register_graph(this);
}

void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/, cudaStreamCaptureMode capture_mode, int num_dynamic_args) {
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph. "
              "To capture a new graph, create a new instance.");

  // default generator is always registered
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      std::nullopt, cuda::detail::getDefaultCUDAGenerator());
  gen->register_graph(this);

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream. "
              "(However, after capture, it's ok to replay them on the "
              "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be nonzero.
    // If pool was created by graph_pool_handle, second should be nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Create graph pool handle using is_user_created=false.
    // Sets just the first value, to distinguish it from MempoolId_ts created by graph_pool_handle().
    auto mempool = c10::cuda::MemPool({}, false);
    mempool_id_ = mempool.id();
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // Addendum: beginAllocateStreamToPool is now called before cudaStreamBeginCapture to prevent an
  // autograd thread's free() call triggering an invalid cudaEventRecord in the caching allocator
  // due to the capture status being updated _after_ a capture had already started.
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(capture_dev_, mempool_id_, [this](cudaStream_t stream) {
      cudaStreamCaptureStatus status;
      CaptureId_t stream_capture_id;
      AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
      return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive && stream_capture_id == capture_id_;
  }, num_dynamic_args ? std::optional{[this](void* ptr, size_t size) {
      std::cout << "tracked alloc " << allocations_.size() << " of size " << size << " returning " << ptr << std::endl;
      allocations_.push_back(TrackedAllocation{
        .ptr = (char*) ptr,
        .size = size,
        .alloc_idx = allocations_.size(),
      });
    }} : std::nullopt);

  num_dynamic_args_ = num_dynamic_args;

  // At this point, any NCCL watchdogs should be aware that we are in capture mode
  // and therefore should not enqueue any additional work that could be event-queried.
  // We still must wait on any existing work that has not been cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE("Waiting for pending NCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(
      std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe CUDA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  cudaStreamCaptureStatus status;
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);

}

void CUDAGraph::capture_end() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");
  has_graph_ = true;

  if (num_dynamic_args_) {
    // must do this before cudaGraphInstantiate, since it calls cudaGraphKernelNodeSetAttribute
    introspect_dynamic_graph();
    // TODO: we could releasePool here? would save a good amount of memory...
    // (because all our allocations will be redone and overwritten on every replay_dynamic)
  }

  // In typical graph usage some tensors (e.g. the tensors used for graph IO) are not freed
  // between replays.
  // If Pytorch compiles and runs with a CUDA 11.4+ toolkit, there's a chance the allocator backend
  // is cudaMallocAsync.
  // cudaMallocAsync is generally graph-safe, but if some tensors are not freed between replays,
  // the graph's internal bookkeeping requires that we instantiate with
  // cudaGraphInstantiateFlagAutoFreeOnLaunch. See
  // cudaGraphLaunch
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597
  // cudaGraphInstantiateWithFlags
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1ga2c652a24ba93e52b99a47bec0888233
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
#endif
    // Trailing NULL, NULL, 0 arguments were recommended by Cuda driver people,
    // who prefer not to report error message through these arguments moving forward
    // (they prefer return value, or errors on api calls internal to the capture)
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 12000)
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0));
#else
    AT_CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
#endif
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11040)
  } else {
    AT_CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec_,
                                                graph_,
                                                cudaGraphInstantiateFlagAutoFreeOnLaunch));
  }
#endif

  has_graph_exec_ = true;

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty. This usually means that the graph was ",
                 "attempted to be captured on wrong device or stream.");
  }

  // check if debug path is set
  if (!_cuda_graphs_debug && !num_dynamic_args_) {
    // Now that we've instantiated graph_ into graph_exec_,
    // we don't need graph_ anymore.
    AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (_cuda_graphs_debug) {
    TORCH_WARN("DEBUG: TORCH_CUDAGRAPHS_DEBUG_PATH detected. graph_ will not be freed until debug_dump is called.");
  }
}

void CUDAGraph::replay() {
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay without a preceding successful capture.");
  TORCH_CHECK(!num_dynamic_args_, "Call replay_dynamic instead");

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }
  // graph_exec_ may be replayed in any stream.
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    // Workaround for bug in libcuda.so that causes replayed graphs with
    // certain topologies to be corrupted (kernels elided, internal syncs
    // ignored) when replayed back to back without a sync in between.
    // The bug is fixed in CUDA 11.4+.
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void CUDAGraph::enable_debug_mode() {
  _cuda_graphs_debug = true;
}

void CUDAGraph::debug_dump(const std::string& debug_path) {
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11030)|| defined(USE_ROCM)
  if (_cuda_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling cudaGraphDebugDotPrint() with ", debug_path);
      C10_CUDA_CHECK_WARN(cudaGraphDebugDotPrint(graph_, debug_path.c_str(), cudaGraphDebugDotFlagsVerbose)); // most verbose output
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
      has_graph_ = false;
    }
  } else {
    TORCH_WARN("CUDA Graphs debug not enabled, set with torch._C._cuda_enable_graphs_debug_mode");
  }
#else
  TORCH_CHECK(false, "CUDA graphs may only be used in Pytorch built with CUDA >= 11.3 or ROCM >= 5.6");
#endif
}

void CUDAGraph::reset() {
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this CUDAGraph, the generator,
  // and the allocator could end up in all kinds of weird states depending where failure occurred.
  // If the user catches the failure exception in a script, or is running in REPL or (god forbid)
  // a Jupyter notebook, I don't see an easy way for reset() to gracefully fix all such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::cuda::CUDACachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_CUDA_CHECK_WARN(cudaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_CUDA_CHECK_WARN(cudaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
  if (num_dynamic_args_) {
    allocations_.clear();
    kernel_param_updates_.clear();
    graph_node_param_updates_.clear();
    num_dynamic_args_ = 0;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory pool as this graph.
MempoolId_t CUDAGraph::pool() {
TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

CUDAGraph::~CUDAGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(this);
  }
  reset();
}

std::optional<std::tuple<size_t, size_t>> checkAllocationWithinGraph(void* ptr, const std::vector<TrackedAllocation>& sorted_allocations) {
  // since allocations is sorted in ptr order, we can search it in log time
  auto it = std::upper_bound(sorted_allocations.begin(), sorted_allocations.end(), ptr, [](const void* p, const TrackedAllocation& alloc) {
    return p < alloc.ptr;
  });
  // upper_bound finds the first allocation whose ptr is strictly greater than our search
  if (it == sorted_allocations.begin()) {
    return std::nullopt; // the ptr is before our first allocation
  }
  // so we must decrement to get to the one we actually want
  --it;
  void* begin = it->ptr;
  void* end = (char*) begin + it->size;
  if (ptr >= begin && ptr < end) {
    size_t offset = (char*) ptr - (char*) begin;
    return std::make_tuple(it->alloc_idx, offset);
  }
  return std::nullopt;
}

std::vector<TrackedAllocation> combine_ranges(const std::vector<TrackedAllocation>& input, size_t num_dynamic_args) {
  // the allocator can return overlaps, e.g. (128, 256) is allocated then freed, later (0, 256) could be given to another allocation
  // in such a case, we don't know whether a reference to, say, 200 was from the first or second allocation
  // to sidestep this conundrum, we will find any overlaps, and do one big allocation for the union of each overlapping area
  std::vector<TrackedAllocation> ranges = input;
  std::sort(ranges.begin(), ranges.end(), [](auto& a, auto& b) {
    return a.ptr < b.ptr;
  });
  std::vector<TrackedAllocation> result;
  for (const TrackedAllocation& range : ranges) {
    if (result.empty() || range.ptr >= result.back().ptr + result.back().size) { // ">=" because ranges are exclusive on upper end
      result.push_back(range); // No overlap, add new range
    } else {
      TrackedAllocation& last = result.back(); // Overlap found, extend the previous range
      TORCH_CHECK(last.alloc_idx >= num_dynamic_args && range.alloc_idx >= num_dynamic_args, "Cannot combine argument allocations");
      char* last_end = last.ptr + last.size;
      char* range_end = range.ptr + range.size;
      last.size = std::max(last_end, range_end) - last.ptr;
    }
  }
  // sort by original alloc_idx to bring the original dynamic args back to the beginning, in original order
  // (since the dynamic args were allocated sequentially from a private pool, and never freed for the duration of the function, they should always be in ascending ptr order anyway? but there's no reason to believe that for the rest of the allocations)
  std::sort(result.begin(), result.end(), [](auto& a, auto& b) {
    return a.alloc_idx < b.alloc_idx;
  });
  for (size_t i = 0; i < result.size(); i++) {
    if (i < num_dynamic_args) {
      // really important, let's make absolutely sure that the inputs to the function emerge unscathed (no merging or reordering)
      TORCH_INTERNAL_ASSERT(result[i].alloc_idx == i);
    } else {
      result[i].alloc_idx = i; // reassign alloc_idx to actually match the order we'll be going in
    }
  }
  return result;
}

void CUDAGraph::introspect_dynamic_graph() {
  allocations_ = combine_ranges(allocations_, num_dynamic_args_);

  std::vector<TrackedAllocation> sorted_allocations = allocations_;
  // copy since we're going to mess up allocation order, just for this function
  // we will need to look up a bunch of pointers, and binary search can speed this up from linear to log time
  // therefore:
  std::sort(sorted_allocations.begin(), sorted_allocations.end(), [](auto& a, auto& b) {
    return a.ptr < b.ptr;
  });

  size_t num_nodes;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &num_nodes));
  std::cout << "number of nodes captured " << num_nodes << std::endl;
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nodes.data(), &num_nodes));

  for (size_t i = 0; i < num_nodes; ++i) {
    cudaGraphNode_t node = nodes[i];

    cudaGraphNodeType type;
    AT_CUDA_CHECK(cudaGraphNodeGetType(node, &type));

    if (type == cudaGraphNodeTypeKernel) {
      cudaKernelNodeParams nodeParams;
      AT_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &nodeParams));
      cudaFunction_t func;
      AT_CUDA_CHECK(cudaGetFuncBySymbol(&func, nodeParams.func));

      const char* func_name;
      globalContext().getNVRTC().cuFuncGetName(&func_name, func);

      size_t param_index = 0;
      size_t param_offset;
      size_t param_size;
      while (globalContext().getNVRTC().cuFuncGetParamInfo(
                 func, param_index, &param_offset, &param_size) !=
             CUDA_ERROR_INVALID_VALUE) {
        char** arg1_speculative_pointer =
            (char**)nodeParams.kernelParams[param_index];
        
        // ABI guarantees that pointers have 8-byte alignment
        for (size_t address_start = 0; param_size - address_start >= 8;
             address_start += 8) {
          char* arg1_value = arg1_speculative_pointer[address_start / 8];
          if (auto result = checkAllocationWithinGraph(arg1_value, sorted_allocations)) {
            auto [alloc_idx, offset] = *result;
            std::cout << "LEIJURV: I have decided that " << address_start << " bytes into argument #" << param_index << " of kernel " << func_name << " is actually allocation " << alloc_idx << " indexed to offset " << offset << std::endl;
            cudaKernelNodeAttrValue attr_value = {
                .deviceUpdatableKernelNode = {
                    .deviceUpdatable = 1,
                    .devNode = nullptr,
                }
            };
            AT_CUDA_CHECK(cudaGraphKernelNodeSetAttribute(
                node,
                cudaLaunchAttributeDeviceUpdatableKernelNode,
                &attr_value));
            TORCH_CHECK(
                attr_value.deviceUpdatableKernelNode.devNode != nullptr);

            kernel_param_updates_.push_back({
              .dev_node = attr_value.deviceUpdatableKernelNode.devNode,
              .param_offset = param_offset + address_start,
              .alloc_idx = alloc_idx,
              .offset = offset,
            });
          }
        }
        param_index++;
      }

    } else if (type == cudaGraphNodeTypeMemcpy) {
      cudaMemcpy3DParms memcpyParams1;
      AT_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &memcpyParams1));
      auto srcPtrResult = checkAllocationWithinGraph(memcpyParams1.srcPtr.ptr, sorted_allocations);
      auto dstPtrResult = checkAllocationWithinGraph(memcpyParams1.dstPtr.ptr, sorted_allocations);
      if (!srcPtrResult && !dstPtrResult) {
        continue;
      }
      
      graph_node_param_updates_.push_back({
        .node = node,
        .compute_new_params = [memcpyParams1, srcPtrResult, dstPtrResult](std::vector<void*> actualDataPtrs) {
          cudaPitchedPtr srcPtr = memcpyParams1.srcPtr;
          cudaPitchedPtr dstPtr = memcpyParams1.dstPtr;
          if (srcPtrResult) {
            auto [alloc_idx, offset] = *srcPtrResult;
            srcPtr.ptr = (char*)actualDataPtrs[alloc_idx] + offset;
          }
          if (dstPtrResult) {
            auto [alloc_idx, offset] = *dstPtrResult;
            dstPtr.ptr = (char*)actualDataPtrs[alloc_idx] + offset;
          }
          return cudaGraphNodeParams{
            .type = cudaGraphNodeTypeMemcpy,
            .memcpy = {
              .copyParams = {
                .srcArray = memcpyParams1.srcArray,
                .srcPos = memcpyParams1.srcPos,
                .srcPtr = srcPtr,
                .dstArray = memcpyParams1.dstArray,
                .dstPos = memcpyParams1.dstPos,
                .dstPtr = dstPtr,
                .extent = memcpyParams1.extent,
                .kind = memcpyParams1.kind
              },
            },
          };
        },
      });
    } else if (type == cudaGraphNodeTypeMemset) {
      cudaMemsetParams memsetParams1;
      AT_CUDA_CHECK(cudaGraphMemsetNodeGetParams(node, &memsetParams1));

      auto dstPtrResult = checkAllocationWithinGraph(memsetParams1.dst, sorted_allocations);
      if (!dstPtrResult) {
        continue;
      }
      
      graph_node_param_updates_.push_back({
        .node = node,
        .compute_new_params = [memsetParams1, dstPtrResult](std::vector<void*> actualDataPtrs) {
          void* dstPtr = memsetParams1.dst;
          if (dstPtrResult) {
            auto [alloc_idx, offset] = *dstPtrResult;
            dstPtr = (char*)actualDataPtrs[alloc_idx] + offset;
          }
          return cudaGraphNodeParams{
            .type = cudaGraphNodeTypeMemset,
            .memset = {
              .dst = dstPtr,
              .pitch = memsetParams1.pitch,
              .value = memsetParams1.value,
              .elementSize = memsetParams1.elementSize,
              .width = memsetParams1.width,
              .height = memsetParams1.height,
            },
          };
        },
      });
    } else if (type == cudaGraphNodeTypeGraph || type == cudaGraphNodeTypeConditional) {
      throw std::runtime_error("horrible. please don't make me support these.");
      assert(false);
    } else if (
      type == cudaGraphNodeTypeEmpty ||
      type == cudaGraphNodeTypeWaitEvent ||
      type == cudaGraphNodeTypeHost ||
      type == cudaGraphNodeTypeEventRecord ||
      type == cudaGraphNodeTypeExtSemaphoreSignal ||
      type == cudaGraphNodeTypeExtSemaphoreWait ||
      type == cudaGraphNodeTypeMemAlloc ||
      type == cudaGraphNodeTypeMemFree
    ) {
      // this is fine
    } else {
      throw std::runtime_error("graph node type unknown");
      assert(false);
    }
  }
}

void CUDART_CB hostMemoryFreeCallback(void* data) {
  at::getCPUAllocator()->raw_deallocate(data);
}

void CUDAGraph::replay_dynamic(const std::vector<at::Tensor>& input_tensors) {
  TORCH_CHECK(num_dynamic_args_, "Must be a dynamic graph");
  TORCH_CHECK(has_graph_exec_,
              "Called CUDAGraph::replay_dynamic without a preceding successful capture.");
  TORCH_CHECK(input_tensors.size() == (size_t) num_dynamic_args_);
  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // take the allocations that were requested during the capture, and allocate them "for real"
  std::vector<void*> actualDataPtrs;
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  for (size_t i = 0; i < allocations_.size(); i++) {
    TORCH_INTERNAL_ASSERT(allocations_[i].alloc_idx == i);
    if (i < (size_t) num_dynamic_args_) {
      // the first few are inputs/outputs, so they are "prefilled"
      TORCH_CHECK(input_tensors[i].is_contiguous(), "All tensors must be contiguous");
      // TODO ^ this isn't quite right. really, the assert should be that the stride is the same as the original input arg.
      TORCH_CHECK(input_tensors[i].nbytes() == allocations_[i].size, "Prefilled tensors must be same shape");
      actualDataPtrs.push_back(input_tensors[i].data_ptr());
    } else {
      // the rest are allocated empty
      actualDataPtrs.push_back(allocator->raw_alloc_with_stream(allocations_[i].size, stream));
    }
  }

  for (auto& update : graph_node_param_updates_) {
    // run the updates that can't be done device-side
    // note that this is quite rare - it only applies to a few misc memsets/memcpys
    cudaGraphNodeParams newParams = update.compute_new_params(actualDataPtrs);
    AT_CUDA_CHECK(cudaGraphExecNodeSetParams(graph_exec_, update.node, &newParams));
  }

  // practically all of the updates are kernel param updates, which are batched into a single device-side call:
  size_t totalUpdatesSize = kernel_param_updates_.size() * sizeof(cudaGraphKernelNodeUpdate);
  cudaGraphKernelNodeUpdate* hostUpdates = (cudaGraphKernelNodeUpdate*) at::getCPUAllocator()->raw_allocate(totalUpdatesSize);
  for (size_t i = 0; i < kernel_param_updates_.size(); i++) {
    auto update = kernel_param_updates_[i];
    hostUpdates[i] = cudaGraphKernelNodeUpdate{
      .node = update.dev_node,
      .field = cudaGraphKernelNodeFieldParam,
      .updateData = {
        .param = {
          .pValue = (char*)actualDataPtrs[update.alloc_idx] + update.offset, // the kernel will overwrite this to indirect it in GPU memory
          .offset = update.param_offset,
          .size = sizeof(void*),
        },
      },
    };
  }
  cudaGraphKernelNodeUpdate* deviceUpdates = (cudaGraphKernelNodeUpdate*) allocator->raw_alloc_with_stream(totalUpdatesSize, stream);
  AT_CUDA_CHECK(cudaMemcpyAsync(deviceUpdates, hostUpdates, totalUpdatesSize, cudaMemcpyHostToDevice, stream));
  AT_CUDA_CHECK(cudaLaunchHostFunc(stream, hostMemoryFreeCallback, hostUpdates)); // free once the memcpy is done

  AT_CUDA_CHECK(cudaGraphUpload(graph_exec_, stream));

  dynamic_graph_updater(deviceUpdates, kernel_param_updates_.size());

  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
  allocator->raw_delete(deviceUpdates);
  for (size_t i = num_dynamic_args_; i < actualDataPtrs.size(); i++) { // don't free prefilled
    allocator->raw_delete(actualDataPtrs[i]);
  }
}

} // namespace at::cuda
