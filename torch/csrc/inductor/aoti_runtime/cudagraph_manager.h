#pragma once

#ifdef USE_CUDA

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace torch::aot_inductor {

struct CudaGraphOptions {
  // Eager calls per new key before capturing it, so lazy initialization,
  // e.g. kernel loading, happens outside the capture. Unless 0, each thread
  // also makes one eager call before its first capture, since libraries such
  // as cuBLAS set up per thread.
  int64_t warmup_calls = 1;
  // Calls with new keys run eagerly once this many graphs exist.
  int64_t max_graphs = 8;
};

// Deleters run in destructors, so they log errors instead of throwing.
inline void delete_cuda_graph(void* ptr) {
  if (aoti_torch_delete_cuda_graph(reinterpret_cast<CUDAGraphHandle>(ptr)) !=
      AOTI_TORCH_SUCCESS) {
    std::cerr << "CudaGraphManager: failed to delete a CUDA graph\n";
  }
}

inline void delete_cuda_mem_pool(void* ptr) {
  if (aoti_torch_delete_cuda_mem_pool(
          reinterpret_cast<CUDAMemPoolHandle>(ptr)) != AOTI_TORCH_SUCCESS) {
    std::cerr << "CudaGraphManager: failed to delete a CUDA memory pool\n";
  }
}

// Runs a capturable part of a model instance eagerly or as CUDA graphs, one
// graph per input layout (dtypes, sizes, strides). The graphs share one
// memory pool, so replays run one at a time; each copies the inputs in and
// clones the outputs out. A key whose capture fails runs eagerly from then on,
// and after kMaxCaptureFailures failures in a row, so do new keys.
class CudaGraphManager {
 public:
  explicit CudaGraphManager(int32_t device_index, CudaGraphOptions options = {})
      : device_index_(device_index), options_(checked(options)) {
    AOTICudaGuard device_guard(device_index_);
    try {
      // Owned, so no other thread enqueues work into a capture, and
      // non-blocking, as the shim requires.
      AOTI_RUNTIME_CUDA_CHECK(
          cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
      AOTI_RUNTIME_CUDA_CHECK(
          cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
      AOTI_RUNTIME_CUDA_CHECK(
          cudaEventCreateWithFlags(&last_replay_, cudaEventDisableTiming));
    } catch (...) {
      destroy_cuda_objects();
      throw;
    }
  }

  ~CudaGraphManager() {
    try {
      AOTICudaGuard device_guard(device_index_);
      if (pool_) {
        // The last replay may still be running on the caller's stream, and
        // deleting the pool returns its memory to the device.
        cudaError_t code = cudaEventSynchronize(last_replay_);
        if (code != cudaSuccess) {
          std::cerr << "CudaGraphManager: " << cudaGetErrorString(code) << '\n';
        }
      }
      graphs_.clear();
      pool_.reset();
      destroy_cuda_objects();
    } catch (const std::exception& e) {
      std::cerr << "CudaGraphManager: " << e.what() << '\n';
    }
  }

  CudaGraphManager(const CudaGraphManager&) = delete;
  CudaGraphManager& operator=(const CudaGraphManager&) = delete;
  CudaGraphManager(CudaGraphManager&&) = delete;
  CudaGraphManager& operator=(CudaGraphManager&&) = delete;

  // Calls fn(inputs, outputs, stream) eagerly or replays its capture. A
  // manager serves one fn with a fixed number of outputs. fn borrows `inputs`,
  // writes new references into `outputs`, enqueues all its GPU work on
  // `stream`, and must not mutate its inputs: a replay would only mutate the
  // graph's copies. fn may also see views over the graph's buffers, at other
  // storage offsets than the caller's inputs, and must not use random numbers:
  // replays advance the generator without its lock. Each output slot gets its
  // own handle; replayed outputs never alias the inputs, nor each other unless
  // fn wrote the same handle twice.
  template <typename Fn>
  void run(
      Fn&& fn,
      AtenTensorHandle* inputs,
      size_t num_inputs,
      AtenTensorHandle* outputs,
      size_t num_outputs,
      cudaStream_t stream) {
    // A null stream means the legacy default stream of the current device.
    AOTICudaGuard device_guard(device_index_);
    if (is_stream_capturing(stream)) {
      // The caller is capturing the whole call itself.
      fn(inputs, outputs, stream);
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto key = make_key(inputs, num_inputs);
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {
      if (uncapturable_.count(key) > 0) {
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
        return;
      }
      if (graphs_.size() >= static_cast<size_t>(options_.max_graphs)) {
        warn_max_graphs();
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
        return;
      }
      if (capture_failures_ >= kMaxCaptureFailures) {
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
        return;
      }
      if (warmup_counts_.size() >= kMaxWarmupKeys &&
          warmup_counts_.count(key) == 0) {
        warmup_counts_.clear();
      }
      int64_t& warmups = warmup_counts_[key];
      bool thread_warm =
          options_.warmup_calls == 0 || warm_threads_.count(thread_token()) > 0;
      if (warmups < options_.warmup_calls || !thread_warm) {
        ++warmups;
        run_eagerly(fn, inputs, outputs, num_outputs, stream, /*warmup=*/true);
        return;
      }
      warmup_counts_.erase(key);
      std::unique_ptr<Graph> graph;
      try {
        graph = capture(fn, inputs, num_inputs, num_outputs, stream);
      } catch (const std::exception& e) {
        // A failed capture ran nothing, so run eagerly; an error here is the
        // caller's. Without graphs, the pool only holds the failed capture's
        // memory.
        if (graphs_.empty()) {
          release_pool();
        }
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
        mark_uncapturable(std::move(key), e.what());
        return;
      }
      capture_failures_ = 0;
      it = graphs_.emplace(std::move(key), std::move(graph)).first;
    }
    replay(*it->second, inputs, outputs, num_outputs, stream);
  }

  // Drops all graphs; the next capture starts a new pool.
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    // The last replay may still be using the graphs' buffers.
    AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(last_replay_));
    graphs_.clear();
    release_pool();
    warmup_counts_.clear();
    uncapturable_.clear();
    capture_failures_ = 0;
  }

  size_t num_graphs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return graphs_.size();
  }

  // Caps the keys and threads remembered for warm-up, so one-off shapes or
  // threads can't grow the manager without bound. Failed captures are bounded
  // by kMaxCaptureFailures per successful one.
  static constexpr size_t kMaxWarmupKeys = 1024;
  static constexpr int kMaxCaptureFailures = 8;

 private:
  static constexpr size_t kNoOutput = std::numeric_limits<size_t>::max();

  static CudaGraphOptions checked(const CudaGraphOptions& options) {
    AOTI_RUNTIME_CHECK(
        options.warmup_calls >= 0, "warmup_calls must be non-negative");
    AOTI_RUNTIME_CHECK(
        options.max_graphs >= 0, "max_graphs must be non-negative");
    return options;
  }

  // How an input gets into its buffer: a dense one straight into the view,
  // an overlapping one as a 1-D block of its extent, and one with gaps, e.g.
  // a slice, element by element.
  enum class InputCopy { kDense, kBlock, kElementwise };

  // Same-dtype inputs, copied in with one fused _foreach_copy_.
  struct CopyGroup {
    std::vector<size_t> inputs;
    std::vector<AtenTensorHandle> destinations;
  };

  struct Graph {
    std::unique_ptr<CUDAGraphOpaque, DeleterFnPtr> graph{
        nullptr,
        delete_cuda_graph};
    // Buffers covering each input's storage extent, and the views over them
    // that the region was captured with.
    std::vector<RAIIAtenTensorHandle> input_buffers;
    std::vector<RAIIAtenTensorHandle> static_inputs;
    std::vector<int64_t> input_extents;
    std::vector<InputCopy> input_copies;
    std::vector<CopyGroup> copy_groups;
    std::vector<size_t> elementwise_inputs;
    // Distinct outputs, and for each output slot its index into them, or
    // kNoOutput for a null output.
    std::vector<RAIIAtenTensorHandle> static_outputs;
    std::vector<size_t> output_slots;
  };

  struct TensorLayout {
    int32_t dtype = 0;
    int64_t dim = 0;
    int64_t* sizes = nullptr;
    int64_t* strides = nullptr;
  };

  static TensorLayout layout_of(AtenTensorHandle tensor) {
    TensorLayout layout;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(tensor, &layout.dtype));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(tensor, &layout.dim));
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(tensor, &layout.sizes));
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_strides(tensor, &layout.strides));
    return layout;
  }

  // Elements from a tensor's storage offset to its last element.
  static int64_t storage_extent(const TensorLayout& layout) {
    int64_t extent = 1;
    for (int64_t d = 0; d < layout.dim; ++d) {
      if (layout.sizes[d] == 0) {
        return 0;
      }
      extent += (layout.sizes[d] - 1) * layout.strides[d];
    }
    return extent;
  }

  // Possible overlap counts as overlap, which only costs a block copy.
  static InputCopy input_copy(const TensorLayout& layout) {
    std::vector<std::pair<int64_t, int64_t>> dims; // (stride, size)
    int64_t numel = 1;
    for (int64_t d = 0; d < layout.dim; ++d) {
      numel *= layout.sizes[d];
      if (layout.sizes[d] > 1) {
        dims.emplace_back(layout.strides[d], layout.sizes[d]);
      }
    }
    std::sort(dims.begin(), dims.end());
    int64_t max_offset = 0;
    for (const auto& [stride, size] : dims) {
      if (stride <= max_offset) {
        return InputCopy::kBlock;
      }
      max_offset += (size - 1) * stride;
    }
    return storage_extent(layout) > numel ? InputCopy::kElementwise
                                          : InputCopy::kDense;
  }

  // Groups the dense and overlapping inputs by dtype for _foreach_copy_.
  static void group_input_copies(
      Graph& graph,
      const std::vector<int32_t>& dtypes) {
    std::map<int32_t, size_t> group_of_dtype;
    for (size_t i = 0; i < graph.input_copies.size(); ++i) {
      if (graph.input_copies[i] == InputCopy::kElementwise) {
        graph.elementwise_inputs.push_back(i);
        continue;
      }
      auto [it, added] =
          group_of_dtype.try_emplace(dtypes[i], graph.copy_groups.size());
      if (added) {
        graph.copy_groups.emplace_back();
      }
      auto& group = graph.copy_groups[it->second];
      group.inputs.push_back(i);
      group.destinations.push_back(
          graph.input_copies[i] == InputCopy::kDense
              ? graph.static_inputs[i].get()
              : graph.input_buffers[i].get());
    }
  }

  static void copy_inputs(Graph& graph, AtenTensorHandle* inputs) {
    std::vector<RAIIAtenTensorHandle> blocks;
    std::vector<AtenTensorHandle> sources;
    for (auto& group : graph.copy_groups) {
      sources.clear();
      for (size_t i : group.inputs) {
        if (graph.input_copies[i] == InputCopy::kDense) {
          sources.push_back(inputs[i]);
          continue;
        }
        int64_t stride = 1;
        AtenTensorHandle block = nullptr;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor(
            inputs[i], 1, &graph.input_extents[i], &stride, 0, &block));
        blocks.emplace_back(block);
        sources.push_back(block);
      }
      auto n = static_cast<int64_t>(sources.size());
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cuda_foreach_copy_(
          group.destinations.data(), sources.data(), n));
    }
    for (size_t i : graph.elementwise_inputs) {
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_copy_(graph.static_inputs[i], inputs[i], 0));
    }
  }

  // Keys don't include the device, so every call checks it.
  std::vector<int64_t> make_key(AtenTensorHandle* inputs, size_t num_inputs) {
    std::vector<int64_t> key;
    for (size_t i = 0; i < num_inputs; ++i) {
      AOTI_RUNTIME_CHECK(inputs[i] != nullptr, "CUDA graph input is null");
      int32_t device_type = 0;
      int32_t device_index = 0;
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_type(inputs[i], &device_type));
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_index(inputs[i], &device_index));
      AOTI_RUNTIME_CHECK(
          device_type == aoti_torch_device_type_cuda() &&
              device_index == device_index_,
          "CUDA graph input " + std::to_string(i) +
              " is not on the model's CUDA device");
      auto layout = layout_of(inputs[i]);
      key.push_back(layout.dtype);
      key.push_back(layout.dim);
      key.insert(key.end(), layout.sizes, layout.sizes + layout.dim);
      key.insert(key.end(), layout.strides, layout.strides + layout.dim);
    }
    return key;
  }

  template <typename Fn>
  void run_eagerly(
      Fn& fn,
      AtenTensorHandle* inputs,
      AtenTensorHandle* outputs,
      size_t num_outputs,
      cudaStream_t stream,
      bool warmup = false) {
    std::fill(outputs, outputs + num_outputs, nullptr);
    try {
      fn(inputs, outputs, stream);
      // Each slot gets its own handle, as after a replay.
      for (size_t i = 1; i < num_outputs; ++i) {
        if (outputs[i] != nullptr &&
            std::find(outputs, outputs + i, outputs[i]) != outputs + i) {
          AOTI_TORCH_ERROR_CODE_CHECK(
              aoti_torch_new_tensor_handle(outputs[i], &outputs[i]));
        }
      }
    } catch (...) {
      delete_outputs(outputs, num_outputs);
      throw;
    }
    // Only a call that succeeded did the thread's lazy setup.
    if (warm_threads_.size() >= kMaxWarmupKeys) {
      warm_threads_.clear();
    }
    warm_threads_.insert(thread_token());
  }

  void mark_uncapturable(std::vector<int64_t> key, const char* error) {
    uncapturable_.insert(std::move(key));
    ++capture_failures_;
    std::string msg;
    if (capture_failures_ == kMaxCaptureFailures) {
      msg = "AOTInductor CUDA graphs: capturing failed " +
          std::to_string(kMaxCaptureFailures) +
          " times in a row, so calls with new input shapes run eagerly too";
    } else if (!warned_capture_failure_) {
      warned_capture_failure_ = true;
      std::string reason(error);
      msg =
          "AOTInductor CUDA graphs: capturing failed, so calls with these "
          "input shapes run eagerly: " +
          reason.substr(0, reason.find('\n'));
    } else {
      return;
    }
    aoti_torch_warn(
        __func__, __FILE__, static_cast<uint32_t>(__LINE__), msg.c_str());
  }

  // Deleting the pool returns its memory to the device.
  void release_pool() {
    if (pool_) {
      AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(last_replay_));
      pool_.reset();
    }
  }

  void destroy_cuda_objects() {
    if (last_replay_ != nullptr) {
      (void)cudaEventDestroy(last_replay_);
    }
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
    if (capture_stream_ != nullptr) {
      (void)cudaStreamDestroy(capture_stream_);
    }
  }

  void warn_max_graphs() {
    if (warned_max_graphs_) {
      return;
    }
    warned_max_graphs_ = true;
    std::string msg = "AOTInductor CUDA graphs: reached max_graphs=" +
        std::to_string(options_.max_graphs) +
        ", so calls with new input shapes run eagerly";
    aoti_torch_warn(
        __func__, __FILE__, static_cast<uint32_t>(__LINE__), msg.c_str());
  }

  // std::thread::id values are reused once a thread exits, so a new thread
  // could look warm.
  static uint64_t thread_token() {
    static std::atomic<uint64_t> next_token{0};
    thread_local uint64_t token = ++next_token;
    return token;
  }

  // Deletes each distinct output once and nulls every slot.
  static void delete_outputs(AtenTensorHandle* outputs, size_t num_outputs) {
    for (size_t i = 0; i < num_outputs; ++i) {
      if (outputs[i] != nullptr &&
          std::find(outputs, outputs + i, outputs[i]) == outputs + i) {
        (void)aoti_torch_delete_tensor_object(outputs[i]);
      }
    }
    std::fill(outputs, outputs + num_outputs, nullptr);
  }

  // Takes ownership of each distinct output once; generated code can write
  // the same handle into several output slots.
  static void steal_outputs(
      std::vector<AtenTensorHandle>& raw_outputs,
      Graph& graph) {
    for (size_t i = 0; i < raw_outputs.size(); ++i) {
      size_t slot = kNoOutput;
      if (raw_outputs[i] != nullptr) {
        for (size_t j = 0; j < i; ++j) {
          if (raw_outputs[j] == raw_outputs[i]) {
            slot = graph.output_slots[j];
            break;
          }
        }
        if (slot == kNoOutput) {
          slot = graph.static_outputs.size();
          graph.static_outputs.emplace_back(raw_outputs[i]);
        }
      }
      graph.output_slots.push_back(slot);
    }
  }

  template <typename Fn>
  std::unique_ptr<Graph> capture(
      Fn& fn,
      AtenTensorHandle* inputs,
      size_t num_inputs,
      size_t num_outputs,
      cudaStream_t stream) {
    // A pool can't take a capture once every graph captured into it is gone.
    if (graphs_.empty()) {
      release_pool();
      CUDAMemPoolHandle pool = nullptr;
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_create_cuda_mem_pool(device_index_, &pool));
      pool_.reset(pool);
    }
    auto graph = std::make_unique<Graph>();
    CUDAGraphHandle graph_handle = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_create_cuda_graph(&graph_handle));
    graph->graph.reset(graph_handle);

    AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(event_, stream));
    AOTI_RUNTIME_CUDA_CHECK(cudaStreamWaitEvent(capture_stream_, event_, 0));
    std::vector<AtenTensorHandle> raw_outputs(num_outputs, nullptr);
    {
      AOTICudaStreamGuard stream_guard(capture_stream_, device_index_);
      // Allocated outside the capture: under cudaMallocAsync, a capture's
      // allocations become graph memory nodes, which copies outside the graph
      // can't safely write.
      std::vector<AtenTensorHandle> input_handles;
      input_handles.reserve(num_inputs);
      std::vector<int32_t> dtypes;
      for (size_t i = 0; i < num_inputs; ++i) {
        auto layout = layout_of(inputs[i]);
        int64_t extent = storage_extent(layout);
        int64_t stride = 1;
        AtenTensorHandle buffer = nullptr;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
            1,
            &extent,
            &stride,
            layout.dtype,
            aoti_torch_device_type_cuda(),
            device_index_,
            &buffer));
        graph->input_buffers.emplace_back(buffer);
        AtenTensorHandle view = nullptr;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor(
            buffer, layout.dim, layout.sizes, layout.strides, 0, &view));
        graph->static_inputs.emplace_back(view);
        graph->input_extents.push_back(extent);
        graph->input_copies.push_back(input_copy(layout));
        dtypes.push_back(layout.dtype);
        input_handles.push_back(view);
      }
      group_input_copies(*graph, dtypes);

      // The region's allocations come from the pool.
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_cuda_graph_capture_begin(graph_handle, pool_.get()));
      bool capturing = true;
      try {
        fn(input_handles.data(), raw_outputs.data(), capture_stream_);
        capturing = false;
        AOTI_TORCH_ERROR_CODE_CHECK(
            aoti_torch_cuda_graph_capture_end(graph_handle));
      } catch (...) {
        // A capture must end before its graph is deleted. The region's error is
        // the one to report.
        if (capturing) {
          (void)aoti_torch_cuda_graph_capture_end(graph_handle);
        }
        (void)aoti_torch_delete_cuda_graph(graph->graph.release());
        Graph discarded;
        steal_outputs(raw_outputs, discarded);
        throw;
      }
    }
    steal_outputs(raw_outputs, *graph);

    // capture_begin may allocate on the capture stream outside the pool, so
    // the caller's stream waits for that work.
    AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(event_, capture_stream_));
    AOTI_RUNTIME_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
    return graph;
  }

  void replay(
      Graph& graph,
      AtenTensorHandle* inputs,
      AtenTensorHandle* outputs,
      size_t num_outputs,
      cudaStream_t stream) {
    AOTI_RUNTIME_CHECK(
        num_outputs == graph.output_slots.size(),
        "CudaGraphManager: a graph was captured with a different number of "
        "outputs");
    AOTICudaStreamGuard stream_guard(stream, device_index_);
    AOTI_RUNTIME_CUDA_CHECK(cudaStreamWaitEvent(stream, last_replay_, 0));
    std::fill(outputs, outputs + num_outputs, nullptr);
    try {
      copy_inputs(graph, inputs);
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_cuda_graph_replay(graph.graph.get()));
      // Clones, so later replays can't overwrite the returned outputs.
      std::vector<RAIIAtenTensorHandle> clones;
      clones.reserve(graph.static_outputs.size());
      for (auto& output : graph.static_outputs) {
        AtenTensorHandle clone = nullptr;
        AOTI_TORCH_ERROR_CODE_CHECK(
            aoti_torch_clone_preserve_strides(output, &clone));
        clones.emplace_back(clone);
      }
      for (size_t i = 0; i < num_outputs; ++i) {
        if (graph.output_slots[i] != kNoOutput) {
          AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_tensor_handle(
              clones[graph.output_slots[i]], &outputs[i]));
        }
      }
    } catch (...) {
      delete_outputs(outputs, num_outputs);
      // Later calls must still wait for whatever this one enqueued.
      (void)cudaEventRecord(last_replay_, stream);
      throw;
    }
    AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(last_replay_, stream));
  }

  int32_t device_index_;
  CudaGraphOptions options_;
  cudaStream_t capture_stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
  cudaEvent_t last_replay_ = nullptr;
  // Shared by all graphs, created with the first one.
  std::unique_ptr<CUDAMemPoolOpaque, DeleterFnPtr> pool_{
      nullptr,
      delete_cuda_mem_pool};
  std::mutex mutex_;
  bool warned_max_graphs_ = false;
  bool warned_capture_failure_ = false;
  int capture_failures_ = 0;
  std::map<std::vector<int64_t>, int64_t> warmup_counts_;
  std::set<std::vector<int64_t>> uncapturable_;
  std::set<uint64_t> warm_threads_;
  std::map<std::vector<int64_t>, std::unique_ptr<Graph>> graphs_;
};

} // namespace torch::aot_inductor

#endif // USE_CUDA
