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
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace torch::aot_inductor {

struct CudaGraphOptions {
  // Eager calls with a new key before it is captured, so that lazy
  // initialization such as kernel loading happens outside the capture. Unless
  // this is 0, a thread also makes an eager call before its first capture,
  // since libraries such as cuBLAS create their handles per thread.
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

// Runs a capturable piece of a model instance eagerly or as CUDA graphs, with
// one graph per key: the inputs' dtypes, sizes and strides. The graphs share
// one memory pool, so each replay waits for the previous one, whatever stream
// it ran on, copies its inputs in right before and clones its outputs right
// after. A key whose capture fails but whose eager call succeeds runs eagerly
// from then on.
class CudaGraphManager {
 public:
  explicit CudaGraphManager(int32_t device_index, CudaGraphOptions options = {})
      : device_index_(device_index), options_(checked(options)) {
    AOTICudaGuard device_guard(device_index_);
    // Owned rather than taken from the stream pool, so no other thread can
    // enqueue work into a capture. Non-blocking, as the shim requires:
    // while a blocking stream captures, work that other threads launch on the
    // legacy default stream would invalidate the capture.
    AOTI_RUNTIME_CUDA_CHECK(
        cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
    AOTI_RUNTIME_CUDA_CHECK(
        cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    AOTI_RUNTIME_CUDA_CHECK(
        cudaEventCreateWithFlags(&last_replay_, cudaEventDisableTiming));
  }

  ~CudaGraphManager() {
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
    cudaEventDestroy(last_replay_);
    cudaEventDestroy(event_);
    cudaStreamDestroy(capture_stream_);
  }

  CudaGraphManager(const CudaGraphManager&) = delete;
  CudaGraphManager& operator=(const CudaGraphManager&) = delete;
  CudaGraphManager(CudaGraphManager&&) = delete;
  CudaGraphManager& operator=(CudaGraphManager&&) = delete;

  // Calls fn(inputs, outputs, stream), or replays a capture of it. A manager
  // serves one fn with a fixed number of outputs. fn borrows `inputs`, writes
  // new references into `outputs`, and must enqueue all of its GPU work on the
  // stream it is given, which may be the capture stream.
  // Callers must make sure fn doesn't mutate its inputs: a replay runs on the
  // graph's copies of them, so the caller's tensors would miss the change.
  // Replayed outputs never alias the inputs, and alias each other only where
  // fn wrote the same handle into several output slots; eager calls return
  // whatever fn returns.
  template <typename Fn>
  void run(
      Fn&& fn,
      AtenTensorHandle* inputs,
      size_t num_inputs,
      AtenTensorHandle* outputs,
      size_t num_outputs,
      cudaStream_t stream) {
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
        run_eagerly(fn, inputs, outputs, stream);
        return;
      }
      if (graphs_.size() >= static_cast<size_t>(options_.max_graphs)) {
        warn_max_graphs();
        run_eagerly(fn, inputs, outputs, stream);
        return;
      }
      if (warmup_counts_.size() >= kMaxWarmupKeys &&
          warmup_counts_.count(key) == 0) {
        warmup_counts_.clear();
      }
      int64_t& warmups = warmup_counts_[key];
      bool thread_warm = options_.warmup_calls == 0 ||
          warm_threads_.count(std::this_thread::get_id()) > 0;
      if (warmups < options_.warmup_calls || !thread_warm) {
        ++warmups;
        run_eagerly(fn, inputs, outputs, stream);
        return;
      }
      warmup_counts_.erase(key);
      std::unique_ptr<Graph> graph;
      try {
        graph = capture(fn, inputs, num_inputs, num_outputs, stream);
      } catch (const std::exception& e) {
        // A discarded capture ran nothing, so the call can still run eagerly.
        // An error there isn't about capturing, and is the caller's.
        run_eagerly(fn, inputs, outputs, stream);
        mark_uncapturable(std::move(key), e.what());
        return;
      }
      it = graphs_.emplace(std::move(key), std::move(graph)).first;
    }
    replay(*it->second, inputs, outputs, num_outputs, stream);
  }

  // Drops all graphs, e.g. when the model's constants move. The next capture
  // starts a new pool.
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    graphs_.clear();
    warmup_counts_.clear();
    uncapturable_.clear();
  }

  size_t num_graphs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return graphs_.size();
  }

  // Keys still warming up or that failed to capture, and threads that made
  // eager calls, are forgotten past this many, so one-off shapes or threads
  // can't grow the manager without bound.
  static constexpr size_t kMaxWarmupKeys = 1024;

 private:
  static constexpr size_t kNoOutput = std::numeric_limits<size_t>::max();

  static CudaGraphOptions checked(const CudaGraphOptions& options) {
    AOTI_RUNTIME_CHECK(
        options.warmup_calls >= 0, "warmup_calls must be non-negative");
    AOTI_RUNTIME_CHECK(
        options.max_graphs >= 0, "max_graphs must be non-negative");
    return options;
  }

  struct Graph {
    std::unique_ptr<CUDAGraphOpaque, DeleterFnPtr> graph{
        nullptr,
        delete_cuda_graph};
    // Pool buffers holding each input's storage extent, the input views the
    // region was captured with, and how to copy each input in.
    std::vector<RAIIAtenTensorHandle> input_buffers;
    std::vector<RAIIAtenTensorHandle> static_inputs;
    std::vector<int64_t> input_extents;
    std::vector<bool> copy_elementwise;
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

  // Whether the extent holds elements that aren't the tensor's, e.g. for a
  // slice of a larger tensor, with no element appearing twice.
  static bool has_gaps_but_no_overlap(const TensorLayout& layout) {
    std::vector<std::pair<int64_t, int64_t>> dims; // (stride, size)
    int64_t numel = 1;
    for (int64_t d = 0; d < layout.dim; ++d) {
      numel *= layout.sizes[d];
      if (layout.sizes[d] > 1) {
        dims.emplace_back(layout.strides[d], layout.sizes[d]);
      }
    }
    if (storage_extent(layout) <= numel) {
      return false;
    }
    std::sort(dims.begin(), dims.end());
    int64_t max_offset = 0;
    for (const auto& [stride, size] : dims) {
      if (stride <= max_offset) {
        return false;
      }
      max_offset += (size - 1) * stride;
    }
    return true;
  }

  // Copies input i into the graph's buffer for it. Copying the input's
  // storage extent as one 1-D block keeps any strides, including overlapping
  // ones; an input with gaps is copied element by element instead, so the gaps
  // aren't copied.
  static void copy_input(Graph& graph, size_t i, AtenTensorHandle input) {
    if (graph.copy_elementwise[i]) {
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_copy_(graph.static_inputs[i], input, 0));
      return;
    }
    int64_t stride = 1;
    AtenTensorHandle view = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor(
        input, 1, &graph.input_extents[i], &stride, 0, &view));
    RAIIAtenTensorHandle source(view);
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch_copy_(graph.input_buffers[i], source, 0));
  }

  // Rejects inputs on another device in every call, since a graph's key
  // doesn't include the device.
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
      cudaStream_t stream) {
    fn(inputs, outputs, stream);
    // Only a call that succeeded did the thread's lazy setup.
    if (warm_threads_.size() >= kMaxWarmupKeys) {
      warm_threads_.clear();
    }
    warm_threads_.insert(std::this_thread::get_id());
  }

  void mark_uncapturable(std::vector<int64_t> key, const char* error) {
    if (uncapturable_.size() >= kMaxWarmupKeys) {
      uncapturable_.clear();
    }
    uncapturable_.insert(std::move(key));
    std::string reason(error);
    reason = reason.substr(0, reason.find('\n'));
    std::string msg =
        "AOTInductor CUDA graphs: capturing failed, so calls with these input "
        "shapes run eagerly: " +
        reason;
    aoti_torch_warn(
        __func__, __FILE__, static_cast<uint32_t>(__LINE__), msg.c_str());
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
      if (pool_) {
        // Deleting the pool returns its memory to the device.
        AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(last_replay_));
        pool_.reset();
      }
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
      // ATen allocates on the current stream, so this puts the input buffers
      // below, allocated during the capture, into the capture's pool.
      AOTICudaStreamGuard stream_guard(capture_stream_, device_index_);
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_cuda_graph_capture_begin(graph_handle, pool_.get()));
      bool capturing = true;
      try {
        std::vector<AtenTensorHandle> input_handles;
        input_handles.reserve(num_inputs);
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
          graph->copy_elementwise.push_back(has_gaps_but_no_overlap(layout));
          input_handles.push_back(view);
        }
        fn(input_handles.data(), raw_outputs.data(), capture_stream_);
        capturing = false;
        AOTI_TORCH_ERROR_CODE_CHECK(
            aoti_torch_cuda_graph_capture_end(graph_handle));
      } catch (...) {
        // A capture must end before its graph is deleted. The region's error
        // explains the failure, so an error ending the capture adds nothing.
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

    // capture_begin can allocate on the capture stream outside the pool, so
    // the replay on the caller's stream must come after that stream's work.
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
      for (size_t i = 0; i < graph.input_buffers.size(); ++i) {
        copy_input(graph, i, inputs[i]);
      }
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_cuda_graph_replay(graph.graph.get()));
      // Clones, so that later replays can't overwrite the outputs. Aliasing
      // between different outputs is not preserved.
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
      for (size_t i = 0; i < num_outputs; ++i) {
        if (outputs[i] != nullptr) {
          (void)aoti_torch_delete_tensor_object(outputs[i]);
          outputs[i] = nullptr;
        }
      }
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
  std::map<std::vector<int64_t>, int64_t> warmup_counts_;
  std::set<std::vector<int64_t>> uncapturable_;
  std::set<std::thread::id> warm_threads_;
  std::map<std::vector<int64_t>, std::unique_ptr<Graph>> graphs_;
};

} // namespace torch::aot_inductor

#endif // USE_CUDA
