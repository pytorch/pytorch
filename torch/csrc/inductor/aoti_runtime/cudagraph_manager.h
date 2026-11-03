#pragma once

#ifdef USE_CUDA

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>
#include <torch/csrc/stable/c/shim.h>

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
  // also makes one eager call, of any key, before its first capture, since
  // libraries such as cuBLAS set up per thread.
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
// clones the outputs out. After a capture fails, for whatever reason and even
// if the call then fails eagerly too, keys without a graph run eagerly, until
// reset(). While other graphs exist, the pool also keeps the failed capture's
// memory until then. Unlike torch.cuda.graph, a capture doesn't empty the
// allocator's cache first, so it can run out of memory where an eager call
// wouldn't, and that too stops capturing.
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
  // manager serves one fn with a fixed number of outputs, every input is a
  // CUDA tensor on the manager's device, and `stream` must not be capturing.
  // Replayed kernels run at default priority, whatever `stream`'s priority. fn
  // borrows `inputs`, writes new references into `outputs`, enqueues all its
  // GPU work on `stream`, returns only tensors on the manager's device (or
  // capturing stops), and must not mutate its inputs: a replay would only
  // mutate the graph's copies. fn may also see views over the graph's buffers,
  // at other storage offsets than the caller's inputs. Graphs have no random
  // number state, since replays would advance the generator without its lock,
  // so a capture of fn that uses random numbers fails.
  // Each output slot gets its own handle; replayed outputs never alias the
  // inputs, nor each other unless fn wrote the same handle twice. A graph keeps
  // reading everything else fn read at capture, e.g. constants, at the same
  // addresses, so call reset() before replacing or freeing any of it, with no
  // run() in flight or starting until the replacement is done. run(),
  // reset() and the destructor may delete the pool, which the shim forbids
  // while the calling thread has a capture open, or while any thread has a
  // global-mode capture open. Eager calls don't hold the manager's lock, so fn
  // must be safe to call from several threads at once, including while another
  // thread captures it. On success the caller owns every non-null slot; a
  // failed call leaves every slot null, deleting any handle fn wrote.
  template <typename Fn>
  void run(
      Fn&& fn,
      AtenTensorHandle* inputs,
      size_t num_inputs,
      AtenTensorHandle* outputs,
      size_t num_outputs,
      cudaStream_t stream) {
    std::fill(outputs, outputs + num_outputs, nullptr);
    // A null stream means the legacy default stream of the current device.
    AOTICudaGuard device_guard(device_index_);
    AOTI_RUNTIME_CHECK(
        !is_stream_capturing(stream),
        "CudaGraphManager can't run inside a CUDA graph capture");

    auto key = make_key(inputs, num_inputs);
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {
      bool eager = capture_failed_;
      if (!eager &&
          graphs_.size() >= static_cast<size_t>(options_.max_graphs)) {
        warn_max_graphs();
        eager = true;
      }
      bool warming_up = false;
      if (!eager) {
        if (warmup_counts_.size() >= kMaxWarmupKeys &&
            warmup_counts_.count(key) == 0) {
          warmup_counts_.clear();
        }
        warming_up = warmup_counts_[key] < options_.warmup_calls ||
            (options_.warmup_calls > 0 &&
             warm_threads_.count(thread_token()) == 0);
        eager = warming_up;
      }
      if (eager) {
        lock.unlock();
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
        lock.lock();
        if (warming_up) {
          ++warmup_counts_[key];
        }
        mark_thread_warm();
        return;
      }
      warmup_counts_.erase(key);
      std::unique_ptr<Graph> graph;
      try {
        graph = capture(fn, inputs, num_inputs, num_outputs, stream);
      } catch (...) {
        // A failed capture ran nothing, so run eagerly; an error there is the
        // caller's. Without graphs, the pool only holds the failed capture's
        // memory.
        capture_failed_ = true;
        if (graphs_.empty()) {
          pool_.reset();
        }
        std::string error = "unknown error";
        try {
          throw;
        } catch (const std::exception& e) {
          error = e.what();
        } catch (...) {
        }
        std::string msg =
            "AOTInductor CUDA graphs: capturing failed, so calls with new "
            "input layouts (dtypes, sizes or strides) run eagerly from now "
            "on (code that uses random numbers, for one, can't be "
            "captured): " +
            error.substr(0, error.find('\n'));
        aoti_torch_warn(
            __func__, __FILE__, static_cast<uint32_t>(__LINE__), msg.c_str());
        lock.unlock();
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
        lock.lock();
        mark_thread_warm();
        return;
      }
      it = graphs_.emplace(std::move(key), std::move(graph)).first;
    }
    replay(*it->second, inputs, outputs, num_outputs, stream);
  }

  // Drops all graphs and returns the pool's memory, including any a failed
  // capture left; the next capture starts a new pool.
  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    AOTICudaGuard device_guard(device_index_);
    // The last replay may still be using the graphs' buffers.
    AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(last_replay_));
    graphs_.clear();
    pool_.reset();
    warmup_counts_.clear();
    capture_failed_ = false;
  }

  size_t num_graphs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return graphs_.size();
  }

  // Caps the keys and threads remembered for warm-up, so one-off shapes or
  // threads can't grow the manager without bound.
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
    // Buffers covering each input's storage extent, and the views over them
    // that the region was captured with.
    std::vector<RAIIAtenTensorHandle> input_buffers;
    std::vector<RAIIAtenTensorHandle> static_inputs;
    std::vector<int64_t> input_extents;
    // Whether each input overlaps itself, e.g. an expand.
    std::vector<bool> input_overlaps;
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
  static bool overlaps(const TensorLayout& layout) {
    std::vector<std::pair<int64_t, int64_t>> dims; // (stride, size)
    for (int64_t d = 0; d < layout.dim; ++d) {
      if (layout.sizes[d] > 1) {
        dims.emplace_back(layout.strides[d], layout.sizes[d]);
      }
    }
    std::sort(dims.begin(), dims.end());
    int64_t max_offset = 0;
    for (const auto& [stride, size] : dims) {
      if (stride <= max_offset) {
        return true;
      }
      max_offset += (size - 1) * stride;
    }
    return false;
  }

  // An overlapping input's view overlaps itself, so copy the block it covers.
  static RAIIAtenTensorHandle block_of(AtenTensorHandle input, int64_t extent) {
    int64_t stride = 1;
    AtenTensorHandle block = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch__reinterpret_tensor(input, 1, &extent, &stride, 0, &block));
    return RAIIAtenTensorHandle(block);
  }

  static void copy_inputs(Graph& graph, AtenTensorHandle* inputs) {
    for (size_t i = 0; i < graph.input_overlaps.size(); ++i) {
      if (graph.input_overlaps[i]) {
        auto block = block_of(inputs[i], graph.input_extents[i]);
        AOTI_TORCH_ERROR_CODE_CHECK(
            aoti_torch_copy_(graph.input_buffers[i], block.get(), 0));
      } else {
        AOTI_TORCH_ERROR_CODE_CHECK(
            aoti_torch_copy_(graph.static_inputs[i], inputs[i], 0));
      }
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
      cudaStream_t stream) {
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
  }

  // Only a call that succeeded did the thread's lazy setup.
  void mark_thread_warm() {
    if (warm_threads_.size() >= kMaxWarmupKeys) {
      warm_threads_.clear();
    }
    warm_threads_.insert(thread_token());
  }

  // Unlike AOTI_TORCH_ERROR_CODE_CHECK, keeps the shim's message, which the
  // capture-failure warning shows.
  static void check_shim(AOTITorchError error, const char* call) {
    if (error != AOTI_TORCH_SUCCESS) {
      const char* what = torch_exception_get_what_without_backtrace();
      AOTI_RUNTIME_CHECK(
          false,
          std::string(call) + " failed: " + (what != nullptr ? what : ""));
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
        ", so calls with new input layouts (dtypes, sizes or strides) run "
        "eagerly";
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
      CUDAMemPoolHandle pool = nullptr;
      check_shim(
          aoti_torch_create_cuda_mem_pool(device_index_, &pool),
          "aoti_torch_create_cuda_mem_pool");
      pool_.reset(pool);
    }
    auto graph = std::make_unique<Graph>();
    CUDAGraphHandle graph_handle = nullptr;
    check_shim(
        aoti_torch_create_cuda_graph(
            /*keep_graph=*/0, /*num_rng_states=*/0, &graph_handle),
        "aoti_torch_create_cuda_graph");
    graph->graph.reset(graph_handle);

    std::vector<AtenTensorHandle> raw_outputs(num_outputs, nullptr);
    {
      // Allocated outside the capture, for copies outside the graph to write,
      // and on the caller's stream rather than the manager's, which is
      // destroyed with it.
      AOTICudaStreamGuard stream_guard(stream, device_index_);
      std::vector<AtenTensorHandle> input_handles;
      input_handles.reserve(num_inputs);
      for (size_t i = 0; i < num_inputs; ++i) {
        auto layout = layout_of(inputs[i]);
        int64_t extent = storage_extent(layout);
        int64_t stride = 1;
        AtenTensorHandle buffer = nullptr;
        check_shim(
            aoti_torch_empty_strided(
                1,
                &extent,
                &stride,
                layout.dtype,
                aoti_torch_device_type_cuda(),
                device_index_,
                &buffer),
            "aoti_torch_empty_strided");
        graph->input_buffers.emplace_back(buffer);
        AtenTensorHandle view = nullptr;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__reinterpret_tensor(
            buffer, layout.dim, layout.sizes, layout.strides, 0, &view));
        graph->static_inputs.emplace_back(view);
        graph->input_extents.push_back(extent);
        graph->input_overlaps.push_back(overlaps(layout));
        input_handles.push_back(view);
      }

      // The region's allocations come from the pool.
      AOTICudaStreamGuard capture_guard(capture_stream_, device_index_);
      check_shim(
          aoti_torch_cuda_graph_capture_begin(
              graph_handle, pool_.get(), cudaStreamCaptureModeThreadLocal),
          "aoti_torch_cuda_graph_capture_begin");
      bool capturing = true;
      try {
        fn(input_handles.data(), raw_outputs.data(), capture_stream_);
        capturing = false;
        check_shim(
            aoti_torch_cuda_graph_capture_end(graph_handle),
            "aoti_torch_cuda_graph_capture_end");
      } catch (...) {
        // A capture must end before its graph is deleted. The region's error is
        // the one to report.
        if (capturing) {
          // A node, so ATen doesn't also warn that the graph is empty.
          (void)cudaEventRecordWithFlags(
              event_, capture_stream_, cudaEventRecordExternal);
          (void)aoti_torch_cuda_graph_capture_end(graph_handle);
          (void)cudaGetLastError();
        }
        (void)aoti_torch_delete_cuda_graph(graph->graph.release());
        Graph discarded;
        steal_outputs(raw_outputs, discarded);
        throw;
      }
    }
    steal_outputs(raw_outputs, *graph);
    for (auto& output : graph->static_outputs) {
      // Work on another device would run only at capture.
      int32_t device_type = 0;
      int32_t device_index = 0;
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_type(output, &device_type));
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_index(output, &device_index));
      AOTI_RUNTIME_CHECK(
          device_type == aoti_torch_device_type_cuda() &&
              device_index == device_index_,
          "a region output is not on the manager's device");
    }
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
      AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(last_replay_, stream));
    } catch (...) {
      delete_outputs(outputs, num_outputs);
      // Later calls must still wait for whatever this one enqueued.
      (void)cudaEventRecord(last_replay_, stream);
      throw;
    }
  }

  int32_t device_index_;
  CudaGraphOptions options_;
  cudaStream_t capture_stream_ = nullptr;
  // Recorded only into abandoned captures.
  cudaEvent_t event_ = nullptr;
  cudaEvent_t last_replay_ = nullptr;
  // Shared by all graphs, created with the first one.
  std::unique_ptr<CUDAMemPoolOpaque, DeleterFnPtr> pool_{
      nullptr,
      delete_cuda_mem_pool};
  std::mutex mutex_;
  bool warned_max_graphs_ = false;
  bool capture_failed_ = false;
  std::map<std::vector<int64_t>, int64_t> warmup_counts_;
  std::set<uint64_t> warm_threads_;
  std::map<std::vector<int64_t>, std::unique_ptr<Graph>> graphs_;
};

} // namespace torch::aot_inductor

#endif // USE_CUDA
