#pragma once

#ifdef USE_CUDA

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>

// Built on the CUDA graph shim functions.
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace torch::aot_inductor {

struct CudaGraphOptions {
  // Eager calls per new key before capturing it, so lazy initialization,
  // e.g. kernel loading, happens outside the capture. Unless 0, the capturing
  // thread must also have run the key eagerly, since libraries such as cuBLAS
  // set up per thread.
  int64_t warmup_calls = 1;
  // Calls with new keys run eagerly once this many graphs exist; 0 never
  // captures.
  int64_t max_graphs = 8;
};

// Runs a capturable part of a model instance eagerly or as CUDA graphs, one
// graph per input layout (dtypes, sizes, strides). The graphs share one
// memory pool, so replays run one at a time; each copies the inputs in and
// clones the outputs out. As with torch.cuda.graph, a failed capture's error
// propagates to the caller; the key captures again on its next call. A
// failed capture's memory stays in the pool until reset(), or until the next
// capture if no graph exists. Unlike torch.cuda.graph, a capture doesn't
// empty the allocator's cache first, so it can run out of memory where an
// eager call wouldn't. Under PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,
// which the shim's memory pools don't support, every capture fails.
class CudaGraphManager {
 public:
  explicit CudaGraphManager(int32_t device_index, CudaGraphOptions options = {})
      : device_index_(device_index),
        options_(checked(options)),
        capture_stream_(device_index),
        last_replay_(device_index) {}

  ~CudaGraphManager() {
    try {
      AOTICudaGuard device_guard(device_index_);
      if (pool_) {
        // The last replay may still be running on the caller's stream, and
        // deleting the pool returns its memory to the device.
        AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(last_replay_.get()));
      }
      graphs_.clear();
      pool_.reset();
    } catch (const std::exception& e) {
      std::cerr << "CudaGraphManager: " << e.what() << '\n';
    }
  }

  CudaGraphManager(const CudaGraphManager&) = delete;
  CudaGraphManager& operator=(const CudaGraphManager&) = delete;
  CudaGraphManager(CudaGraphManager&&) = delete;
  CudaGraphManager& operator=(CudaGraphManager&&) = delete;

  // Calls fn(inputs, outputs, stream) eagerly, or captures it and replays the
  // capture. Requirements:
  // - Every input is a CUDA tensor on the manager's device, and `stream` isn't
  //   capturing; a null `stream` means the device's legacy default stream.
  // - A manager serves one fn, with a fixed number of outputs.
  // - fn borrows `inputs` and writes new references into `outputs`.
  // - fn enqueues all its GPU work on the stream it is passed: `stream` in
  //   eager calls (not made current), the manager's own stream in captures
  //   (made current).
  // - fn returns only tensors on the manager's device, or its capture fails.
  // - fn doesn't mutate its inputs (a replay would only mutate the graph's
  //   copies), and doesn't read other parts of their storage: the graph's
  //   buffers hold only the inputs' elements, at other storage offsets
  //   (aot_inductor.cudagraphs rejects models that would).
  // - fn doesn't use random numbers: graphs have no RNG state, so its capture
  //   fails.
  // - fn is safe to call from several threads at once, including while another
  //   thread captures it, and doesn't call back into the manager, since
  //   captures hold the manager's lock.
  // - A graph keeps reading everything else fn read at capture, e.g.
  //   constants, at the same addresses: call reset() before replacing or
  //   freeing any of it, with no run() in flight until that's done.
  // - run(), reset() and the destructor may delete the pool, which the shim
  //   forbids while the calling thread, or any thread in global mode, has a
  //   capture open.
  // Results:
  // - On success the caller owns every non-null slot; a failed call leaves
  //   every slot null, deleting any handle fn wrote.
  // - Each slot gets its own handle. Replayed outputs are copies that never
  //   alias the inputs, nor each other unless fn wrote the same handle twice;
  //   eager calls return fn's outputs as they are.
  // - Replayed kernels run at default priority, whatever `stream`'s priority.
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
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    AOTI_RUNTIME_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    AOTI_RUNTIME_CHECK(
        status == cudaStreamCaptureStatusNone,
        "CudaGraphManager: can't run inside a CUDA graph capture");

    auto key = make_key(inputs, num_inputs);
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {
      bool eager = false;
      if (graphs_.size() >= static_cast<size_t>(options_.max_graphs)) {
        warn_max_graphs();
        eager = true;
      }
      bool warming_up = false;
      if (!eager) {
        if (warmups_.size() >= kMaxWarmupEntries && warmups_.count(key) == 0) {
          warmups_.clear();
        }
        auto& warmup = warmups_[key];
        warming_up = warmup.calls < options_.warmup_calls ||
            (options_.warmup_calls > 0 &&
             warmup.threads.count(thread_token()) == 0);
        eager = warming_up;
      }
      if (eager) {
        lock.unlock();
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
        if (warming_up) {
          // Only a call that succeeded did the thread's lazy setup. The entry
          // is gone if the key was captured or forgotten meanwhile.
          lock.lock();
          auto found = warmups_.find(key);
          if (found != warmups_.end()) {
            auto& threads = found->second.threads;
            ++found->second.calls;
            if (threads.size() >= kMaxWarmupEntries) {
              threads.clear();
            }
            threads.insert(thread_token());
          }
        }
        return;
      }
      auto graph = capture(fn, inputs, num_inputs, num_outputs, stream);
      warmups_.erase(key);
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
    AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(last_replay_.get()));
    graphs_.clear();
    pool_.reset();
    warmups_.clear();
    warned_max_graphs_ = false;
  }

  size_t num_graphs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return graphs_.size();
  }

  // Caps the keys remembered for warm-up, and the threads remembered per key.
  // A full table is mostly one-off shapes or exited threads, so it is cleared
  // rather than frozen: live keys just warm up again, while a frozen table
  // would keep any later key from ever being captured.
  static constexpr size_t kMaxWarmupEntries = 1024;

 private:
  static constexpr size_t kNoOutput = std::numeric_limits<size_t>::max();

  static CudaGraphOptions checked(const CudaGraphOptions& options) {
    AOTI_RUNTIME_CHECK(
        options.warmup_calls >= 0 && options.max_graphs >= 0,
        "CudaGraphManager: warmup_calls and max_graphs must be non-negative");
    CudaGraphOptions result = options;
#ifdef USE_ROCM
    // Capturing isn't supported on ROCm yet, so never try it.
    result.max_graphs = 0;
#endif
    return result;
  }

  struct Graph {
    AOTICudaGraph graph;
    // Buffers covering each input's storage extent, and the views over them
    // that fn was captured with.
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
      AOTI_RUNTIME_CHECK(
          inputs[i] != nullptr,
          "CudaGraphManager: input " + std::to_string(i) + " is null");
      int32_t device_type = 0;
      int32_t device_index = 0;
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_type(inputs[i], &device_type));
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_index(inputs[i], &device_index));
      AOTI_RUNTIME_CHECK(
          device_type == aoti_torch_device_type_cuda() &&
              device_index == device_index_,
          "CudaGraphManager: input " + std::to_string(i) +
              " is not on the manager's device, cuda:" +
              std::to_string(device_index_));
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
    OnExit cleanup([&] { delete_outputs(outputs, num_outputs); });
    fn(inputs, outputs, stream);
    // Each slot gets its own handle, as after a replay.
    for (size_t i = 1; i < num_outputs; ++i) {
      if (outputs[i] != nullptr &&
          std::find(outputs, outputs + i, outputs[i]) != outputs + i) {
        AOTI_TORCH_ERROR_CODE_CHECK(
            aoti_torch_new_tensor_handle(outputs[i], &outputs[i]));
      }
    }
    cleanup.dismiss();
  }

  // Runs a cleanup when it goes out of scope, unless dismissed.
  template <typename F>
  class OnExit {
   public:
    explicit OnExit(F f) : f_(std::move(f)) {}
    ~OnExit() {
      if (active_) {
        f_();
      }
    }
    OnExit(const OnExit&) = delete;
    OnExit& operator=(const OnExit&) = delete;
    void dismiss() {
      active_ = false;
    }

   private:
    F f_;
    bool active_ = true;
  };

  void warn_max_graphs() {
    if (warned_max_graphs_ || options_.max_graphs == 0) {
      return;
    }
    warned_max_graphs_ = true;
    std::string msg = "AOTInductor CUDA graphs: reached max_graphs=" +
        std::to_string(options_.max_graphs) +
        ", so calls with new input layouts (dtypes, sizes or strides) run "
        "eagerly";
#ifndef DISABLE_WARN
    aoti_torch_warn(
        __func__, __FILE__, static_cast<uint32_t>(__LINE__), msg.c_str());
#endif
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
    // Reserved first, so no handle is taken unless all are.
    graph.static_outputs.reserve(raw_outputs.size());
    graph.output_slots.reserve(raw_outputs.size());
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
      pool_.emplace(device_index_);
    }
    auto graph = std::make_unique<Graph>();

    std::vector<AtenTensorHandle> raw_outputs(num_outputs, nullptr);
    // The capture ends before fn's outputs are deleted.
    OnExit cleanup([&] {
      graph.reset(); // Ends a capture left open.
      delete_outputs(raw_outputs.data(), raw_outputs.size());
    });
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
        graph->input_overlaps.push_back(overlaps(layout));
        input_handles.push_back(view);
      }

      // The region's allocations come from the pool.
      AOTICudaStreamGuard capture_guard(capture_stream_.get(), device_index_);
      graph->graph.capture_begin(*pool_, cudaStreamCaptureModeThreadLocal);
      fn(input_handles.data(), raw_outputs.data(), capture_stream_.get());
      graph->graph.capture_end();
    }
    steal_outputs(raw_outputs, *graph);
    cleanup.dismiss();
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
          "CudaGraphManager: an output of fn is not on the manager's device");
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
        "CudaGraphManager: the call has " + std::to_string(num_outputs) +
            " outputs, but the graph was captured with " +
            std::to_string(graph.output_slots.size()));
    AOTICudaStreamGuard stream_guard(stream, device_index_);
    AOTI_RUNTIME_CUDA_CHECK(cudaStreamWaitEvent(stream, last_replay_.get(), 0));
    // Later calls wait for whatever this one enqueued, even if it fails.
    OnExit record([&] {
      log_cuda_error(
          cudaEventRecord(last_replay_.get(), stream), "cudaEventRecord");
    });
    OnExit cleanup([&] { delete_outputs(outputs, num_outputs); });
    copy_inputs(graph, inputs);
    graph.graph.replay();
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
    record.dismiss();
    AOTI_RUNTIME_CUDA_CHECK(cudaEventRecord(last_replay_.get(), stream));
    cleanup.dismiss();
  }

  int32_t device_index_;
  CudaGraphOptions options_;
  // Owned, so no other thread enqueues work into a capture.
  AOTICudaStreamNonBlocking capture_stream_;
  AOTICudaEventDisableTiming last_replay_;
  // Shared by all graphs, created with the first one.
  std::optional<AOTICudaMemPool> pool_;
  std::mutex mutex_;
  bool warned_max_graphs_ = false;
  // Eager calls of each key not yet captured, and the threads that made them.
  struct Warmup {
    int64_t calls = 0;
    std::set<uint64_t> threads;
  };
  std::map<std::vector<int64_t>, Warmup> warmups_;
  std::map<std::vector<int64_t>, std::unique_ptr<Graph>> graphs_;
};

} // namespace torch::aot_inductor

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0
#endif // USE_CUDA
