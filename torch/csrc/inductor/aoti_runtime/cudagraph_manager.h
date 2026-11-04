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
// clones the outputs out. After a capture fails, for whatever reason and even
// if the call then fails eagerly too, keys without a graph run eagerly, until
// reset(). While other graphs exist, the pool also keeps the failed capture's
// memory until then. Unlike torch.cuda.graph, a capture doesn't empty the
// allocator's cache first, so it can run out of memory where an eager call
// wouldn't, and that too stops capturing.
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
  // so a capture of fn that uses random numbers fails. After a failed capture
  // fn runs again eagerly in the same call.
  // Each output slot gets its own handle; replayed outputs never alias the
  // inputs, nor each other unless fn wrote the same handle twice, while eager
  // calls return fn's outputs as they are. A graph keeps
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
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    AOTI_RUNTIME_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    AOTI_RUNTIME_CHECK(
        status == cudaStreamCaptureStatusNone,
        "CudaGraphManager: can't run inside a CUDA graph capture");

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
        if (warmups_.size() >= kMaxWarmupKeys && warmups_.count(key) == 0) {
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
            if (threads.size() >= kMaxWarmupKeys) {
              threads.clear();
            }
            threads.insert(thread_token());
          }
        }
        return;
      }
      warmups_.erase(key);
      std::unique_ptr<Graph> graph;
      std::string error;
      try {
        graph = capture(fn, inputs, num_inputs, num_outputs, stream);
      } catch (const std::exception& e) {
        error = e.what();
      }
      if (!graph) {
        // A failed capture ran nothing, so run eagerly; an error there is the
        // caller's. Without graphs, the pool only holds the failed capture's
        // memory.
        capture_failed_ = true;
        if (graphs_.empty()) {
          pool_.reset();
        }
        std::string msg =
            "AOTInductor CUDA graphs: capturing failed, so calls with new "
            "input layouts (dtypes, sizes or strides) run eagerly until "
            "reset() (code that uses random numbers, for one, can't be "
            "captured): " +
            error.substr(0, error.find('\n'));
        aoti_torch_warn(
            __func__, __FILE__, static_cast<uint32_t>(__LINE__), msg.c_str());
        lock.unlock();
        run_eagerly(fn, inputs, outputs, num_outputs, stream);
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
    AOTI_RUNTIME_CUDA_CHECK(cudaEventSynchronize(last_replay_.get()));
    graphs_.clear();
    pool_.reset();
    warmups_.clear();
    capture_failed_ = false;
    warned_max_graphs_ = false;
  }

  size_t num_graphs() {
    std::lock_guard<std::mutex> lock(mutex_);
    return graphs_.size();
  }

  // Caps the keys, and the threads per key, remembered for warm-up, so one-off
  // shapes or threads can't grow the manager without bound.
  static constexpr size_t kMaxWarmupKeys = 1024;
  // Inputs and outputs of at least this size are copied one by one: on a B200,
  // the fused kernel stops beating memcpys for a pair of them around here.
  static constexpr int64_t kMaxFusedCopyBytes = 16 << 10;

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

  // How a tensor covers its storage extent: every element once, some more than
  // once (e.g. an expand), or with gaps (e.g. a slice).
  enum class Storage { kDense, kOverlapping, kGapped };

  // Same-dtype inputs, copied in with one _foreach_copy_, which the CUDA
  // kernel fuses for the dtypes it dispatches.
  struct CopyGroup {
    std::vector<size_t> inputs;
    std::vector<AtenTensorHandle> destinations;
  };

  struct Graph {
    AOTICudaGraph graph;
    // Buffers covering each input's storage extent, and the views over them
    // that fn was captured with.
    std::vector<RAIIAtenTensorHandle> input_buffers;
    std::vector<RAIIAtenTensorHandle> static_inputs;
    std::vector<int64_t> input_extents;
    std::vector<Storage> input_storage;
    std::vector<CopyGroup> copy_groups;
    // Inputs copied one by one.
    std::vector<size_t> separate_inputs;
    // Distinct outputs, and for each output slot its index into them, or
    // kNoOutput for a null output.
    std::vector<RAIIAtenTensorHandle> static_outputs;
    std::vector<size_t> output_slots;
    // Small dense outputs, by dtype, cloned with one _foreach_copy_ per group,
    // and the others, cloned one by one.
    std::vector<std::vector<size_t>> output_groups;
    std::vector<size_t> separate_outputs;
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
  static Storage storage_of(const TensorLayout& layout) {
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
        return Storage::kOverlapping;
      }
      max_offset += (size - 1) * stride;
    }
    return storage_extent(layout) > numel ? Storage::kGapped : Storage::kDense;
  }

  // Whether an input, or a dense output, covering `extent` elements joins a
  // fused copy: it isn't empty, and is small enough that the fused kernel
  // copies it no slower than a memcpy.
  static bool fusable(int32_t dtype, int64_t extent) {
    auto element_size =
        static_cast<int64_t>(aoti_torch_dtype_element_size(dtype));
    return extent > 0 && extent * element_size < kMaxFusedCopyBytes;
  }

  // Groups tensors[i], for each i in `indices`, by dtype. A dtype with only
  // one goes to `separate`, as fusing saves nothing.
  static std::vector<std::vector<size_t>> group_by_dtype(
      const std::vector<RAIIAtenTensorHandle>& tensors,
      const std::vector<size_t>& indices,
      std::vector<size_t>& separate) {
    std::map<int32_t, size_t> group_of_dtype;
    std::vector<std::vector<size_t>> groups;
    for (size_t i : indices) {
      int32_t dtype = 0;
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(tensors[i], &dtype));
      auto [it, added] = group_of_dtype.try_emplace(dtype, groups.size());
      if (added) {
        groups.emplace_back();
      }
      groups[it->second].push_back(i);
    }
    std::vector<std::vector<size_t>> fused;
    for (auto& group : groups) {
      if (group.size() == 1) {
        separate.push_back(group[0]);
      } else {
        fused.push_back(std::move(group));
      }
    }
    return fused;
  }

  static void group_input_copies(Graph& graph) {
    std::vector<size_t> candidates;
    for (size_t i = 0; i < graph.input_storage.size(); ++i) {
      bool fuse = fusable(
          layout_of(graph.static_inputs[i]).dtype, graph.input_extents[i]);
      (fuse ? candidates : graph.separate_inputs).push_back(i);
    }
    for (auto& inputs : group_by_dtype(
             graph.static_inputs, candidates, graph.separate_inputs)) {
      CopyGroup group{std::move(inputs), {}};
      for (size_t i : group.inputs) {
        group.destinations.push_back(
            graph.input_storage[i] == Storage::kDense
                ? graph.static_inputs[i].get()
                : graph.input_buffers[i].get());
      }
      graph.copy_groups.push_back(std::move(group));
    }
  }

  static void group_output_copies(Graph& graph) {
    std::vector<size_t> candidates;
    for (size_t i = 0; i < graph.static_outputs.size(); ++i) {
      auto layout = layout_of(graph.static_outputs[i]);
      bool fuse = storage_of(layout) == Storage::kDense &&
          fusable(layout.dtype, storage_extent(layout));
      (fuse ? candidates : graph.separate_outputs).push_back(i);
    }
    graph.output_groups = group_by_dtype(
        graph.static_outputs, candidates, graph.separate_outputs);
  }

  // Clones, so later replays can't overwrite the returned outputs, with the
  // captured outputs' sizes and strides.
  std::vector<RAIIAtenTensorHandle> clone_outputs(Graph& graph) {
    std::vector<RAIIAtenTensorHandle> clones(graph.static_outputs.size());
    std::vector<AtenTensorHandle> destinations;
    std::vector<AtenTensorHandle> sources;
    for (const auto& group : graph.output_groups) {
      destinations.clear();
      sources.clear();
      for (size_t i : group) {
        auto layout = layout_of(graph.static_outputs[i]);
        AtenTensorHandle clone = nullptr;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
            layout.dim,
            layout.sizes,
            layout.strides,
            layout.dtype,
            aoti_torch_device_type_cuda(),
            device_index_,
            &clone));
        clones[i] = RAIIAtenTensorHandle(clone);
        destinations.push_back(clone);
        sources.push_back(graph.static_outputs[i].get());
      }
      auto n = static_cast<int64_t>(sources.size());
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__foreach_copy_(
          destinations.data(), n, sources.data(), n, 0));
    }
    for (size_t i : graph.separate_outputs) {
      AtenTensorHandle clone = nullptr;
      AOTI_TORCH_ERROR_CODE_CHECK(
          aoti_torch_clone_preserve_strides(graph.static_outputs[i], &clone));
      clones[i] = RAIIAtenTensorHandle(clone);
    }
    return clones;
  }

  // The block of storage an input covers, for copying a non-dense input: an
  // overlapping view can't be a copy destination, and a gapped one is copied
  // with its gaps when fused.
  static RAIIAtenTensorHandle block_of(AtenTensorHandle input, int64_t extent) {
    int64_t stride = 1;
    AtenTensorHandle block = nullptr;
    AOTI_TORCH_ERROR_CODE_CHECK(
        aoti_torch__reinterpret_tensor(input, 1, &extent, &stride, 0, &block));
    return RAIIAtenTensorHandle(block);
  }

  static void copy_inputs(Graph& graph, AtenTensorHandle* inputs) {
    std::vector<RAIIAtenTensorHandle> blocks;
    std::vector<AtenTensorHandle> sources;
    for (auto& group : graph.copy_groups) {
      sources.clear();
      for (size_t i : group.inputs) {
        if (graph.input_storage[i] == Storage::kDense) {
          sources.push_back(inputs[i]);
          continue;
        }
        blocks.push_back(block_of(inputs[i], graph.input_extents[i]));
        sources.push_back(blocks.back().get());
      }
      auto n = static_cast<int64_t>(sources.size());
      AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__foreach_copy_(
          group.destinations.data(), n, sources.data(), n, 0));
    }
    for (size_t i : graph.separate_inputs) {
      if (graph.input_storage[i] == Storage::kOverlapping) {
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
        graph->input_storage.push_back(storage_of(layout));
        input_handles.push_back(view);
      }
      group_input_copies(*graph);

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
    group_output_copies(*graph);
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
    OnExit record([&] { (void)cudaEventRecord(last_replay_.get(), stream); });
    OnExit cleanup([&] { delete_outputs(outputs, num_outputs); });
    copy_inputs(graph, inputs);
    graph.graph.replay();
    auto clones = clone_outputs(graph);
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
  AOTICudaStream capture_stream_;
  AOTICudaEvent last_replay_;
  // Shared by all graphs, created with the first one.
  std::optional<AOTICudaMemPool> pool_;
  std::mutex mutex_;
  bool warned_max_graphs_ = false;
  bool capture_failed_ = false;
  // Eager calls of each key not yet captured, and the threads that made them.
  struct Warmup {
    int64_t calls = 0;
    std::set<uint64_t> threads;
  };
  std::map<std::vector<int64_t>, Warmup> warmups_;
  std::map<std::vector<int64_t>, std::unique_ptr<Graph>> graphs_;
};

} // namespace torch::aot_inductor

#endif // USE_CUDA
