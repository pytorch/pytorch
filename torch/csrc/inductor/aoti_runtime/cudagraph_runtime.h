// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

// AOTI whole-graph cuda-graph runtime (flat, no-tree, no-checkpoint).
//
// The entire lowered component is captured as ONE cuda graph per distinct
// dynamic shape, re-captured on demand the first time a shape is seen.
//
//   * One private graph mempool per model instance is shared by every capture.
//     Intermediates allocated inside a capture come from that pool and are baked
//     into the graph; different shape-captures may reuse the same pool bytes for
//     their intermediates, which is safe because a replay always writes an
//     intermediate before reading it and replays on one instance are serialized.
//   * Dispatch is a flat map keyed by the exact tuple of dynamic-symbol values.
//     There is no recording tree: no parent/child walk, no per-node allocator
//     checkpoint, no snapshot/restore, no divergence handling. The key is a
//     vector compared for full equality, NOT a packed integer -- whole-graph
//     capture keys on every dynamic symbol (real models carry two, a request
//     batch and an item batch) and a packing collision would silently replay the
//     WRONG graph.
//   * OUTPUT CONTRACT. The caller always receives non-owning views that stay
//     valid only until the next run_graph() on this instance. On the captured
//     path they point at addresses baked into the graph, which the next replay
//     at that shape overwrites; on the uncaptured path the manager retains the
//     real handles and frees them at the start of the next call.
//
//     THE CALLER MUST COPY OUTPUTS OUT (D2D or D2H) BEFORE IT LETS ANOTHER
//     REQUEST REACH THIS INSTANCE. Nothing below enforces that, and neither
//     does the container: run_pinned()'s per-instance mutex is released the
//     moment execution finishes, which is BEFORE the caller steals the handles
//     and consumes them. So two threads sharing one instance race, and the
//     second one's replay silently overwrites the first one's results -- wrong
//     numbers, no crash, no illegal access. Sharing arises whenever worker
//     threads outnumber instances, threads churn, or several models divide up
//     one process-global worker index. Pinning narrows that window; it does
//     not close it. Copying out is the serving layer's job by design: it knows
//     whether the response needs the bytes on device or on host.
//   * CAPTURE CAP. Each capture reserves pool memory, so a model with a variable
//     serving batch would otherwise reserve without limit. Past max_captures a
//     new shape runs the body UNCAPTURED on the caller stream and warns once --
//     correct, just without the launch saving.
//   * Replay is one-directional and touches no allocator state -- the hot serving
//     path is a pure cudaGraphLaunch. The private pool keeps only capture-stream
//     ordering (sync_before_record) and the single shared cuBLAS workspace slot
//     (clear_cublas_workspaces).
//
// Header-only over the stable AOTI C ABI, so the generated model .so includes it
// and hands the graph body in as a lambda (capturing the wrapper's locals).

#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch::aot_inductor {

// Address + metadata of a captured output, enough to reconstruct a non-owning
// handle at the same device address on replay (mirrors cudagraph_trees'
// reconstruct_outputs / outputs_metadata).
struct CUDAGraphOutputMeta {
  // False for an absent output. A graph may legitimately return null for an
  // output slot (`generate_return` emits nothing for a "nullptr" output ref),
  // and a zero-size tensor has a null data_ptr of its own, so absence needs its
  // own flag rather than being inferred from data_ptr.
  bool present{false};
  void* data_ptr{nullptr};
  int64_t ndim{0};
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int32_t dtype{0};
  int32_t device_type{0};
  int32_t device_index{0};
};

inline CUDAGraphOutputMeta cuda_graph_capture_meta(AtenTensorHandle h) {
  CUDAGraphOutputMeta m;
  if (h == nullptr) {
    return m; // absent output; every accessor below would deref null
  }
  m.present = true;
  aoti_torch_get_data_ptr(h, &m.data_ptr);
  aoti_torch_get_dim(h, &m.ndim);
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  aoti_torch_get_sizes(h, &sizes);
  aoti_torch_get_strides(h, &strides);
  m.sizes.assign(sizes, sizes + m.ndim);
  m.strides.assign(strides, strides + m.ndim);
  aoti_torch_get_dtype(h, &m.dtype);
  aoti_torch_get_device_type(h, &m.device_type);
  aoti_torch_get_device_index(h, &m.device_index);
  return m;
}

// Non-owning view at the recorded address (partition outputs are fresh
// allocations, so storage_offset is 0).
inline AtenTensorHandle cuda_graph_reconstruct(const CUDAGraphOutputMeta& m) {
  AtenTensorHandle h = nullptr;
  if (!m.present) {
    return nullptr; // absent output stays absent on every replay
  }
  aoti_torch_create_tensor_from_blob(
      m.data_ptr,
      m.ndim,
      m.sizes.data(),
      m.strides.data(),
      /*storage_offset=*/0,
      m.dtype,
      m.device_type,
      m.device_index,
      &h);
  return h;
}

// One captured graph for a single dynamic shape.
struct CUDAGraphNode {
  AOTICudaGraphHandle graph{nullptr};

  // Staged input slots (fixed-address; the caller's inputs are copied into them
  // per replay). Whole-graph capture stages every tensor input, so all slots are
  // populated; the nullptr case is reserved for inputs read in place, which only
  // arises once regional capture chains one captured region into the next.
  std::vector<AtenTensorHandle> static_inputs;

  // Per output: address+metadata for reconstruct-on-replay.
  std::vector<CUDAGraphOutputMeta> output_meta;
  // Every output handle is held until teardown, so a recorded output_meta
  // data_ptr can never dangle. Outputs that escape the model (escape_outs) get
  // their deleter neutralized first, so teardown does not free memory the
  // recorded graph still refers to.
  std::vector<AtenTensorHandle> owning_outputs;

  CUDAGraphNode() = default;
  CUDAGraphNode(const CUDAGraphNode&) = delete;
  CUDAGraphNode& operator=(const CUDAGraphNode&) = delete;

  ~CUDAGraphNode() {
    for (auto h : owning_outputs) {
      if (h) {
        aoti_torch_delete_tensor_object(h);
      }
    }
    for (auto h : static_inputs) {
      if (h) {
        aoti_torch_delete_tensor_object(h);
      }
    }
    if (graph) {
      aoti_torch_cuda_graph_destroy(graph);
    }
  }
};

// Hash for the dynamic-symbol-tuple dispatch key. A collision is harmless:
// unordered_map falls back to operator== on the full vector, so it costs one
// extra compare and can never replay the wrong graph.
struct AOTICUDAGraphShapeKeyHash {
  size_t operator()(const std::vector<int64_t>& key) const noexcept {
    size_t h = 1469598103934665603ULL; // FNV-1a offset basis
    for (int64_t v : key) {
      h ^= static_cast<size_t>(v);
      h *= 1099511628211ULL; // FNV-1a prime
    }
    return h;
  }
};

// Owns the shared pool and a flat table of recordings. Mirrors
// cudagraph_trees.CUDAGraphTreeManager (single-pool, inference-only: no
// generation tracking, no runtime liveness -- liveness is a compile-time
// decision passed in via copy_in/escape_outs).
class AOTICUDAGraphManager {
 public:
  using GraphBody = std::function<
      void(AtenTensorHandle* in, AtenTensorHandle* out, void* stream)>;

  // Serving-time override for the codegen-baked cap. Returns codegen_default
  // when unset or unparseable. 0 is honoured and means "never capture": every
  // shape takes the uncaptured path, which is a kill switch for cuda graph on
  // an already-published model. Deliberately strict about trailing garbage, so
  // a typo falls back to the compiled default rather than to 0.
  static size_t resolve_max_captures(size_t codegen_default) {
    const char* env = std::getenv("AOT_INDUCTOR_CUDAGRAPH_MAX_CAPTURES");
    if (env == nullptr || env[0] == '\0') {
      return codegen_default;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0') {
      fprintf(
          stderr,
          "[cudagraph] WARNING: ignoring unparseable "
          "AOT_INDUCTOR_CUDAGRAPH_MAX_CAPTURES=\"%s\"; using the compiled "
          "default %zu.\n",
          env,
          codegen_default);
      fflush(stderr);
      return codegen_default;
    }
    return static_cast<size_t>(parsed);
  }

  // max_captures is the value baked in at lowering time
  // (config.aot_inductor.cudagraph_max_captures); the environment can override
  // it per process, since the right cap depends on host memory headroom and the
  // live shape distribution, neither of which is known when the model is built.
  AOTICUDAGraphManager(int32_t device_index, size_t max_captures)
      : device_index_(device_index),
        max_captures_(resolve_max_captures(max_captures)) {
    // On only when AOTI_CUDAGRAPH_DEBUG is set to a truthy value. An unset, empty,
    // or "0" value is off so production runs that export AOTI_CUDAGRAPH_DEBUG=0
    // (e.g. the scorecard harness) do not get the per-replay RECORD/REPLAY flood.
    const char* debug_env = std::getenv("AOTI_CUDAGRAPH_DEBUG");
    debug_ = debug_env != nullptr && debug_env[0] != '\0' &&
        !(debug_env[0] == '0' && debug_env[1] == '\0');
    // This manager's OWN private graph pool + capture stream (see shim). Owned
    // here and destroyed in the destructor -> per-AOTInductorModel-instance
    // isolation: concurrent instances in a model_container never share pool
    // memory or a capture stream.
    aoti_torch_cuda_graph_pool_create(device_index, &pool_handle_);
    // Balance the pool use_count for the destructor. Captures take a net +N on
    // the pool via cuda_graph_create (+N) that the node dtors drop (-N);
    // pool_destroy_handle then drops one more (-1). ensure_created here adds the
    // matching +1 so the count balances (N+1 vs N+1) instead of going to -1 and
    // tripping the releasePool use_count assert at teardown. It also
    // materializes the PrivatePool so the zero-recorded-nodes case has a real
    // pool for releasePool to release.
    aoti_torch_cuda_graph_pool_ensure_created(pool_handle_);
  }
  AOTICUDAGraphManager(const AOTICUDAGraphManager&) = delete;
  AOTICUDAGraphManager& operator=(const AOTICUDAGraphManager&) = delete;

  ~AOTICUDAGraphManager() {
    release_uncaptured_outputs();
    // Destroy nodes (their captured graphs) BEFORE the pool: each ~CUDAGraph
    // releases the pool ref it took at capture; pool_destroy_handle then drops
    // the remaining ref (from ensure_created) -> use_count 0 -> freed.
    flat_nodes_.clear();
    aoti_torch_cuda_graph_pool_destroy_handle(pool_handle_);
  }

  // Start of one forward. Frees the PREVIOUS forward's uncaptured outputs,
  // which is what gives the uncaptured path the same "valid until the next
  // forward" contract the captured path has.
  //
  // This is per-FORWARD, not per-run_graph, because regional mode makes several
  // run_graph calls per forward and a later partition's uncaptured outputs are
  // frequently still live as an earlier one's chained inputs. Freeing per call
  // would be a use-after-free. The generated run_impl calls this once, up front,
  // in both modes.
  void begin_forward() {
    release_uncaptured_outputs();
  }

  // Run one captured region for one dynamic shape. In whole-graph mode that is
  // the entire component; in regional mode it is one partition, and the caller
  // distinguishes them by prepending the partition id to shape_key.
  //   shape_key     : value of every dynamic symbol, in a codegen-fixed order.
  //   copy_in       : input indices staged into node-owned slots and refreshed
  //                   per replay. Whole-graph capture stages every tensor input,
  //                   since none of them are produced by an earlier capture.
  //   escape_outs   : output indices that escape the model (neutralize + keep).
  //   caller_stream : the stream run_impl is on; used only by the uncaptured
  //                   fallback path.
  //   body          : runs the component eagerly into `out` on the given stream.
  // Fills `outs` with non-owning views. See the OUTPUT CONTRACT at the top of
  // this file: they are valid only until the next run_graph() on this instance.
  void run_graph(
      const std::vector<int64_t>& shape_key,
      AtenTensorHandle* ins,
      int32_t num_ins,
      AtenTensorHandle* outs,
      int32_t num_outs,
      const std::vector<int32_t>& copy_in,
      const std::vector<int32_t>& escape_outs,
      void* caller_stream,
      const GraphBody& body) {
    // Shared (read) lock over capture/replay (see captureMutex in the shim):
    // replay + output reconstruction run concurrently with other instances'
    // replays but are excluded while ANY instance holds the exclusive capture
    // lock -- so the capture-unsafe query in reconstruct (getDeviceFromPtr ->
    // cudaPointerGetAttributes) never runs during a concurrent capture. RAII.
    struct ReplayLock {
      ReplayLock() {
        aoti_torch_cuda_graph_replay_lock();
      }
      ~ReplayLock() {
        aoti_torch_cuda_graph_replay_unlock();
      }
    };
    // Reconstruct non-owning views at the recorded output addresses. The body
    // runs only at capture time, so on replay (and right after record) we hand
    // the caller fresh from_blob handles over the recorded memory.
    auto reconstruct_outputs = [&](CUDAGraphNode* n) {
      for (int32_t i = 0; i < num_outs; i++) {
        outs[i] = cuda_graph_reconstruct(n->output_meta[i]);
      }
    };

    auto it = flat_nodes_.find(shape_key);
    if (it != flat_nodes_.end()) {
      CUDAGraphNode* node = it->second.get();
      ReplayLock rlk;
      replay_node(node, ins, copy_in);
      reconstruct_outputs(node);
      if (debug_) {
        fprintf(
            stderr,
            "[cudagraph] REPLAY shape=%s total_nodes=%zu\n",
            format_shape_key(shape_key).c_str(),
            flat_nodes_.size());
      }
      return;
    }

    if (flat_nodes_.size() >= max_captures_) {
      // Cap reached: run uncaptured rather than reserving more pool memory for
      // a model whose serving batch keeps producing new shapes.
      run_uncaptured(ins, num_ins, outs, num_outs, caller_stream, body);
      if (!warned_capture_cap_) {
        warned_capture_cap_ = true;
        fprintf(
            stderr,
            "[cudagraph] WARNING: cuda-graph capture cap (%zu) reached; shape %s "
            "and every further new shape will run UNCAPTURED. Raise "
            "aot_inductor.cudagraph_max_captures if these shapes are hot.\n",
            max_captures_,
            format_shape_key(shape_key).c_str());
        fflush(stderr);
      }
      return;
    }

    // record_node takes the EXCLUSIVE capture lock internally (no replay of any
    // instance runs during it); reconstruct afterwards under the shared lock.
    CUDAGraphNode* node = record_node(
        shape_key, ins, num_ins, num_outs, copy_in, escape_outs, body);
    {
      ReplayLock rlk;
      reconstruct_outputs(node);
    }
    if (debug_) {
      int64_t used_bytes = 0;
      aoti_torch_cuda_graph_device_used_bytes(device_index_, &used_bytes);
      fprintf(
          stderr,
          "[cudagraph] RECORD shape=%s total_nodes=%zu dev_used_mb=%ld\n",
          format_shape_key(shape_key).c_str(),
          flat_nodes_.size(),
          used_bytes / (1024 * 1024));
      fflush(stderr);
    }
  }

  // Drop every captured graph, so the next call at any shape re-records.
  //
  // A capture bakes the addresses it saw at record time, including those of the
  // constant buffer. A weight update that re-points the constants (notably
  // swap_constant_buffer, which flips to the other buffer) therefore invalidates
  // every capture: replays would keep reading the old buffer and silently serve
  // stale weights. The container calls this on those paths.
  //
  // The caller must exclude concurrent replays (hold model_exec_mutex_
  // exclusively); clearing the table under a live replay would free a graph
  // mid-launch.
  void reset_captures() {
    release_uncaptured_outputs();
    flat_nodes_.clear();
    warned_capture_cap_ = false;
  }

 private:
  static std::string format_shape_key(const std::vector<int64_t>& key) {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < key.size(); i++) {
      if (i != 0) {
        ss << ",";
      }
      ss << key[i];
    }
    ss << ")";
    return std::move(ss).str();
  }

  // The generated body TAKES OWNERSHIP of its input handles (it wraps each in a
  // RAII handle and frees it, or releases passthrough ones as outputs). Our
  // inputs must NOT be freed by it: captured static slots are node-owned and
  // needed for replay, and the wrapper's outer scope still owns the borrowed
  // caller handles. So hand the body fresh NON-OWNING views (from_blob, no-op
  // deleter) over each input's memory. Views point at the same addresses, so a
  // captured graph is identical. A fresh set per call, since each is consumed.
  static std::vector<AtenTensorHandle> make_input_views(
      const std::vector<AtenTensorHandle>& srcs) {
    std::vector<AtenTensorHandle> views(srcs.size(), nullptr);
    for (size_t i = 0; i < srcs.size(); i++) {
      views[i] = cuda_graph_reconstruct(cuda_graph_capture_meta(srcs[i]));
    }
    return views;
  }

  void release_uncaptured_outputs() {
    for (auto h : pending_uncaptured_outputs_) {
      if (h) {
        aoti_torch_delete_tensor_object(h);
      }
    }
    pending_uncaptured_outputs_.clear();
  }

  // Capture-cap fallback: run the body directly on the caller's stream. The
  // body allocates real owning outputs; we retain those and hand the caller
  // non-owning views, so this path has exactly the same output contract as a
  // replay. The retained handles are freed at the start of the next call.
  void run_uncaptured(
      AtenTensorHandle* ins,
      int32_t num_ins,
      AtenTensorHandle* outs,
      int32_t num_outs,
      void* caller_stream,
      const GraphBody& body) {
    std::vector<AtenTensorHandle> owned(num_outs, nullptr);
    {
      std::vector<AtenTensorHandle> views =
          make_input_views(std::vector<AtenTensorHandle>(ins, ins + num_ins));
      body(views.data(), owned.data(), caller_stream);
    }
    pending_uncaptured_outputs_.reserve(
        pending_uncaptured_outputs_.size() + static_cast<size_t>(num_outs));
    for (int32_t i = 0; i < num_outs; i++) {
      outs[i] = cuda_graph_reconstruct(cuda_graph_capture_meta(owned[i]));
      pending_uncaptured_outputs_.push_back(owned[i]);
    }
  }

  void replay_node(
      CUDAGraphNode* node,
      AtenTensorHandle* ins,
      const std::vector<int32_t>& copy_in) {
    for (int32_t i : copy_in) {
      aoti_torch_copy_(node->static_inputs[i], ins[i], /*non_blocking=*/0);
    }
    aoti_torch_cuda_graph_replay(node->graph);
  }

  CUDAGraphNode* record_node(
      const std::vector<int64_t>& shape_key,
      AtenTensorHandle* ins,
      int32_t num_ins,
      int32_t num_outs,
      const std::vector<int32_t>& copy_in,
      const std::vector<int32_t>& escape_outs,
      const GraphBody& body) {
    // Serialize capture across concurrent model instances (see captureMutex in
    // the shim): concurrent cuda-graph capture on a device is unsafe (illegal
    // memory access). REPLAY (the hot path) never takes this lock; only this
    // recording path does. RAII so it releases on any return/exception.
    struct CaptureLock {
      CaptureLock() {
        aoti_torch_cuda_graph_capture_lock();
      }
      ~CaptureLock() {
        aoti_torch_cuda_graph_capture_unlock();
      }
    } capture_lock_guard;

    // Order the shared capture stream after the caller's stream before we touch
    // the pool or capture. All partitions share one capture stream + one cuBLAS
    // workspace slot, so without this the previous partition's first-replay
    // (caller stream) and this one's warmup/capture (capture stream) would race
    // on that workspace. Recording-only; not on the hot path. No checkpoint
    // restore: the slab address is fixed, so there is nothing to rewind.
    aoti_torch_cuda_graph_sync_before_record(device_index_);

    auto child = std::make_unique<CUDAGraphNode>();
    aoti_torch_cuda_graph_create(&child->graph, pool_handle_);

    // Eager inputs get node-owned static slots refreshed per replay via copy_in;
    // chained inputs are read in place.
    std::vector<bool> is_eager(num_ins, false);
    for (int32_t i : copy_in) {
      is_eager[i] = true;
    }
    child->static_inputs.assign(num_ins, nullptr);
    std::vector<AtenTensorHandle> cap_in(num_ins);
    for (int32_t i = 0; i < num_ins; i++) {
      if (is_eager[i]) {
        // Guard the copy_in staging clone: if it fails (or returns a null
        // handle) the downstream make_input_views -> cuda_graph_capture_meta ->
        // aoti_torch_get_data_ptr would dereference a null handle and SIGSEGV
        // at 0x0. The usual trigger is an out-of-bounds input dimension (e.g. a
        // dynamic dim larger than its compiled max). Fail with an actionable
        // message naming the partition and shape instead.
        AOTITorchError clone_err =
            aoti_torch_clone_preserve_strides(ins[i], &child->static_inputs[i]);
        if (clone_err != AOTI_TORCH_SUCCESS ||
            child->static_inputs[i] == nullptr) {
          std::stringstream ss;
          ss << "AOTI cuda-graph: copy_in staging "
                "(aoti_torch_clone_preserve_strides) failed for input "
             << i << " at shape " << format_shape_key(shape_key)
             << " -- likely an out-of-bounds input dimension";
          throw std::runtime_error(std::move(ss).str());
        }
        cap_in[i] = child->static_inputs[i];
      } else {
        cap_in[i] = ins[i]; // chained: read the producer output in place
      }
    }

    void* cap_stream = nullptr;
    aoti_torch_cuda_graph_get_stream(child->graph, &cap_stream);

    // Warmup (lazy cuBLAS/Triton init) with throwaway input views + outputs.
    // make_input_views gives the body non-owning views so it never frees our
    // real inputs; a fresh set per call (warmup + capture) since each consumes
    // its own.
    {
      std::vector<AtenTensorHandle> warm_in = make_input_views(cap_in);
      std::vector<AtenTensorHandle> warm_out(num_outs, nullptr);
      body(warm_in.data(), warm_out.data(), cap_stream);
      for (auto h : warm_out) {
        if (h) {
          aoti_torch_delete_tensor_object(h);
        }
      }
    }

    // Capture. The body writes fresh output handles into child->owning_outputs.
    // Most outputs are slab-resident (their storage is the durable slab, so the
    // captured graph bakes stable slab addresses); the rest are extern / fallback
    // op outputs that cross a partition boundary and are real owning allocations
    // excluded from the slab. We must NOT blanket-delete these handles here: that
    // would free the extern allocations and dangle their recorded output_meta
    // data_ptr (UAF on replay). Instead we retain EVERY handle in owning_outputs
    // (freed at node teardown by ~CUDAGraphNode) and neutralize the
    // model-escaping ones below. At teardown a slab-view delete is a no-op (the
    // slab storage is owned by the model, refcount stays >= 1); an extern
    // boundary output's delete correctly frees it. Recorded addresses stay valid
    // for the node's whole life.
    aoti_torch_cuda_graph_begin_capture(child->graph);
    // cudaStreamEndCapture and endAllocateToPool are reached ONLY through
    // CUDAGraph::capture_end(). ~CUDAGraph calls reset(), which calls neither
    // and whose cleanup is gated on capture_ended_ -- upstream documents that
    // it does not try to recover from a failure mid-capture. So a throwing body
    // would leave this instance's capture stream stuck in capture state, the
    // allocator still routing allocations into the private pool, and the pool
    // itself never released: permanently broken, with no way back. End the
    // capture on every path out instead. aoti_torch_* reports status rather
    // than throwing, so this is safe while unwinding.
    struct CaptureScope {
      AOTICudaGraphHandle graph;
      bool active{true};
      void end() {
        if (active) {
          active = false;
          aoti_torch_cuda_graph_end_capture(graph);
        }
      }
      ~CaptureScope() {
        end();
      }
    } capture_scope{child->graph};
    child->owning_outputs.assign(num_outs, nullptr);
    {
      std::vector<AtenTensorHandle> cap_views = make_input_views(cap_in);
      body(cap_views.data(), child->owning_outputs.data(), cap_stream);
    }
    capture_scope.end();

    // Clear cuBLAS workspaces after capture (mirrors cudagraph_trees'
    // clear_cublas_manager exit). With the single shared capture stream there is
    // exactly one workspace, reused at a fixed slot across all captures. The
    // captured graph keeps using that still-reserved memory as transient scratch.
    aoti_torch_cuda_graph_clear_cublas_workspaces(pool_handle_);

    child->output_meta.resize(num_outs);
    for (int32_t i = 0; i < num_outs; i++) {
      child->output_meta[i] = cuda_graph_capture_meta(child->owning_outputs[i]);
    }
    // Escaping (model) outputs: neutralize the deleter so node teardown does not
    // free an allocation that the caller still owns. For a slab-view output this
    // is also harmless (its storage is already non-owning).
    for (int32_t i : escape_outs) {
      if (child->owning_outputs[i] != nullptr) {
        aoti_torch_storage_set_noop_deleter(child->owning_outputs[i]);
      }
    }

    CUDAGraphNode* node = child.get();
    flat_nodes_.emplace(shape_key, std::move(child));

    // Produce the correct first result with the real inputs (capture ran on the
    // empty static input slots).
    replay_node(node, ins, copy_in);
    return node;
  }

  int32_t device_index_;
  size_t max_captures_; // config.aot_inductor.cudagraph_max_captures
  void* pool_handle_{nullptr}; // this manager's private graph pool + capture stream
  bool debug_{false}; // AOTI_CUDAGRAPH_DEBUG: log RECORD/REPLAY per forward
  bool warned_capture_cap_{false}; // cap warning is emitted at most once
  // Outputs of the most recent UNCAPTURED run. Freed at the start of the next
  // call so this path honours the same output lifetime as a replay.
  std::vector<AtenTensorHandle> pending_uncaptured_outputs_;
  // The only dispatch table: dynamic-symbol tuple -> node. No tree.
  std::unordered_map<
      std::vector<int64_t>,
      std::unique_ptr<CUDAGraphNode>,
      AOTICUDAGraphShapeKeyHash>
      flat_nodes_;
};

} // namespace torch::aot_inductor
