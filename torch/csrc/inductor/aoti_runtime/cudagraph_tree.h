// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

// AOTI regional cuda-graph runtime (flat, no-tree, no-checkpoint).
//
//   * One private graph mempool is shared by every capture, so different dynamic
//     shapes reuse the same memory.
//   * Dispatch is a single flat map keyed by encode(partition_id, shape). There
//     is no recording tree: no parent/child walk, no per-node allocator
//     checkpoint, no checkpoint snapshot/restore, no divergence handling.
//   * Most of a partition's outputs live in the model's memory-planning slab (a
//     durable, address-stable allocation owned by the generated model instance,
//     NOT by this manager's private pool). For a slab-resident output the body's
//     fresh handle is a non-owning view: the slab keeps the memory alive across
//     every replay. The exception is an output produced by an extern / fallback
//     op (cublas/cudnn/aten) that crosses a partition boundary: the scheduler
//     excludes those from the slab, so they are real owning allocations. To keep
//     both cases correct, every output handle is held in owning_outputs until the
//     node is destroyed (a slab-view delete is a harmless no-op there because the
//     slab storage keeps refcount >= 1; an extern boundary output is the sole
//     owner and is freed correctly). Outputs that escape the model (escape_outs)
//     get their deleter neutralized so the caller can keep owning them.
//   * Replay is one-directional and touches no allocator state -- the hot serving
//     path is a pure cudaGraphLaunch. The private pool keeps only capture-stream
//     ordering (sync_before_record) and the single shared cuBLAS workspace slot
//     (clear_cublas_workspaces); the slab address is fixed, so there is no
//     allocator bookkeeping to snapshot or rewind.
//
// Header-only over the stable AOTI C ABI, so the generated model .so includes it
// and hands the partition body in as a lambda (capturing the wrapper's locals).

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

// One captured graph for a single (partition_id, dynamic-shape).
struct CUDAGraphNode {
  AOTICudaGraphHandle graph{nullptr};

  // Eager input slots (external, fixed-address; copied into per replay). nullptr
  // for chained inputs (read a producer output address directly).
  std::vector<AtenTensorHandle> static_inputs;

  // Per output: address+metadata for reconstruct-on-replay.
  std::vector<CUDAGraphOutputMeta> output_meta;
  // Every output handle is held until teardown. Slab-resident outputs are
  // non-owning views (delete is a no-op); extern boundary outputs are the sole
  // owner and are freed here; escaping (model) outputs are neutralized and kept.
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

// Owns the shared pool and a flat table of recordings. Mirrors
// cudagraph_trees.CUDAGraphTreeManager (single-pool, inference-only: no
// generation tracking, no runtime liveness -- liveness is a compile-time
// decision passed in via copy_in/escape_outs).
class AOTICUDAGraphTreeManager {
 public:
  using PartitionBody = std::function<
      void(AtenTensorHandle* in, AtenTensorHandle* out, void* stream)>;

  explicit AOTICUDAGraphTreeManager(int32_t device_index)
      : device_index_(device_index) {
    // On only when AOTI_CGTREE_DEBUG is set to a truthy value. An unset, empty,
    // or "0" value is off so production runs that export AOTI_CGTREE_DEBUG=0
    // (e.g. the scorecard harness) do not get the per-replay RECORD/REPLAY flood.
    const char* debug_env = std::getenv("AOTI_CGTREE_DEBUG");
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
  AOTICUDAGraphTreeManager(const AOTICUDAGraphTreeManager&) = delete;
  AOTICUDAGraphTreeManager& operator=(const AOTICUDAGraphTreeManager&) = delete;

  ~AOTICUDAGraphTreeManager() {
    // Destroy nodes (their captured graphs) BEFORE the pool: each ~CUDAGraph
    // releases the pool ref it took at capture; pool_destroy_handle then drops
    // the remaining ref (from ensure_created) -> use_count 0 -> freed.
    flat_nodes_.clear();
    aoti_torch_cuda_graph_pool_destroy_handle(pool_handle_);
  }

  // Run one captured partition.
  //   copy_in     : EAGER input indices (cloned into static slots; copied per replay).
  //                 All other inputs are chained -- read in place, never copied.
  //   escape_outs : output indices that escape the model (neutralize + keep).
  //   body        : runs the partition eagerly into `out` on the given stream.
  // Fills `outs` with non-owning handles to this node's output addresses.
  void run_partition(
      int32_t partition_id,
      int64_t shape_key,
      AtenTensorHandle* ins,
      int32_t num_ins,
      AtenTensorHandle* outs,
      int32_t num_outs,
      const std::vector<int32_t>& copy_in,
      const std::vector<int32_t>& escape_outs,
      const PartitionBody& body) {
    const int64_t key = encode(partition_id, shape_key);
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
    // Reconstruct non-owning views at the recorded slab addresses. The body runs
    // only at capture time, so on replay (and right after record) we hand the
    // caller fresh from_blob handles over the durable slab memory.
    auto reconstruct_outputs = [&](CUDAGraphNode* n) {
      for (int32_t i = 0; i < num_outs; i++) {
        outs[i] = cuda_graph_reconstruct(n->output_meta[i]);
      }
    };

    auto it = flat_nodes_.find(key);
    CUDAGraphNode* node = nullptr;
    if (it != flat_nodes_.end()) {
      node = it->second.get();
      ReplayLock rlk;
      replay_node(node, ins, copy_in);
      reconstruct_outputs(node);
      // Log once per forward (partition 0) to avoid per-partition spam.
      if (debug_ && partition_id == 0) {
        fprintf(
            stderr,
            "[cgtree] REPLAY forward start pid=0 shape=%ld total_nodes=%zu\n",
            shape_key,
            recorded_nodes_);
      }
    } else {
      // record_node takes the EXCLUSIVE capture lock internally (no replay of any
      // instance runs during it); reconstruct afterwards under the shared lock.
      node = record_node(
          partition_id,
          shape_key,
          key,
          ins,
          num_ins,
          num_outs,
          copy_in,
          escape_outs,
          body);
      ++recorded_nodes_;
      {
        ReplayLock rlk;
        reconstruct_outputs(node);
      }
      if (debug_) {
        int64_t used_bytes = 0;
        aoti_torch_cuda_graph_device_used_bytes(device_index_, &used_bytes);
        fprintf(
            stderr,
            "[cgtree] RECORD pid=%d shape=%ld total_nodes=%zu dev_used_mb=%ld\n",
            partition_id,
            shape_key,
            recorded_nodes_,
            used_bytes / (1024 * 1024));
        fflush(stderr);
      }
    }
  }

 private:
  static int64_t encode(int32_t partition_id, int64_t shape_key) {
    return (static_cast<int64_t>(partition_id) << 40) ^
        (shape_key & 0xFFFFFFFFFFLL);
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
      int32_t partition_id,
      int64_t shape_key,
      int64_t key,
      AtenTensorHandle* ins,
      int32_t num_ins,
      int32_t num_outs,
      const std::vector<int32_t>& copy_in,
      const std::vector<int32_t>& escape_outs,
      const PartitionBody& body) {
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
                "(aoti_torch_clone_preserve_strides) failed for partition "
             << partition_id << " input " << i << " at shape_key " << shape_key
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

    // The generated partition body TAKES OWNERSHIP of its input handles (wraps
    // each in a RAII handle and frees it, or releases passthrough ones as
    // outputs). Our inputs must NOT be freed by it: eager inputs are node-owned
    // static buffers needed for replay, chained inputs are producer-owned, and
    // the wrapper's outer scope still owns and reuses the borrowed handles. So
    // hand the body fresh NON-OWNING views (from_blob, no-op deleter) over each
    // input's memory -- it can take, free, or pass them through without ever
    // touching the real input storage. Views point at the same addresses, so the
    // captured graph is identical. Fresh set per call (warmup + capture) since
    // each consumes its own.
    auto make_input_views = [&]() {
      std::vector<AtenTensorHandle> views(num_ins, nullptr);
      for (int32_t i = 0; i < num_ins; i++) {
        views[i] = cuda_graph_reconstruct(cuda_graph_capture_meta(cap_in[i]));
      }
      return views;
    };

    // Warmup (lazy cuBLAS/Triton init) with throwaway input views + outputs.
    {
      std::vector<AtenTensorHandle> warm_in = make_input_views();
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
    child->owning_outputs.assign(num_outs, nullptr);
    {
      std::vector<AtenTensorHandle> cap_views = make_input_views();
      body(cap_views.data(), child->owning_outputs.data(), cap_stream);
    }
    aoti_torch_cuda_graph_end_capture(child->graph);

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
      aoti_torch_storage_set_noop_deleter(child->owning_outputs[i]);
    }

    CUDAGraphNode* node = child.get();
    flat_nodes_.emplace(key, std::move(child));

    // Produce the correct first result with the real inputs (capture ran on the
    // empty static input slots).
    replay_node(node, ins, copy_in);
    return node;
  }

  int32_t device_index_;
  void* pool_handle_{nullptr}; // this manager's private graph pool + capture stream
  size_t recorded_nodes_{0}; // total captured (partition, shape) nodes
  bool debug_{false}; // AOTI_CGTREE_DEBUG: log RECORD/REPLAY per partition
  // The only dispatch table: a flat encode(pid, shape) -> node map, no tree.
  std::unordered_map<int64_t, std::unique_ptr<CUDAGraphNode>> flat_nodes_;
};

} // namespace torch::aot_inductor
