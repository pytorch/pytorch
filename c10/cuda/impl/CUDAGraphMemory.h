#pragma once

#include <c10/cuda/CUDAGraphMemory.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>

#include <optional>
#include <utility>
#include <vector>

namespace c10::cuda::CUDAGraphMemory {

struct AllocationContext {
  bool is_capturing{false};
  std::optional<CaptureId_t> tracked_capture_id;
  cudaStream_t request_stream;
  cudaStream_t block_reuse_stream;
};

// Tracks conditional capture relationships and the capture that allocated each
// block. Block lifecycle and CUDA-DAG policy are added in later changes.
class CaptureTracker {
 public:
  struct AllocationRecord {
    CaptureId_t capture_id;
    cudaStream_t request_stream;
  };

  void captureBegin();
  void captureBegin(const CaptureRegistration& registration);
  void captureEnd();
  size_t captureEnd(CaptureId_t capture_id);
  bool hasActiveCaptures() const {
    return active_captures_ != 0;
  }
  template <typename Predicate>
  bool captureIsActive(CaptureId_t capture_id, Predicate&& predicate) const {
    auto capture_it = capture_tree_.find(capture_id);
    return capture_it != capture_tree_.end() && capture_it->second.is_active &&
        predicate(capture_it->second.primary_stream);
  }
  AllocationContext allocationContext(cudaStream_t request_stream) const {
    if (C10_LIKELY(active_captures_ == 0)) {
      return {false, std::nullopt, request_stream, request_stream};
    }
    return allocationContextSlow(request_stream);
  }
  void recordAllocation(const void* block, const AllocationContext& context);
  std::optional<AllocationRecord> recordFree(
      const void* block,
      std::optional<CaptureId_t> free_capture_id);

 private:
  struct CaptureTreeNode {
    MempoolId_t mempool_id;
    cudaStream_t primary_stream;
    cudaStream_t block_reuse_stream;
    std::optional<CaptureId_t> parent_capture_id;
    CaptureId_t root_capture_id;
    bool is_active;
    size_t invalid_capture_free_count{0};
  };

  // Walk from the allocation capture to the root. Allow only the same capture
  // or an ancestor capture to free the block.
  bool isFreeInAllocationCaptureOrAncestor(
      CaptureId_t allocation_capture_id,
      CaptureId_t free_capture_id) const;
  AllocationContext allocationContextSlow(cudaStream_t request_stream) const;
  int eraseCaptureTree(CaptureId_t root_capture_id);

  int active_captures_{0};
  ska::flat_hash_map<CaptureId_t, CaptureTreeNode> capture_tree_;
  ska::flat_hash_map<const void*, AllocationRecord> block_allocation_captures_;
};

enum class DeferredFreePolicy {
  DEFER_UNTIL_NO_ACTIVE_CAPTURE,
  REUSE_WHEN_TOPOLOGICALLY_SAFE,
};

struct CaptureDAGInfo {
  cudaGraph_t graph{};
  CaptureId_t capture_id{0};
  const cudaGraphNode_t* terminals{nullptr};
  size_t num_terminals{0};
  cudaStreamCaptureStatus status{cudaStreamCaptureStatusNone};
};

class CaptureDAGQuery {
 public:
  virtual ~CaptureDAGQuery() = default;

  virtual CaptureDAGInfo captureInfo(cudaStream_t stream) const = 0;
  virtual std::vector<cudaGraphNode_t> dependencies(
      cudaGraphNode_t node) const = 0;
};

class CaptureDAG {
 public:
  class FreeMarkerState {
   public:
    FreeMarkerState() = default;

   private:
    bool valid_{true};
    CaptureId_t capture_id_{0};
    cudaGraph_t graph_{nullptr};
    ska::flat_hash_set<cudaGraphNode_t> markers_;

    friend class CaptureDAG;
  };

  class TraversalState {
   public:
    TraversalState() = default;

   private:
    bool initialized_{false};
    CaptureId_t capture_id_{0};
    cudaGraph_t graph_{nullptr};
    ska::flat_hash_set<cudaGraphNode_t> visited_;

    friend class CaptureDAG;
  };

  CaptureDAG();
  explicit CaptureDAG(const CaptureDAGQuery& query);
  CaptureDAG(CaptureDAGQuery&&) = delete;
  CaptureDAG(const CaptureDAGQuery&&) = delete;

  CaptureDAGInfo captureInfo(cudaStream_t stream) const;
  bool recordFreeMarkersForStream(cudaStream_t stream, FreeMarkerState& state)
      const;
  std::vector<cudaGraphNode_t> takeFreeMarkers(FreeMarkerState&& state) const;
  void updateVisited(const CaptureDAGInfo& info, TraversalState& state) const;
  bool areMarkersReachable(
      ArrayRef<cudaGraphNode_t> markers,
      const TraversalState& state) const;

 private:
  const CaptureDAGQuery& query_;
};

// Owns the CUDA Graph-specific lifetime state for native allocator blocks.
// Block is deliberately a template parameter so the allocator can keep its
// type private. While a free is deferred, BlockManager has lifecycle custody of
// Block and may update its stream_uses. It invokes event insertion and final
// freeing through a short-lived allocator capability supplied by the caller;
// the allocator retains storage ownership and implements those mechanisms.
//
// The native caching allocator calls every method while holding its allocator
// mutex. BlockManager has no lock of its own. A registered Block* must not be
// merged away, deleted, or repurposed while an allocator callback is in
// progress.
// A throwing free callback must leave its Block valid. Event insertion may
// consume stream_uses before throwing; BlockManager restores those obligations
// so a later attempt cannot incorrectly classify the Block as immediately
// freeable.
template <typename Block>
class BlockManager {
 public:
  BlockManager() = default;
  explicit BlockManager(const CaptureDAGQuery& query) : capture_dag_(query) {}
  BlockManager(CaptureDAGQuery&&) = delete;
  BlockManager(const CaptureDAGQuery&&) = delete;

  void captureBegin() {
    capture_tracker_.captureBegin();
  }

  void captureBegin(const CaptureRegistration& registration) {
    capture_tracker_.captureBegin(registration);
  }

  void captureEnd() {
    capture_tracker_.captureEnd();
  }

  size_t captureEnd(CaptureId_t capture_id) {
    return capture_tracker_.captureEnd(capture_id);
  }

  bool hasActiveCaptures() const {
    return capture_tracker_.hasActiveCaptures();
  }

  AllocationContext allocationContext(cudaStream_t request_stream) const {
    return capture_tracker_.allocationContext(request_stream);
  }

  void recordAllocation(Block* block, const AllocationContext& context) {
    capture_tracker_.recordAllocation(block, context);
  }

  void recordFree(Block* block, cudaStream_t free_stream) {
    const auto info = capture_dag_.captureInfo(free_stream);
    const std::optional<CaptureId_t> free_capture_id =
        info.status == cudaStreamCaptureStatusActive
        ? std::make_optional(info.capture_id)
        : std::nullopt;
    auto allocation = capture_tracker_.recordFree(block, free_capture_id);
    if (allocation.has_value()) {
      block_state_[block].allocation = *allocation;
    }
  }

  bool isCaptureContext(cudaStream_t stream) const {
    TORCH_INTERNAL_ASSERT(hasActiveCaptures());
    return capture_dag_.captureInfo(stream).status !=
        cudaStreamCaptureStatusNone;
  }

  void recordStreamUse(
      Block* block,
      c10::cuda::CUDAStream stream,
      std::optional<CaptureId_t> capture_id) {
    auto& capture_stream_uses = block_state_[block].capture_stream_uses;
    for (const auto& use : capture_stream_uses) {
      if (use.stream == stream && use.capture_id == capture_id) {
        return;
      }
    }
    capture_stream_uses.push_back({stream, capture_id});
  }

  void deferFree(Block* block, DeferredFreePolicy policy) {
    auto& state = block_state_[block];
    TORCH_INTERNAL_ASSERT(
        !state.deferred.has_value(), "Block free was already deferred");

    std::vector<cudaGraphNode_t> markers;
    const cudaStream_t request_stream = state.allocation.has_value()
        ? state.allocation->request_stream
        : block->stream;
    const std::optional<CaptureId_t> allocation_capture_id =
        state.allocation.has_value()
        ? std::make_optional(state.allocation->capture_id)
        : std::nullopt;
    if (policy == DeferredFreePolicy::REUSE_WHEN_TOPOLOGICALLY_SAFE) {
      CaptureDAG::FreeMarkerState marker_state;
      bool valid =
          capture_dag_.recordFreeMarkersForStream(request_stream, marker_state);
      if (valid) {
        for (const auto& stream : block->stream_uses) {
          if (!capture_dag_.recordFreeMarkersForStream(
                  stream.stream(), marker_state)) {
            valid = false;
            break;
          }
        }
      }
      if (valid) {
        markers = capture_dag_.takeFreeMarkers(std::move(marker_state));
      }
    }

    state.deferred.emplace(DeferredBlock{
        .pool_id = block->pool->owner_MempoolId(),
        .request_stream = request_stream,
        .allocation_capture_id = allocation_capture_id,
        .free_markers = std::move(markers)});
    const bool inserted = deferred_blocks_.insert(block).second;
    TORCH_INTERNAL_ASSERT(inserted);
  }

  // Clears stream uses for blocks whose recorded free points precede the
  // current frontier, then invokes the allocator to return them to its free
  // pool. Graph state remains registered until each callback succeeds.
  template <typename AllocatorOps>
  void reclaimBlocks(
      cudaStream_t stream,
      DeferredFreePolicy policy,
      AllocatorOps&& ops) {
    if (policy == DeferredFreePolicy::DEFER_UNTIL_NO_ACTIVE_CAPTURE) {
      return;
    }

    auto info = capture_dag_.captureInfo(stream);
    if (info.status != cudaStreamCaptureStatusActive ||
        info.num_terminals == 0) {
      return;
    }
    auto capture_it = capture_state_.find(info.capture_id);
    if (capture_it == capture_state_.end()) {
      // Resolve allocator routing only after establishing that this stream is
      // capturing, and only once per capture. Do not publish capture state
      // until pool resolution succeeds.
      const auto pool_id = ops.capturePoolForStream(stream);
      auto [new_capture_it, inserted] =
          capture_state_.emplace(info.capture_id, CaptureState{pool_id, {}});
      TORCH_INTERNAL_ASSERT(inserted);
      capture_it = new_capture_it;
    }
    auto& traversal_state = capture_it->second.traversal_state[stream];
    capture_dag_.updateVisited(info, traversal_state);

    PendingBlocks pending;
    for (auto* block : deferred_blocks_) {
      auto state_it = block_state_.find(block);
      TORCH_INTERNAL_ASSERT(state_it != block_state_.end());
      TORCH_INTERNAL_ASSERT(state_it->second.deferred.has_value());
      const auto& deferred = *state_it->second.deferred;
      if (deferred.free_markers.empty() || deferred.request_stream != stream ||
          (deferred.allocation_capture_id.has_value() &&
           *deferred.allocation_capture_id != info.capture_id)) {
        continue;
      }
      if (capture_dag_.areMarkersReachable(
              deferred.free_markers, traversal_state)) {
        block->stream_uses.clear();
        pending.blocks_to_free.push_back(block);
      }
    }
    executeDeferredBlocks(std::move(pending), ops);
  }

  // Ends the graph-memory state associated with a pool-routing scope. This is
  // intentionally separate from captureEnd(): existing allocator integration
  // ends CUDA capture before it closes the matching routing scope.
  template <typename AllocatorOps>
  void endCapturePool(
      MempoolId_t pool_id,
      DeferredFreePolicy policy,
      AllocatorOps&& ops) {
    for (auto* block : deferred_blocks_) {
      const auto& deferred = *block_state_.at(block).deferred;
      if (deferred.pool_id == pool_id &&
          deferred.allocation_capture_id.has_value() &&
          capture_tracker_.captureIsActive(
              *deferred.allocation_capture_id, [this](cudaStream_t stream) {
                return capture_dag_.captureInfo(stream).status !=
                    cudaStreamCaptureStatusNone;
              })) {
        // A user MemPool scope can end while the graph capture that owns its
        // deferred blocks remains active. Keep those blocks until that capture
        // reaches a real end point.
        return;
      }
    }
    for (auto capture_it = capture_state_.begin();
         capture_it != capture_state_.end();) {
      if (capture_it->second.pool_id != pool_id) {
        ++capture_it;
        continue;
      }
      for (const auto& [stream, _] : capture_it->second.traversal_state) {
        TORCH_INTERNAL_ASSERT(
            capture_dag_.captureInfo(stream).status ==
                cudaStreamCaptureStatusNone,
            "This stream should not be capturing when the capture is ended");
      }
      capture_it = capture_state_.erase(capture_it);
    }

    if (policy == DeferredFreePolicy::DEFER_UNTIL_NO_ACTIVE_CAPTURE) {
      return;
    }
    PendingBlocks pending;
    for (auto* block : deferred_blocks_) {
      auto state_it = block_state_.find(block);
      TORCH_INTERNAL_ASSERT(state_it != block_state_.end());
      TORCH_INTERNAL_ASSERT(state_it->second.deferred.has_value());
      if (state_it->second.deferred->pool_id == pool_id) {
        classifyCaptureEndBlock(block, state_it->second, false, pending);
      }
    }
    executeDeferredBlocks(std::move(pending), ops);
  }

  // Drains frees that could not be associated with a successfully ended graph
  // pool. This is called only after the allocator has established that no CUDA
  // capture is active, immediately before it records ordinary CUDA events.
  template <typename AllocatorOps>
  void drainDeferredBlocks(AllocatorOps&& ops) {
    if (hasActiveCaptures()) {
      return;
    }

    PendingBlocks pending;
    for (auto* block : deferred_blocks_) {
      auto state_it = block_state_.find(block);
      TORCH_INTERNAL_ASSERT(state_it != block_state_.end());
      TORCH_INTERNAL_ASSERT(state_it->second.deferred.has_value());
      classifyCaptureEndBlock(block, state_it->second, true, pending);
    }
    executeDeferredBlocks(std::move(pending), ops);
  }

  void retire(Block* block) {
    if (block_state_.empty()) {
      return;
    }
    auto state_it = block_state_.find(block);
    if (state_it == block_state_.end()) {
      return;
    }
    deferred_blocks_.erase(block);
    block_state_.erase(state_it);
  }

  bool contains(Block* block) const {
    if (block_state_.empty()) {
      return false;
    }
    return block_state_.find(block) != block_state_.end();
  }

  bool isDeferred(Block* block) const {
    if (deferred_blocks_.empty()) {
      return false;
    }
    return deferred_blocks_.find(block) != deferred_blocks_.end();
  }

 private:
  struct PendingBlocks {
    std::vector<Block*> blocks_to_free;
    std::vector<Block*> blocks_to_record_events;
  };

  struct DeferredBlock {
    MempoolId_t pool_id;
    cudaStream_t request_stream;
    std::optional<CaptureId_t> allocation_capture_id;
    std::vector<cudaGraphNode_t> free_markers;
  };

  struct CapturedStreamUse {
    c10::cuda::CUDAStream stream;
    std::optional<CaptureId_t> capture_id;
  };

  struct BlockState {
    std::vector<CapturedStreamUse> capture_stream_uses;
    std::optional<CaptureTracker::AllocationRecord> allocation;
    std::optional<DeferredBlock> deferred;
  };

  struct CaptureState {
    MempoolId_t pool_id;
    ska::flat_hash_map<cudaStream_t, CaptureDAG::TraversalState>
        traversal_state;
  };

  void classifyCaptureEndBlock(
      Block* block,
      const BlockState& state,
      bool all_captures_ended,
      PendingBlocks& pending) {
    if (!all_captures_ended) {
      const auto& allocation_capture_id = state.deferred->allocation_capture_id;
      for (const auto& use : state.capture_stream_uses) {
        if (!allocation_capture_id.has_value() || !use.capture_id.has_value() ||
            *use.capture_id != *allocation_capture_id) {
          // A use owned by another (or unknown) capture cannot be retired when
          // this block's allocation pool ends. Keep the block deferred until
          // no registered capture remains.
          return;
        }
      }
    }
    for (auto it = block->stream_uses.begin();
         it != block->stream_uses.end();) {
      bool known_capture_use = false;
      bool unknown_capture_use = false;
      for (const auto& use : state.capture_stream_uses) {
        if (use.stream == *it) {
          if (use.capture_id.has_value()) {
            known_capture_use = true;
          } else {
            unknown_capture_use = true;
          }
        }
      }
      // Unknown provenance may be an ordinary stream use observed while a
      // different stream was capturing. Preserve it for event insertion.
      if (known_capture_use && !unknown_capture_use) {
        it = block->stream_uses.erase(it);
      } else {
        ++it;
      }
    }
    if (block->stream_uses.empty()) {
      pending.blocks_to_free.push_back(block);
    } else {
      pending.blocks_to_record_events.push_back(block);
    }
  }

  // Allocator callbacks run only after the scan completes, so they may retire
  // entries without invalidating a deferred_blocks_ iterator. Custody remains
  // registered through each callback and is released only after it succeeds.
  template <typename AllocatorOps>
  void executeDeferredBlocks(PendingBlocks pending, AllocatorOps& ops) {
    if (pending.blocks_to_free.empty() &&
        pending.blocks_to_record_events.empty()) {
      return;
    }
    for (auto* block : pending.blocks_to_free) {
      TORCH_INTERNAL_ASSERT(block->stream_uses.empty());
      ops.freeBlock(block);
      completeDeferred(block);
    }
    for (auto* block : pending.blocks_to_record_events) {
      TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
      auto stream_uses = block->stream_uses;
      try {
        ops.insertEvents(block);
      } catch (...) {
        // insert_events consumes stream_uses before recording events. Restore
        // all original obligations after a partial failure. Events already
        // recorded by the failed attempt remain valid; a retry may safely
        // record conservative duplicates.
        block->stream_uses = std::move(stream_uses);
        throw;
      }
      completeDeferred(block);
    }
  }

  void completeDeferred(Block* block) {
    TORCH_INTERNAL_ASSERT(deferred_blocks_.erase(block) == 1);
    TORCH_INTERNAL_ASSERT(block_state_.erase(block) == 1);
  }

  CaptureTracker capture_tracker_;
  ska::flat_hash_map<Block*, BlockState> block_state_;
  // Keep deferred scans proportional to outstanding deferred frees rather than
  // all live blocks that recorded a stream during capture.
  ska::flat_hash_set<Block*> deferred_blocks_;
  ska::flat_hash_map<CaptureId_t, CaptureState> capture_state_;
  CaptureDAG capture_dag_;
};

} // namespace c10::cuda::CUDAGraphMemory
