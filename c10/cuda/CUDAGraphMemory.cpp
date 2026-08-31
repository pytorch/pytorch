#include <c10/cuda/impl/CUDAGraphMemory.h>

#include <c10/util/Exception.h>

namespace c10::cuda::CUDAGraphMemory {

void CaptureTracker::captureBegin() {
  ++active_captures_;
}

void CaptureTracker::captureBegin(const CaptureRegistration& registration) {
  TORCH_INTERNAL_ASSERT(registration.capture_id != 0);
  TORCH_INTERNAL_ASSERT(
      registration.parent_capture_id.has_value() ==
      registration.parent_dependency_stream.has_value());
  TORCH_INTERNAL_ASSERT(!capture_tree_.count(registration.capture_id));

  CaptureId_t root_capture_id = registration.capture_id;
  cudaStream_t block_reuse_stream = registration.primary_stream;
  if (registration.parent_capture_id.has_value()) {
    auto parent_it = capture_tree_.find(*registration.parent_capture_id);
    TORCH_INTERNAL_ASSERT(
        parent_it != capture_tree_.end() && parent_it->second.is_active,
        "Conditional capture parent is not active");
    TORCH_INTERNAL_ASSERT(
        parent_it->second.mempool_id == registration.mempool_id,
        "Conditional capture must share its parent's memory pool");
    const CaptureTreeNode& parent = parent_it->second;
    root_capture_id = parent.root_capture_id;
    block_reuse_stream = *registration.parent_dependency_stream;
    if (block_reuse_stream == parent.primary_stream) {
      block_reuse_stream = parent.block_reuse_stream;
    }
  }

  capture_tree_.emplace(
      registration.capture_id,
      CaptureTreeNode{
          registration.mempool_id,
          registration.primary_stream,
          block_reuse_stream,
          registration.parent_capture_id,
          root_capture_id,
          true});
  captureBegin();
}

void CaptureTracker::captureEnd() {
  TORCH_INTERNAL_ASSERT(
      active_captures_ > 0, "captureEnd called with no active capture");
  --active_captures_;
}

size_t CaptureTracker::captureEnd(CaptureId_t capture_id) {
  auto capture_it = capture_tree_.find(capture_id);
  TORCH_INTERNAL_ASSERT(
      capture_it != capture_tree_.end() && capture_it->second.is_active,
      "Capture is not registered or has already ended");
  const CaptureTreeNode ended_capture = capture_it->second;

  if (ended_capture.parent_capture_id.has_value()) {
    capture_it->second.is_active = false;
    auto parent_it = capture_tree_.find(*ended_capture.parent_capture_id);
    TORCH_INTERNAL_ASSERT(
        parent_it != capture_tree_.end() && parent_it->second.is_active,
        "Conditional capture parent is not active");
    parent_it->second.invalid_capture_free_count +=
        ended_capture.invalid_capture_free_count;
    captureEnd();
  } else {
    const int erased_active_captures =
        eraseCaptureTree(ended_capture.root_capture_id);
    TORCH_INTERNAL_ASSERT(active_captures_ >= erased_active_captures);
    active_captures_ -= erased_active_captures;
  }
  return ended_capture.invalid_capture_free_count;
}

AllocationContext CaptureTracker::allocationContext(
    cudaStream_t request_stream) const {
  const auto info = c10::cuda::captureInfoMayInitCtx(request_stream);
  AllocationContext context{
      info.status != CaptureStatus::None, std::nullopt, request_stream};
  if (info.status != CaptureStatus::Active) {
    return context;
  }

  auto capture_it = capture_tree_.find(info.id);
  if (capture_it == capture_tree_.end()) {
    return context;
  }
  TORCH_INTERNAL_ASSERT(
      capture_it->second.is_active,
      "Active CUDA capture has inactive graph-memory state");
  context.tracked_capture_id = info.id;
  if (request_stream == capture_it->second.primary_stream) {
    context.block_reuse_stream = capture_it->second.block_reuse_stream;
  }
  return context;
}

void CaptureTracker::recordAllocation(
    const void* block,
    const AllocationContext& context) {
  if (!context.tracked_capture_id.has_value()) {
    return;
  }
  block_allocation_captures_.insert_or_assign(
      block, *context.tracked_capture_id);
}

void CaptureTracker::recordFree(
    const void* block,
    cudaStream_t free_stream) {
  TORCH_INTERNAL_ASSERT(hasActiveCaptures());
  auto allocation_it = block_allocation_captures_.find(block);
  if (allocation_it == block_allocation_captures_.end()) {
    return;
  }

  const AllocationContext free_context = allocationContext(free_stream);
  if (free_context.tracked_capture_id.has_value() &&
      !isFreeInAllocationCaptureOrAncestor(
          allocation_it->second, *free_context.tracked_capture_id)) {
    ++capture_tree_.at(*free_context.tracked_capture_id)
          .invalid_capture_free_count;
  }
  block_allocation_captures_.erase(allocation_it);
}

bool CaptureTracker::isFreeInAllocationCaptureOrAncestor(
    CaptureId_t allocation_capture_id,
    CaptureId_t free_capture_id) const {
  while (true) {
    if (allocation_capture_id == free_capture_id) {
      return true;
    }
    auto allocation_capture_it = capture_tree_.find(allocation_capture_id);
    if (allocation_capture_it == capture_tree_.end() ||
        !allocation_capture_it->second.parent_capture_id.has_value()) {
      return false;
    }
    allocation_capture_id = *allocation_capture_it->second.parent_capture_id;
  }
}

int CaptureTracker::eraseCaptureTree(CaptureId_t root_capture_id) {
  int erased_active_captures = 0;
  ska::flat_hash_set<CaptureId_t> capture_ids;
  for (auto it = capture_tree_.begin(); it != capture_tree_.end();) {
    if (it->second.root_capture_id == root_capture_id) {
      erased_active_captures += it->second.is_active ? 1 : 0;
      capture_ids.insert(it->first);
      it = capture_tree_.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = block_allocation_captures_.begin();
       it != block_allocation_captures_.end();) {
    if (capture_ids.count(it->second)) {
      it = block_allocation_captures_.erase(it);
    } else {
      ++it;
    }
  }
  return erased_active_captures;
}

} // namespace c10::cuda::CUDAGraphMemory
