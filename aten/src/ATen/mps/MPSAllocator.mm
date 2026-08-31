//  Copyright © 2022 Apple Inc.

#include <ATen/CPUFunctions.h>
#include <ATen/EmptyTensor.h>
#include <ATen/mps/MPSAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/util/Logging.h>
#include <c10/util/env.h>

#include <algorithm>
#include <atomic>
#include <unordered_map>

namespace at::mps {

C10_DEFINE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback)

namespace HeapAllocator {

uint64_t BufferBlock::buffer_counter = 0;
uint64_t HeapBlock::heap_counter = 0;

// Set once the heap allocator singleton has been constructed (i.e. MPS/Metal is
// actually in use). Lets the registered c10 allocator report readiness via
// DeviceAllocator::initialized() without forcing Metal initialization.
static std::atomic<bool> s_mps_allocator_initialized{false};

void MPSHeapAllocatorImpl::init_allocator() {
  TORCH_CHECK(m_device.hasUnifiedMemory, "MPS backend is only supported on devices with unified memory");
  init_buffer_pools();

  // debug verbosity flags (see DebugVerbosity enum)
  static const auto verbosity_str = c10::utils::get_env("PYTORCH_DEBUG_MPS_ALLOCATOR");
  m_debug_verbosity = verbosity_str ? strtol(verbosity_str->c_str(), nullptr, 0) : DebugVerbosity::SILENT;

  static const auto high_watermark_ratio_str = c10::utils::get_env("PYTORCH_MPS_HIGH_WATERMARK_RATIO");
  const double high_watermark_ratio =
      high_watermark_ratio_str ? strtod(high_watermark_ratio_str->c_str(), nullptr) : default_high_watermark_ratio;
  setHighWatermarkRatio(high_watermark_ratio);

  static const auto low_watermark_ratio_str = c10::utils::get_env("PYTORCH_MPS_LOW_WATERMARK_RATIO");
  const double low_watermark_ratio =
      low_watermark_ratio_str ? strtod(low_watermark_ratio_str->c_str(), nullptr) : default_low_watermark_ratio;
  setLowWatermarkRatio(low_watermark_ratio);

  if (m_debug_verbosity & DebugVerbosity::PROFILING) {
    LOG(INFO) << "Initializing heap allocator on unified device memory of size " << format_size(max_device_size());
  }

  s_mps_allocator_initialized.store(true);
}

void MPSHeapAllocatorImpl::init_buffer_pools() {
  // using a container for pools to simplify iterating over them
  // Pool of large buffers with shared storage mode
  m_pools.emplace(BufferPool::Kind::SHARED_LARGE,
                  std::make_unique<BufferPool>(m_device, UsageFlags::SHARED | UsageFlags::HAZARD));
  // Pool of small buffers with shared storage mode
  m_pools.emplace(BufferPool::Kind::SHARED_SMALL,
                  std::make_unique<BufferPool>(m_device, UsageFlags::SMALL | UsageFlags::SHARED | UsageFlags::HAZARD));
  // Pool of small buffers with shared storage mode used to allocate and copy Scalars
  // from CPU to Metal buffers (see allocScalarBufferWithValue()).
  // no Hazard Tracking required for the Scalar pool (synchronized manually).
  m_pools.emplace(BufferPool::Kind::SCALAR,
                  std::make_unique<BufferPool>(m_device, UsageFlags::SMALL | UsageFlags::SHARED | UsageFlags::SCALAR));
}

BufferPool& MPSHeapAllocatorImpl::get_pool(size_t requested_size, size_t aligned_size, uint32_t usage) {
  BufferPool::Kind poolKind;

  if (usage & UsageFlags::SCALAR) {
    poolKind = BufferPool::Kind::SCALAR;
  } else if (aligned_size <= kMaxSmallAlloc) {
    poolKind = BufferPool::Kind::SHARED_SMALL;
  } else {
    poolKind = BufferPool::Kind::SHARED_LARGE;
  }
  return *m_pools[poolKind];
}

size_t MPSHeapAllocatorImpl::get_allocation_size(size_t size, uint32_t usage) const {
  MTLSizeAndAlign sizeAlign = [m_device heapBufferSizeAndAlignWithLength:size options:HeapBlock::getOptions(usage)];
  const size_t aligned = BufferBlock::alignUp(sizeAlign.size, sizeAlign.align);

  // Round large allocations up into coarse buckets so a slowly growing allocation
  // (e.g. a KV cache reallocated at size+epsilon each decode step) reuses the
  // previous step's freed buffer instead of stranding a new heap.
  if ((usage & UsageFlags::SCALAR) || aligned <= kMaxSmallAlloc) {
    return aligned;
  }
  constexpr int kLargeBucketShift = 5; // 32 buckets per power-of-two magnitude
  // The early return above guarantees aligned > kMaxSmallAlloc > 0, so the clz
  // below never operates on 0 (which would be undefined behavior).
  size_t granule = (size_t(1) << (63 - __builtin_clzll(aligned))) >> kLargeBucketShift;
  if (granule < vm_page_size) {
    granule = vm_page_size;
  }
  const size_t bucketed = BufferBlock::alignUp(aligned, granule);
  // Keep the request in its original heap-size class (see getHeapTier): never let
  // rounding cross into a larger class, which would reserve a much larger backing
  // heap, nor push it past Metal's per-buffer limit.
  if (bucketed >= m_max_buffer_size ||
      getHeapTier(bucketed, /*has_memory_pressure=*/false) != getHeapTier(aligned, /*has_memory_pressure=*/false)) {
    return aligned;
  }
  return bucketed;
}

void MPSHeapAllocatorImpl::setHighWatermarkRatio(double ratio) {
  TORCH_CHECK(ratio >= 0.0 && ratio <= default_high_watermark_upper_bound, "invalid high watermark ratio ", ratio);
  m_max_total_allowed_size =
      (ratio == 0.0) ? std::numeric_limits<size_t>::max() : static_cast<size_t>(ratio * (double)max_device_size());
  if (m_debug_verbosity & DebugVerbosity::PROFILING) {
    LOG(INFO) << "High watermark memory allocation limit: "
              << (ratio == 0.0 ? "unlimited" : format_size(m_max_total_allowed_size));
  }
  m_high_watermark_ratio = ratio;
}

void MPSHeapAllocatorImpl::setLowWatermarkRatio(double ratio) {
  // used for comparison with lower_watermark_ratio
  const double high_watermark_limit =
      m_high_watermark_ratio == 0.0 ? default_high_watermark_upper_bound : m_high_watermark_ratio;
  TORCH_CHECK(ratio >= 0.0 && ratio <= high_watermark_limit, "invalid low watermark ratio ", ratio);
  // we use this to detect if there's memory pressure
  m_low_watermark_limit =
      (ratio == 0.0) ? std::numeric_limits<size_t>::max() : static_cast<size_t>(ratio * (double)max_device_size());
  if (m_debug_verbosity & DebugVerbosity::PROFILING) {
    LOG(INFO) << "Low watermark memory allocation limit: "
              << (ratio == 0.0 ? "unlimited" : format_size(m_low_watermark_limit));
  }
  m_low_watermark_ratio = ratio;
}

bool MPSHeapAllocatorImpl::get_free_buffer(AllocParams& params) {
  // this helps to monitor "implicit" allocations from MPS backend and to prevent OOM and system failure.
  if (m_high_watermark_ratio > 0.0 && current_allocated_size() + params.size() > m_max_total_allowed_size) {
    return false;
  }
  BufferPool& pool = *params.pool;
  const size_t alloc_size = params.size();

  // A cached buffer is handed back as it is only to the stream that allocated
  // it, and only when cutting it down would not leave a reusable remainder.
  // In-flight reuse is safe for stream-ordered GPU work, but not for a pinned
  // allocation that the CPU can access immediately.
  // Nor while a MetalGraph capture holds the buffer: the capture re-binds it by
  // address on replay, so handing the same MTLBuffer to a new tensor would make
  // the replay read that tensor's data. The pin set is empty unless a capture is
  // alive, so that is one empty() check in the common case.
  auto stream_it = pool.available_buffers_by_stream.find(getCurrentMPSStream());
  if (stream_it != pool.available_buffers_by_stream.end()) {
    auto it = stream_it->second.lower_bound(&params.search_key);
    if (it != stream_it->second.end() && (*it)->buffer && (*it)->size - alloc_size < pool.min_split &&
        (params.allow_in_flight_reuse || (*it)->retainCount() == 1) && !is_capture_pinned((*it)->buffer)) {
      params.buffer_block = split_free_block(params, *it);
    }
  }
  // Otherwise a new buffer is placed over the free range, which destroys the
  // buffer cached on it. That requires the range to be free of in-flight work,
  // which is also what makes it safe to hand over to a different stream.
  if (!params.buffer_block) {
    auto it = pool.available_buffers.lower_bound(&params.search_key);
    if (it != pool.available_buffers.end() && (!(*it)->buffer || (*it)->retainCount() <= 1)) {
      params.buffer_block = split_free_block(params, *it);
    }
  }
  // no single block fits, so try to build one out of adjacent free ranges
  if (!params.buffer_block && !coalesce_free_blocks(params)) {
    return false;
  }

  if ((m_debug_verbosity & DebugVerbosity::RECYCLES) &&
      (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    LOG(INFO) << "Reusing " << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((pool.usage & UsageFlags::SCALAR) ? " scalar" : "") << " buffer #" << params.buffer_block->buf_id
              << " of size " << format_size(params.buffer_block->size) << " at " << params.buffer_block->buffer
              << " (requested: " << format_size(params.requested_size)
              << ", use#: " << params.buffer_block->use_count + 1 << ", retain#: " << params.buffer_block->retainCount()
              << ")";
  }
  return true;
}

void MPSHeapAllocatorImpl::create_block_buffer(BufferPool& pool, BufferBlock* block) {
  TORCH_INTERNAL_ASSERT(!block->buffer);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(block->offset % pool.alignment == 0);
  block->buffer = block->heap->newMTLBuffer(block->size, pool.usage, block->offset);
  // this should never happen as the backing memory (i.e., heap) was allocated successfully.
  TORCH_INTERNAL_ASSERT(block->buffer);
  block->buf_id = ++BufferBlock::buffer_counter;
  block->stream = getCurrentMPSStream();
  m_allocated_buffers[block->buffer] = block;

  if ((m_debug_verbosity & DebugVerbosity::ALLOCATIONS) &&
      (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    LOG(INFO) << "Allocated " << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((pool.usage & UsageFlags::SCALAR) ? " scalar" : "") << " buffer #" << block->buf_id << " of size "
              << format_size(block->size) << " at " << block->buffer << " from heap #" << block->heap->heap_id
              << " (offset: " << format_size(block->offset) << ", heap free: " << format_size(block->heap->free_bytes)
              << ", total: " << format_size(m_total_allocated_memory.current) << ")";
  }
}

void MPSHeapAllocatorImpl::release_block_buffer(BufferPool& pool, BufferBlock* block) {
  TORCH_INTERNAL_ASSERT(!block->in_use);
  // a free range that was never placed has no resource, pointer-map entry, or
  // stream ownership to remove
  if (!block->buffer) {
    return;
  }
  m_allocated_buffers.erase(block->buffer);
  const uint32_t retainCount = block->heap->releaseMTLBuffer(block->buffer);
  block->cpu_ptr = nullptr;
  block->buf_id = 0;
  block->stream = nullptr;

  if ((m_debug_verbosity & DebugVerbosity::RELEASES) &&
      (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    LOG(INFO) << "Released buffer of size " << format_size(block->size) << " from heap #" << block->heap->heap_id
              << " (use#: " << block->use_count << ", retain#: " << retainCount << ")";
  }
}

BufferBlock* MPSHeapAllocatorImpl::split_free_block(AllocParams& params, BufferBlock* block) {
  BufferPool& pool = *params.pool;
  HeapBlock* heap = block->heap;
  const size_t block_size = block->size;
  const size_t alloc_size = params.size();
  TORCH_INTERNAL_ASSERT(!block->in_use && block_size >= alloc_size);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(block->offset + block_size <= heap->total_size);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(pool.available_size >= block_size && heap->free_bytes >= block_size);

  erase_available_buffer(pool, block);
  pool.available_size -= block_size;
  heap->free_bytes -= block_size;

  const size_t remainder_size = block_size - alloc_size;
  // A buffer that spans the remainder has to be re-placed, and so has one that
  // belongs to another stream: re-placing it is what lets this stream use the
  // range without synchronizing against the stream that had it before.
  if (remainder_size >= pool.min_split || block->stream != getCurrentMPSStream()) {
    release_block_buffer(pool, block);
  }
  if (remainder_size >= pool.min_split) {
    BufferBlock* remainder = new BufferBlock(remainder_size, block->offset + alloc_size, heap);
    remainder->next = block->next;
    block->next = remainder;
    block->size = alloc_size;
    insert_available_buffer(pool, remainder);
    pool.available_size += remainder_size;
    heap->free_bytes += remainder_size;
  }
  if (!block->buffer) {
    create_block_buffer(pool, block);
  }
  // taking free space away can only shrink the heap's largest free run
  heap->max_free_run = std::min(heap->max_free_run, heap->free_bytes);
  block->requested_size = params.requested_size;
  return block;
}

BufferBlock* MPSHeapAllocatorImpl::merge_free_blocks(BufferPool& pool, BufferBlock* first, BufferBlock* last) {
  if (first == last) {
    return first;
  }
  TORCH_INTERNAL_ASSERT(first->heap == last->heap);
  BufferBlock* after = last->next;
  // the merged range is served by a single buffer, so the blocks it absorbs
  // (the first one included) give up the buffers they had cached
  erase_available_buffer(pool, first);
  release_block_buffer(pool, first);

  // keep the first node as the merged range and delete the absorbed ones; this
  // keeps the offset-ordered list valid without replacement metadata
  size_t merged_size = first->size;
  for (BufferBlock* block = first->next; block != after;) {
    BufferBlock* next = block->next;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(block->offset == first->offset + merged_size);
    erase_available_buffer(pool, block);
    release_block_buffer(pool, block);
    merged_size += block->size;
    delete block;
    block = next;
  }
  first->size = merged_size;
  first->next = after;
  insert_available_buffer(pool, first);
  return first;
}

bool MPSHeapAllocatorImpl::coalesce_free_blocks(AllocParams& params) {
  BufferPool& pool = *params.pool;
  const size_t alloc_size = params.size();

  for (HeapBlock* heap : pool.heaps) {
    if (heap->max_free_run < alloc_size) {
      continue;
    }
    // walk the heap in offset order, tracking the run of free blocks and the
    // (possibly shorter) suffix of it that can be merged right now
    size_t largest_run = 0, free_run = 0, merge_run = 0;
    BufferBlock* merge_first = nullptr;
    for (BufferBlock* block = heap->first_block; block != nullptr; block = block->next) {
      if (block->in_use) {
        free_run = merge_run = 0;
        merge_first = nullptr;
        continue;
      }
      free_run += block->size;
      largest_run = std::max(largest_run, free_run);
      // merging destroys the buffers cached on the range, so a block still
      // retained by an outstanding command buffer breaks the run
      if (block->buffer && block->retainCount() > 1) {
        merge_run = 0;
        merge_first = nullptr;
        continue;
      }
      if (!merge_first) {
        merge_first = block;
      }
      merge_run += block->size;
      if (merge_run >= alloc_size) {
        params.buffer_block = split_free_block(params, merge_free_blocks(pool, merge_first, block));
        return true;
      }
    }
    // the walk measured the heap's largest free run exactly, so larger requests
    // can skip this heap until its blocks change
    heap->max_free_run = largest_run;
  }
  return false;
}

bool MPSHeapAllocatorImpl::alloc_heap(AllocParams& params) {
  if (m_max_total_allowed_size != std::numeric_limits<size_t>::max() &&
      current_allocated_size() + params.size() > m_max_total_allowed_size) {
    return false;
  }
  BufferPool& pool = *params.pool;
  HeapBlock* heap = HeapBlock::createHeapBlock(params, pool.device, pool.usage);
  if (!heap) {
    return false; // this will cause releasing pool buffers to free up memory
  }
  m_total_allocated_memory.increase(heap->total_size);
  pool.heaps.insert(heap);

  // the heap starts out as a single free block the request is then cut out of
  BufferBlock* block = new BufferBlock(heap->total_size, 0, heap);
  heap->first_block = block;
  insert_available_buffer(pool, block);
  pool.available_size += heap->total_size;

  if (m_debug_verbosity & DebugVerbosity::ALLOCATIONS) {
    LOG(INFO) << "Allocated " << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private") << " heap #"
              << heap->heap_id << " of size " << format_size(heap->total_size) << " (#heaps: " << pool.heaps.size()
              << ", current allocated: " << format_size(current_allocated_size()) << ")";
  }
  params.buffer_block = split_free_block(params, block);
  return true;
}

void MPSHeapAllocatorImpl::release_heap(BufferPool& pool, HeapBlock* heap) {
  for (BufferBlock* block = heap->first_block; block != nullptr;) {
    BufferBlock* next = block->next;
    TORCH_INTERNAL_ASSERT(!block->in_use);
    erase_available_buffer(pool, block);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(pool.available_size >= block->size);
    pool.available_size -= block->size;
    release_block_buffer(pool, block);
    delete block;
    block = next;
  }
  heap->first_block = nullptr;
  heap->free_bytes = 0;
  pool.heaps.erase(heap);
  m_total_allocated_memory.decrease(heap->total_size);
  const uint32_t retain_count = heap->releaseMTLHeap();
  if (m_debug_verbosity & DebugVerbosity::RELEASES) {
    LOG(INFO) << "Released heap #" << heap->heap_id << " of size " << format_size(heap->total_size)
              << " (current allocated: " << format_size(current_allocated_size()) << ", retain#: " << retain_count
              << ")";
  }
  delete heap;
}

size_t MPSHeapAllocatorImpl::release_free_heaps(BufferPool& pool, size_t target_size) {
  // a heap returns its memory to the system only as a whole, so freeing single
  // blocks reclaims nothing; only heaps no allocation is left in can be released
  std::vector<HeapBlock*> releasable_heaps;
  size_t released_size = 0;
  for (HeapBlock* heap : pool.heaps) {
    if (released_size >= target_size) {
      break;
    }
    if (heap->free_bytes != heap->total_size) {
      continue;
    }
    bool releasable = true;
    for (BufferBlock* block = heap->first_block; block != nullptr; block = block->next) {
      if (block->buffer && block->retainCount() > 1) {
        releasable = false;
        break;
      }
    }
    if (releasable) {
      releasable_heaps.push_back(heap);
      released_size += heap->total_size;
    }
  }

  for (HeapBlock* heap : releasable_heaps) {
    release_heap(pool, heap);
  }
  return released_size;
}

BufferBlock* MPSHeapAllocatorImpl::alloc_buffer_block(size_t size, uint32_t usage, bool allow_in_flight_reuse) {
  TORCH_CHECK(size < m_max_buffer_size, "Invalid buffer size: ", format_size(size));

  size_t alloc_size = get_allocation_size(size, usage);
  auto& pool = get_pool(size, alloc_size, usage);
  AllocParams params(alloc_size, size, &pool, allow_in_flight_reuse);
  // we care about memory pressure if only we're allocating large buffers when the
  // low watermark limit has been reached
  params.has_memory_pressure = !(pool.usage & UsageFlags::SMALL) && getLowWatermarkValue() <= 0;

  // first, try to get a block from the existing pool.
  bool block_found = get_free_buffer(params);
  if (!block_found) {
    // do garbage collection if memory pressure is high and there's enough memory in pool
    if (params.has_memory_pressure && alloc_size < pool.available_size) {
      garbage_collect_cached_buffers(params);
    }

    block_found =
        // Attempt allocate
        alloc_heap(params) ||
        // Callbacks might release more memory (eg. by forcing a GC in the host language) thus
        // we can retry getting a free buffer in the pool, before trying to alloc again.
        (trigger_memory_callbacks(nullptr, IMpsAllocatorCallback::EventType::ALLOCATION_FAILED) &&
         get_free_buffer(params)) ||
        // Give back the heaps that went fully free and retry alloc.
        (release_free_heaps(pool, alloc_size) && alloc_heap(params)) ||
        // Free all cached buffers and retry alloc. The stream sync it does also
        // drops the retain counts that kept free ranges from being coalesced.
        (release_cached_buffers() && (get_free_buffer(params) || alloc_heap(params))) ||
        // Last resort: wait for buffers parked in-flight (freed but still used by
        // the GPU) to complete, reclaim them, and retry -- avoids a spurious
        // watermark OOM when those buffers are about to drain anyway.
        (wait_for_pending_free_buffers(pool) && (get_free_buffer(params) || alloc_heap(params)));
  }

  BufferBlock* buffer_block = params.buffer_block;

  // the OOM could be triggered if:
  //   1- the High Watermark limit has been reached (if enabled)
  //   2- ran out of device memory, or the memory fragmentation is so high that a contiguous
  //      chunk of requested size couldn't be found.
  if (!block_found || !buffer_block) {
    if (m_high_watermark_ratio > 0.0) {
      TORCH_CHECK(
          false,
          "MPS backend out of memory (MPS allocated: ",
          format_size(m_total_allocated_memory.current),
          ", other allocations: ",
          format_size(current_allocated_size() - m_total_allocated_memory.current),
          ", max allowed: ",
          format_size(m_max_total_allowed_size),
          "). Tried to allocate ",
          format_size(alloc_size),
          " on ",
          ((pool.usage & UsageFlags::SHARED) ? "shared" : "private"),
          " pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).");
    } else {
      TORCH_CHECK(false,
                  "MPS backend out of memory (MPS allocated: ",
                  format_size(m_total_allocated_memory.current),
                  ", other allocations: ",
                  format_size(current_allocated_size() - m_total_allocated_memory.current),
                  "). Tried to allocate ",
                  format_size(alloc_size),
                  " on ",
                  ((pool.usage & UsageFlags::SHARED) ? "shared" : "private"),
                  " pool.");
    }
  }
  buffer_block->in_use = true;
  buffer_block->use_count++;
  m_current_allocated_memory.increase(buffer_block->size);
  m_active_bytes.increase(buffer_block->size);
  return buffer_block;
}

bool MPSHeapAllocatorImpl::insert_available_buffer(BufferPool& pool, BufferBlock* buffer_block) {
  bool inserted = pool.available_buffers.insert(buffer_block).second;
  auto it = pool.available_buffers_by_stream.find(buffer_block->stream);
  if (it == pool.available_buffers_by_stream.end()) {
    it = pool.available_buffers_by_stream.emplace(buffer_block->stream, std::set<BufferBlock*, BufferComparison>())
             .first;
  }
  it->second.insert(buffer_block);
  return inserted;
}

void MPSHeapAllocatorImpl::erase_available_buffer(BufferPool& pool, BufferBlock* buffer_block) {
  pool.available_buffers.erase(buffer_block);
  auto it = pool.available_buffers_by_stream.find(buffer_block->stream);
  if (it != pool.available_buffers_by_stream.end()) {
    it->second.erase(buffer_block);
  }
}

void MPSHeapAllocatorImpl::free_buffer(BufferBlock* buffer_block) {
  TORCH_INTERNAL_ASSERT(buffer_block->in_use);

  BufferPool& pool = *buffer_block->heap->pool;
  // Makes sure the BufferBlock* isn't already present in the pool we're freeing it back into.
  TORCH_INTERNAL_ASSERT(insert_available_buffer(pool, buffer_block));
  pool.available_size += buffer_block->size;
  buffer_block->shape.clear(); // reset shape
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_current_allocated_memory.current >= static_cast<int64_t>(buffer_block->size));
  m_current_allocated_memory.decrease(buffer_block->size);
  m_active_bytes.decrease(buffer_block->size);
  if (buffer_block->event) {
    // returns the MPSEvent back to MPSEventPool
    buffer_block->event.reset();
  }
  buffer_block->in_use = false;
  HeapBlock* heap = buffer_block->heap;
  heap->free_bytes += buffer_block->size;
  // the block may now be adjacent to other free ones; free_bytes bounds how much
  // coalescing them could yield
  heap->max_free_run = heap->free_bytes;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(heap->free_bytes <= heap->total_size);
}

BufferBlock* MPSHeapAllocatorImpl::get_allocated_buffer_block(const void* ptr) {
  auto it = m_allocated_buffers.find(ptr);
  if (it == m_allocated_buffers.end()) {
    return nullptr;
  }
  return it->second;
}

bool MPSHeapAllocatorImpl::release_cached_buffers() {
  if (m_debug_verbosity >= DebugVerbosity::PROFILING) {
    LOG(INFO) << "Attempting to release cached buffers (MPS allocated: "
              << format_size(m_total_allocated_memory.current)
              << ", other allocations: " << format_size(current_allocated_size() - m_total_allocated_memory.current)
              << ")";
  }
  // before releasing the buffers make sure the command buffer has finished.
  // we need to release the lock temporarily as synchronizing may cause deadlock with completion handlers.
  m_mutex.unlock();
  auto stream = getDefaultMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  m_mutex.lock();
  // Give back every heap no allocation is left in
  for (const auto& poolIt : m_pools) {
    release_free_heaps(*poolIt.second, std::numeric_limits<size_t>::max());
  }
  return true;
}

void MPSHeapAllocatorImpl::garbage_collect_cached_buffers(AllocParams& params) {
  // skip garbage collection if memory pressure has already relieved
  if (current_allocated_size() < m_low_watermark_limit) {
    return;
  }
  // attempt to collect garbage until we reach below low watermark limit
  const size_t target_size = current_allocated_size() - m_low_watermark_limit;
  BufferPool& pool = *params.pool;
  const size_t gc_reclaimed = release_free_heaps(pool, target_size);

  if (m_debug_verbosity & DebugVerbosity::RELEASES) {
    LOG(INFO) << "Garbage collected " << format_size(gc_reclaimed) << " from "
              << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private")
              << " pool (target: " << format_size(target_size) << ", #heaps: " << pool.heaps.size() << ")";
  }
}

// public interface to MPSAllocator
id<MTLBuffer> MPSHeapAllocatorImpl::malloc(size_t size, uint32_t usage) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  // Return any buffers parked in-flight by free() whose GPU work has since
  // completed back to their pools before serving this allocation.
  freeInactiveBuffers();
  BufferBlock* buffer_block = alloc_buffer_block(size, usage, /*allow_in_flight_reuse=*/true);
  return buffer_block ? buffer_block->buffer : nullptr;
}

id<MTLBuffer> MPSHeapAllocatorImpl::malloc_host(size_t size, uint32_t usage) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  freeInactiveBuffers();
  BufferBlock* buffer_block = alloc_buffer_block(size, usage, /*allow_in_flight_reuse=*/false);
  return buffer_block ? buffer_block->buffer : nullptr;
}

bool MPSHeapAllocatorImpl::isSharedBuffer(const void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  BufferBlock* buffer_block = get_allocated_buffer_block(ptr);
  // it's OK for the buffer_block to not exist yet
  return buffer_block && (buffer_block->heap->pool->usage & UsageFlags::SHARED);
}

id<MTLBuffer> MPSHeapAllocatorImpl::allocScalarBufferWithValue(void* value, size_t size) {
  BufferBlock* buffer_block = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    buffer_block = alloc_buffer_block(size, UsageFlags::SCALAR, /*allow_in_flight_reuse=*/true);
    if (!buffer_block) {
      return nullptr;
    }
    if (!buffer_block->cpu_ptr) {
      buffer_block->cpu_ptr = [buffer_block->buffer contents];
    }
  }
  // buffer is out of the pool, so no mutex lock is needed
  memcpy(buffer_block->cpu_ptr, value, size);
  return buffer_block->buffer;
}

std::pair<const void*, uint32_t> MPSHeapAllocatorImpl::getSharedBufferPtr(const void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  BufferBlock* buffer_block = get_allocated_buffer_block(ptr);
  // return if buffer was not allocated on MPSAllocator or isn't a Shared buffer
  if (!buffer_block || !(buffer_block->heap->pool->usage & UsageFlags::SHARED)) {
    return {nullptr, 0};
  }
  if (!buffer_block->cpu_ptr) {
    buffer_block->cpu_ptr = [buffer_block->buffer contents];
  }
  return {buffer_block->cpu_ptr, buffer_block->retainCount()};
}

namespace {
// Deleter for the CPU-device DataPtr produced by getHostAliasStorage.
// The context is a heap-allocated c10::Storage holding a refcount on the
// source MPS storage; destroying it releases that refcount, which in turn
// lets the MPSAllocator recycle the underlying MTLBuffer.
void hostAliasDeleter(void* ctx) {
  delete static_cast<c10::Storage*>(ctx);
}
} // namespace

c10::Storage MPSHeapAllocatorImpl::getHostAliasStorage(const c10::Storage& mps_storage) {
  TORCH_CHECK_VALUE(mps_storage.device().type() == c10::DeviceType::MPS,
                    "getHostAliasStorage: expected an MPS storage, got device=",
                    mps_storage.device());

  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  BufferBlock* buffer_block = get_allocated_buffer_block(mps_storage.data());
  TORCH_CHECK(buffer_block, "getHostAliasStorage: storage was not allocated by the MPSAllocator");
  TORCH_CHECK(buffer_block->heap->pool->usage & UsageFlags::SHARED,
              "getHostAliasStorage: storage is not backed by a shared (unified) MTLBuffer");

  if (!buffer_block->cpu_ptr) {
    buffer_block->cpu_ptr = [buffer_block->buffer contents];
  }

  // Retain the source MPS storage through the DataPtr's context so the
  // MTLBuffer cannot be recycled while the host alias is in use.
  auto* ctx = new c10::Storage(mps_storage);
  c10::DataPtr data_ptr(buffer_block->cpu_ptr, ctx, &hostAliasDeleter, c10::Device(c10::DeviceType::CPU));

  return c10::Storage(c10::Storage::use_byte_size_t(),
                      mps_storage.nbytes(),
                      std::move(data_ptr),
                      /*allocator=*/nullptr,
                      /*resizable=*/false);
}

bool MPSHeapAllocatorImpl::recordEvents(c10::ArrayRef<const void*> buffers) {
  bool recordedEvent = false;
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  for (const auto& buffer : buffers) {
    BufferBlock* buffer_block = get_allocated_buffer_block(buffer);
    // return if buffer was not allocated on MPSAllocator or isn't a Shared buffer
    if (buffer_block && (buffer_block->heap->pool->usage & UsageFlags::SHARED)) {
      if (!buffer_block->event) {
        buffer_block->event = m_event_pool->acquireEvent(false, nullptr);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(buffer_block->event);
      }
      buffer_block->event->record(/*needsLock*/ false);
      recordedEvent = true;
    }
  }
  return recordedEvent;
}

bool MPSHeapAllocatorImpl::waitForEvents(c10::ArrayRef<const void*> buffers) {
  std::vector<BufferBlock*> buffer_blocks;
  {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    for (const auto& buffer : buffers) {
      BufferBlock* buffer_block = get_allocated_buffer_block(buffer);
      // wait on event if "shared" buffer was allocated on MPSAllocator and
      // or actually needs waiting (based on retainCount)
      if (buffer_block && (buffer_block->heap->pool->usage & UsageFlags::SHARED) && buffer_block->retainCount() > 1 &&
          buffer_block->event) {
        buffer_blocks.push_back(buffer_block);
      }
    }
  }
  bool waitedForEvent = false;

  for (const auto& buffer_block : buffer_blocks) {
    // check for retain count again as the previous wait might have released the buffer
    if (buffer_block->retainCount() > 1) {
      bool waitedOnCPU = buffer_block->event->synchronize();
      if (waitedOnCPU) {
        // after waiting, it's a good time to free some pending inactive buffers
        freeInactiveBuffers();
        waitedForEvent |= buffer_block->retainCount() <= 1;
      } else {
        // even if one of the buffers weren't recorded beforehand, we return
        // without continuing with other buffers since retainCount > 1
        waitedForEvent = false;
        break;
      }
    }
  }
  return waitedForEvent;
}

bool MPSHeapAllocatorImpl::wait_for_pending_free_buffers(BufferPool& pool) {
  std::vector<const void*> buffers;
  {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    buffers.reserve(pool.buffers_pending_free.size());
    for (BufferBlock* buffer_block : pool.buffers_pending_free) {
      buffers.push_back(buffer_block->buffer);
    }
  }
  // waitForEvents CPU-waits on the in-flight buffers and calls freeInactiveBuffers,
  // returning the completed ones to the pool for the caller to retry allocation.
  return !buffers.empty() && waitForEvents(buffers);
}

void MPSHeapAllocatorImpl::pinBufferForCapture(const void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  m_capture_pinned[ptr]++;
}

void MPSHeapAllocatorImpl::unpinBufferForCapture(const void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  auto it = m_capture_pinned.find(ptr);
  if (it != m_capture_pinned.end() && --it->second == 0) {
    m_capture_pinned.erase(it);
  }
}

id_t MPSHeapAllocatorImpl::getBufferId(const void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  BufferBlock* buffer_block = get_allocated_buffer_block(ptr);
  return buffer_block ? buffer_block->buf_id : 0;
}

ssize_t MPSHeapAllocatorImpl::getUnalignedBufferSize(const void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  BufferBlock* buffer_block = get_allocated_buffer_block(ptr);
  if (buffer_block) {
    return (ssize_t)buffer_block->requested_size;
  }
  // -1 indicates the passed buffer pointer wasn't found
  return -1;
}

void MPSHeapAllocatorImpl::setBufferShape(const void* ptr, const IntArrayRef& shape) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  BufferBlock* buffer_block = get_allocated_buffer_block(ptr);
  TORCH_INTERNAL_ASSERT(buffer_block, "failed to find the buffer ", ptr);
  // note that the IntArrayRef doesn't own the underlying data, and the backing
  // memory for shape data must persist as long as the buffer is in use.
  // So we need to copy to vector.
  buffer_block->shape = shape.vec();
}

IntArrayRef MPSHeapAllocatorImpl::getBufferShape(const void* ptr) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  BufferBlock* buffer_block = get_allocated_buffer_block(ptr);
  if (buffer_block && !buffer_block->shape.empty()) {
    return IntArrayRef{buffer_block->shape};
  }
  return IntArrayRef();
}

void MPSHeapAllocatorImpl::free(void* ptr) {
  BufferBlock* buffer_block = nullptr;
  {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);

    buffer_block = get_allocated_buffer_block(ptr);
    TORCH_INTERNAL_ASSERT(buffer_block);
    BufferPool& pool = *buffer_block->heap->pool;
    if (!(pool.usage & UsageFlags::SCALAR)) {
      // A buffer marked by recordEvents (used by the GPU in a context where a
      // subsequent CPU reuse could race it, e.g. a pinned-memory copy) and still
      // referenced by an in-flight command buffer must not be recycled yet:
      // handing it to a new allocation could let the CPU overwrite it before the
      // GPU is done. Park it in limbo; freeInactiveBuffers() reclaims it once its
      // command buffer completes. Unmarked buffers are only GPU-accessed, so
      // serial-stream ordering already makes immediate reuse safe.
      if (buffer_block->event && buffer_block->retainCount() > 1) {
        pool.buffers_pending_free.insert(buffer_block);
      } else {
        free_buffer(buffer_block);
      }
      return;
    }
  }
  // we sync the scalar pool manually with completion handler at the time buffer is
  // freed when the MPSScalar instance goes our of scope
  m_stream->addCompletedHandler(^(id<MTLCommandBuffer>) {
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    free_buffer(buffer_block);
  });
}

void MPSHeapAllocatorImpl::freeInactiveBuffers() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  for (const auto& poolIt : m_pools) {
    BufferPool& pool = *poolIt.second;
    if (!pool.buffers_pending_free.empty()) {
      for (auto it = pool.buffers_pending_free.begin(), last = pool.buffers_pending_free.end(); it != last;) {
        BufferBlock* buffer_block = *it;
        if (buffer_block->retainCount() <= 1) {
          it = pool.buffers_pending_free.erase(it);
          free_buffer(buffer_block);
        } else {
          ++it;
        }
      }
    }
  }
}

void MPSHeapAllocatorImpl::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  release_cached_buffers();
}

ssize_t MPSHeapAllocatorImpl::getLowWatermarkValue() {
  // check if low watermark limit is disabled
  if (m_low_watermark_ratio == 0.0) {
    return std::numeric_limits<ssize_t>::max();
  }
  // current_allocated_size could exceed m_low_watermark_limit (e.g., when swapping to disk)
  return std::max<ssize_t>(0, (ssize_t)(m_low_watermark_limit - current_allocated_size()) / 1048576L);
}

c10::CachingDeviceAllocator::DeviceStats MPSHeapAllocatorImpl::getDeviceStats() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  c10::CachingDeviceAllocator::DeviceStats stats;
  // MPS does not distinguish small/large pools in these stats, so only the
  // aggregate ("all") entry is populated.
  constexpr auto kAggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  stats.allocated_bytes[kAggregate] = m_current_allocated_memory;
  stats.reserved_bytes[kAggregate] = m_total_allocated_memory;
  stats.active_bytes[kAggregate] = m_active_bytes;
  return stats;
}

void MPSHeapAllocatorImpl::resetAccumulatedStats() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  m_current_allocated_memory.reset_accumulated();
  m_total_allocated_memory.reset_accumulated();
  m_active_bytes.reset_accumulated();
}

void MPSHeapAllocatorImpl::resetPeakStats() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  m_current_allocated_memory.reset_peak();
  m_total_allocated_memory.reset_peak();
  m_active_bytes.reset_peak();
}

inline std::string MPSHeapAllocatorImpl::format_size(uint64_t size) const {
  return c10::CachingAllocator::format_size(size);
}

} // namespace HeapAllocator

// Use "at::mps::GetMPSAllocator()" to acquire a handle to MPS Allocator
namespace {
HeapAllocator::MPSHeapAllocatorImpl& _getAllocImpl() {
  static HeapAllocator::MPSHeapAllocatorImpl s_allocatorImpl;
  return s_allocatorImpl;
}
} // namespace

// MPS allocator struct to be registered with Pytorch
struct TORCH_API MPSAllocator final : public IMPSAllocator {
 public:
  // Construction is intentionally cheap (no Metal access) so the allocator can
  // be registered with c10 at static-init time without forcing MPS/Metal
  // initialization in processes that never touch the MPS device.
  explicit MPSAllocator(uint32_t Usage) : m_usage(Usage) {}

  // No destructor: the underlying MPSHeapAllocatorImpl singleton empties its own
  // cache in its destructor. Calling _getAllocImpl() from here at process exit
  // would be unsafe, since that singleton may already have been destroyed.
  DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

  DataPtr allocate(const size_t nbytes) override {
    __block id<MTLBuffer> buf = nbytes > 0 ? _getAllocImpl().malloc(nbytes, m_usage) : nullptr;
    return {buf, buf, &Delete, at::Device(at::DeviceType::MPS, 0)};
  }
  DataPtr allocateForHost(const size_t nbytes) {
    __block id<MTLBuffer> buf = nbytes > 0 ? _getAllocImpl().malloc_host(nbytes, m_usage) : nullptr;
    return {buf, buf, &Delete, at::Device(at::DeviceType::MPS, 0)};
  }

  // implementation of IMPSAllocator interface
  DataPtr allocScalarBufferWithValue(void* value, size_t size) const override {
    id<MTLBuffer> buf = _getAllocImpl().allocScalarBufferWithValue(value, size);
    return {buf, buf, &Delete, at::Device(at::DeviceType::MPS, 0)};
  }
  std::pair<const void*, uint32_t> getSharedBufferPtr(const void* ptr) const override {
    return _getAllocImpl().getSharedBufferPtr(ptr);
  }
  c10::Storage getHostAliasStorage(const c10::Storage& mps_storage) const override {
    return _getAllocImpl().getHostAliasStorage(mps_storage);
  }
  bool isSharedBuffer(const void* ptr) const override {
    return _getAllocImpl().isSharedBuffer(ptr);
  }
  // c10::DeviceAllocator interface
  bool initialized() override {
    return HeapAllocator::s_mps_allocator_initialized.load();
  }
  void emptyCache(c10::MempoolId_t mempool_id [[maybe_unused]] = {0, 0}) override {
    _getAllocImpl().emptyCache();
  }
  void recordStream(const DataPtr& ptr [[maybe_unused]], c10::Stream stream [[maybe_unused]]) override {
    // MPS executes on a single serial stream, so there is no cross-stream
    // dependency to track for buffer reuse.
  }
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device [[maybe_unused]]) override {
    return _getAllocImpl().getDeviceStats();
  }
  void resetAccumulatedStats(c10::DeviceIndex device [[maybe_unused]]) override {
    _getAllocImpl().resetAccumulatedStats();
  }
  void resetPeakStats(c10::DeviceIndex device [[maybe_unused]]) override {
    _getAllocImpl().resetPeakStats();
  }
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device [[maybe_unused]]) override {
    const size_t total = _getAllocImpl().getRecommendedMaxMemory();
    const size_t used = _getAllocImpl().getDriverAllocatedMemory();
    return {total > used ? total - used : 0, total};
  }
  void freeInactiveBuffers() const override {
    _getAllocImpl().freeInactiveBuffers();
  }
  ssize_t getUnalignedBufferSize(const void* ptr) const override {
    return _getAllocImpl().getUnalignedBufferSize(ptr);
  }
  id_t getBufferId(const void* ptr) const override {
    return _getAllocImpl().getBufferId(ptr);
  };
  void pinBufferForCapture(const void* ptr) const override {
    _getAllocImpl().pinBufferForCapture(ptr);
  };
  void unpinBufferForCapture(const void* ptr) const override {
    _getAllocImpl().unpinBufferForCapture(ptr);
  };
  IntArrayRef getBufferShape(const void* ptr) const override {
    return _getAllocImpl().getBufferShape(ptr);
  }
  void setBufferShape(const void* ptr, const IntArrayRef& shape) const override {
    _getAllocImpl().setBufferShape(ptr, shape);
  }
  size_t getTotalAllocatedMemory() const override {
    return _getAllocImpl().getTotalAllocatedMemory();
  }
  size_t getCurrentAllocatedMemory() const override {
    return _getAllocImpl().getCurrentAllocatedMemory();
  }
  size_t getDriverAllocatedMemory() const override {
    return _getAllocImpl().getDriverAllocatedMemory();
  }
  size_t getRecommendedMaxMemory() const override {
    return _getAllocImpl().getRecommendedMaxMemory();
  }
  ssize_t getLowWatermarkValue() const override {
    return _getAllocImpl().getLowWatermarkValue();
  }
  size_t getLowWatermarkLimit() const override {
    return _getAllocImpl().getLowWatermarkLimit();
  }
  size_t getHighWatermarkLimit() const override {
    return _getAllocImpl().getHighWatermarkLimit();
  }
  void setLowWatermarkRatio(double ratio) const override {
    _getAllocImpl().setLowWatermarkRatio(ratio);
  }
  void setHighWatermarkRatio(double ratio) const override {
    _getAllocImpl().setHighWatermarkRatio(ratio);
  }
  bool recordEvents(c10::ArrayRef<const void*> buffers) const override {
    return _getAllocImpl().recordEvents(buffers);
  }
  bool waitForEvents(c10::ArrayRef<const void*> buffers) const override {
    return _getAllocImpl().waitForEvents(buffers);
  }
  std::string formatSize(size_t size) const override {
    return _getAllocImpl().format_size(size);
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }

 private:
  uint32_t m_usage;

  static void Delete(void* ptr) {
    if (ptr) {
      _getAllocImpl().free(ptr);
    }
  }
};

namespace {
MPSAllocator& _getSharedAllocator() {
  static MPSAllocator s_mps_shared_alloc(HeapAllocator::UsageFlags::SHARED);
  return s_mps_shared_alloc;
}

// Register the shared allocator as the c10 allocator for MPS at static-init
// time so the generic c10::GetAllocator(MPS) / at::getDeviceAllocator(MPS) paths
// (used by the torch.accelerator memory APIs) resolve to it. The allocator is
// cheap to construct and does not touch Metal, so this does not force MPS
// initialization; DeviceAllocator::initialized() reports actual readiness.
struct MPSAllocatorRegisterer {
  MPSAllocatorRegisterer() {
    c10::SetAllocator(c10::DeviceType::MPS, &_getSharedAllocator());
  }
};
static MPSAllocatorRegisterer s_mps_allocator_registerer;

// Allocator returned by MPSHooks::getPinnedMemoryAllocator(). Backs each
// allocation with a shared (unified-memory) MTLBuffer from the MPS allocator
// but exposes the buffer's host-visible pointer as a CPU-device DataPtr, so
// `torch.empty(..., device="cpu", pin_memory=True)` and `tensor.pin_memory()`
// stay on the CPU device while still being zero-copy reachable from the GPU.
class MPSPinnedAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t nbytes) override {
    if (nbytes == 0) {
      return {nullptr, nullptr, &deleter, c10::Device(c10::DeviceType::CPU)};
    }
    auto& shared = _getSharedAllocator();
    c10::DataPtr mps_dp = shared.allocateForHost(nbytes);
    // allocateForHost() returns a DataPtr whose data pointer is the
    // id<MTLBuffer> itself; capture it before mps_dp is moved into the backing
    // storage.
    void* mtl_buffer = mps_dp.get();
    auto host_ptr_pair = shared.getSharedBufferPtr(mtl_buffer);
    TORCH_INTERNAL_ASSERT(host_ptr_pair.first, "MPS pinned allocator: failed to map shared buffer");
    void* cpu_ptr = const_cast<void*>(host_ptr_pair.first);
    // Hold a refcount on the source MPS storage so the MTLBuffer stays alive
    // for as long as the host alias is in use.
    c10::Storage mps_storage(c10::Storage::use_byte_size_t(),
                             nbytes,
                             std::move(mps_dp),
                             /*allocator=*/nullptr,
                             /*resizable=*/false);
    auto* ctx = new PinnedCtx{std::move(mps_storage), cpu_ptr};
    {
      std::lock_guard<std::mutex> lk(s_mutex);
      s_pinned_buffers.emplace(cpu_ptr, mtl_buffer);
    }
    return {cpu_ptr, ctx, &deleter, c10::Device(c10::DeviceType::CPU)};
  }
  c10::DeleterFnPtr raw_deleter() const override {
    return &deleter;
  }
  void copy_data(void* dest, const void* src, size_t count) const final {
    default_copy_data(dest, src, count);
  }
  static bool isPinned(const void* ptr) {
    return getMTLBuffer(ptr) != nullptr;
  }
  // Returns the shared MTLBuffer backing a pinned host allocation (the base of
  // the storage, keyed by the host-visible pointer), or nullptr if not pinned.
  static void* getMTLBuffer(const void* ptr) {
    if (!ptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lk(s_mutex);
    auto it = s_pinned_buffers.find(ptr);
    return it == s_pinned_buffers.end() ? nullptr : it->second;
  }

 private:
  struct PinnedCtx {
    c10::Storage mps_storage;
    void* cpu_ptr;
  };
  static void deleter(void* ctx) {
    if (!ctx) {
      return;
    }
    auto* pinned = static_cast<PinnedCtx*>(ctx);
    {
      std::lock_guard<std::mutex> lk(s_mutex);
      s_pinned_buffers.erase(pinned->cpu_ptr);
    }
    delete pinned;
  }
  static std::mutex s_mutex;
  static std::unordered_map<const void*, void*> s_pinned_buffers;
};
std::mutex MPSPinnedAllocator::s_mutex;
std::unordered_map<const void*, void*> MPSPinnedAllocator::s_pinned_buffers;

MPSPinnedAllocator& _getPinnedAllocator() {
  static MPSPinnedAllocator s_mps_pinned_alloc;
  return s_mps_pinned_alloc;
}

} // anonymous namespace

IMPSAllocator* getIMPSAllocator() {
  // MPS requires unified memory (enforced in MPSHeapAllocatorImpl::init_allocator),
  // so the shared allocator is always usable.
  return &_getSharedAllocator();
}

c10::Allocator* getMPSPinnedAllocator() {
  // MPS requires unified memory (enforced in MPSHeapAllocatorImpl::init_allocator),
  // so shared (unified-memory) buffers backing the pinned alias are always usable.
  return &_getPinnedAllocator();
}

void* getMPSPinnedMTLBuffer(const void* host_ptr) {
  return MPSPinnedAllocator::getMTLBuffer(host_ptr);
}

// torch.is_pinned() implementation
// Pinned memory will be helpful on Apple Silicon Macs with Unified memory as we
// will be able to use SharedStorageMode for MTLBuffer allocations. This will
// avoid extra copies on DataLoading operations.
bool isMPSPinnedPtr(const void* data) {
  return MPSPinnedAllocator::isPinned(data) || at::mps::_getSharedAllocator().isSharedBuffer(data);
}

} // namespace at::mps
