//  Copyright © 2022 Apple Inc.

#include <ATen/CPUFunctions.h>
#include <ATen/EmptyTensor.h>
#include <ATen/mps/MPSAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <c10/util/env.h>

#include <iostream>

namespace at::mps {

C10_DEFINE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback)

namespace HeapAllocator {

uint64_t BufferBlock::buffer_counter = 0;
uint64_t HeapBlock::heap_counter = 0;

void MPSHeapAllocatorImpl::init_allocator() {
  init_buffer_pools();

  // debug verbosity flags (see DebugVerbosity enum)
  static const auto verbosity_str = c10::utils::get_env("PYTORCH_DEBUG_MPS_ALLOCATOR");
  m_debug_verbosity = verbosity_str ? strtol(verbosity_str->c_str(), nullptr, 0) : DebugVerbosity::SILENT;

  static const auto high_watermark_ratio_str = c10::utils::get_env("PYTORCH_MPS_HIGH_WATERMARK_RATIO");
  const double high_watermark_ratio =
      high_watermark_ratio_str ? strtod(high_watermark_ratio_str->c_str(), nullptr) : default_high_watermark_ratio;
  setHighWatermarkRatio(high_watermark_ratio);

  const double default_low_watermark_ratio =
      m_device.hasUnifiedMemory ? default_low_watermark_ratio_unified : default_low_watermark_ratio_discrete;
  static const auto low_watermark_ratio_str = c10::utils::get_env("PYTORCH_MPS_LOW_WATERMARK_RATIO");
  const double low_watermark_ratio =
      low_watermark_ratio_str ? strtod(low_watermark_ratio_str->c_str(), nullptr) : default_low_watermark_ratio;
  setLowWatermarkRatio(low_watermark_ratio);
}

void MPSHeapAllocatorImpl::init_buffer_pools() {
  // using a container for pools to simplify iterating over them
  // Pool of large buffers with private storage mode
  m_pools.emplace(BufferPool::Kind::PRIVATE_LARGE,
                  std::make_unique<BufferPool>(m_device, UsageFlags::PRIVATE | UsageFlags::HAZARD));
  // Pool of large buffers with shared storage mode
  m_pools.emplace(BufferPool::Kind::SHARED_LARGE,
                  std::make_unique<BufferPool>(m_device, UsageFlags::SHARED | UsageFlags::HAZARD));
  // Pool of small buffers with private storage mode
  m_pools.emplace(BufferPool::Kind::PRIVATE_SMALL,
                  std::make_unique<BufferPool>(m_device, UsageFlags::SMALL | UsageFlags::PRIVATE | UsageFlags::HAZARD));
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
  } else if (requested_size <= kMaxScalarAlloc && m_device.hasUnifiedMemory) {
    poolKind = BufferPool::Kind::SHARED_SMALL;
  } else if (aligned_size <= kMaxSmallAlloc) {
    poolKind = (usage & UsageFlags::SHARED) ? BufferPool::Kind::SHARED_SMALL : BufferPool::Kind::PRIVATE_SMALL;
  } else {
    poolKind = (usage & UsageFlags::SHARED) ? BufferPool::Kind::SHARED_LARGE : BufferPool::Kind::PRIVATE_LARGE;
  }
  return *m_pools[poolKind];
}

size_t MPSHeapAllocatorImpl::get_allocation_size(size_t size, uint32_t usage) const {
  MTLSizeAndAlign sizeAlign = [m_device heapBufferSizeAndAlignWithLength:size options:HeapBlock::getOptions(usage)];
  return BufferBlock::alignUp(sizeAlign.size, sizeAlign.align);
}

void MPSHeapAllocatorImpl::setHighWatermarkRatio(double ratio) {
  TORCH_CHECK(ratio >= 0.0 && ratio <= default_high_watermark_upper_bound, "invalid high watermark ratio ", ratio);
  m_max_total_allowed_size =
      (ratio == 0.0) ? std::numeric_limits<size_t>::max() : static_cast<size_t>(ratio * (double)max_device_size());
  if (m_debug_verbosity & DebugVerbosity::PROFILING) {
    std::cerr << "\nHigh watermark memory allocation limit: "
              << (ratio == 0.0 ? "unlimited" : format_size(m_max_total_allowed_size)) << "\n";
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
    std::cerr << "Low watermark memory allocation limit: "
              << (ratio == 0.0 ? "unlimited" : format_size(m_low_watermark_limit)) << "\n";
  }
  m_low_watermark_ratio = ratio;
}

HeapBlock* MPSHeapAllocatorImpl::get_free_heap(AllocParams& params) {
  BufferPool& pool = *params.pool;
  HeapBlock* heap_block = nullptr;
  HeapBlock search_key(params.size());

  auto it = pool.heaps.lower_bound(&search_key);
  if (it == pool.heaps.end()) {
    heap_block = HeapBlock::createHeapBlock(params, pool.device, pool.usage);
    if (heap_block) {
      m_total_allocated_memory += heap_block->size.total;
      if (m_debug_verbosity & DebugVerbosity::ALLOCATIONS) {
        std::cerr << "\nAllocated " << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private") << " heap #"
                  << heap_block->heap_id << " of size " << format_size(heap_block->size.total)
                  << " (#heaps: " << (pool.heaps.size() + 1)
                  << ", current allocated: " << format_size(current_allocated_size()) << ")\n";
      }
    }
  } else {
    heap_block = *it;
    // remove and re-insert heap in the set later after a buffer is created.
    // this ensures updating the order of heaps based on their new available sizes
    pool.heaps.erase(it);
  }
  return heap_block;
}

bool MPSHeapAllocatorImpl::alloc_buffer(AllocParams& params) {
  if (m_max_total_allowed_size != std::numeric_limits<size_t>::max() &&
      current_allocated_size() + params.size() > m_max_total_allowed_size) {
    return false;
  }
  HeapBlock* heap = get_free_heap(params);
  if (!heap) {
    return false; // this will cause releasing pool buffers to free up memory
  }
  BufferPool& pool = *params.pool;

  id<MTLBuffer> buffer = heap->newMTLBuffer(params.size(), pool.usage);
  // this should never happen as the backing memory (i.e., heap) was allocated successfully.
  TORCH_INTERNAL_ASSERT(buffer);
  // insert heap after a buffer was created on it to update the order of heap's set
  pool.heaps.insert(heap);
  params.buffer_block = new BufferBlock(params.size(), params.requested_size, buffer, heap);
  m_allocated_buffers[params.buffer_block->buffer] = params.buffer_block;
  pool.allocated_size += params.size();
  pool.n_buffers++;

  if ((m_debug_verbosity & DebugVerbosity::ALLOCATIONS) &&
      (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    std::cerr << "Allocated " << ((params.pool->usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((params.pool->usage & UsageFlags::SCALAR) ? " scalar" : "") << " buffer #"
              << params.buffer_block->buf_id << " of size " << format_size(params.size()) << " at "
              << params.buffer_block->buffer << " from heap #" << heap->heap_id
              << " (requested: " << format_size(params.requested_size)
              << ", heap: " << format_size(heap->size.available) << ", total: " << format_size(m_total_allocated_memory)
              << ")\n";
  }
  return true;
}

bool MPSHeapAllocatorImpl::get_free_buffer(AllocParams& params) {
  // this helps to monitor "implicit" allocations from MPS backend and to prevent OOM and system failure.
  if (m_high_watermark_ratio > 0.0 && current_allocated_size() + params.size() > m_max_total_allowed_size) {
    return false;
  }
  BufferPool& pool = *params.pool;
  // track buffer reuse intervals only on large pool when low watermark limit is enabled.
  if (m_low_watermark_ratio > 0.0 && !(pool.usage & UsageFlags::SMALL)) {
    for (auto& b : pool.available_buffers) {
      ++b->gc_count;
    }
  }
  auto it = pool.available_buffers.lower_bound(&params.search_key);
  if (it != pool.available_buffers.end()) {
    BufferBlock* buffer_block = *it;

    // the logic in here is simple: keep reusing existing heaps capacity as long as possible (by splitting
    // or releasing oversize buffers, if required), and avoid 'new' heap allocations as much as possible.
    if (buffer_block->size <= params.size() + kLargeHeap) {
      // return the existing buffer if it already fits the requested size (i.e., not oversize)
      params.buffer_block = buffer_block;
    } else {
      HeapBlock search_key(params.size());
      // if there's an 'existing' heap with enough capacity, then don't
      // return the oversize buffer and sub-allocate from that existing heap.
      if (pool.heaps.lower_bound(&search_key) != pool.heaps.end()) {
        params.buffer_block = nullptr;
      } else if (buffer_block->retainCount() <= 1) {
        // otherwise if buffer is releasable immediately, we make room by releasing the
        // buffer and reuse the new space within its heap container for the new smaller buffer allocation
        release_buffer(buffer_block, false);
        // this will skip unnecessary garbage collection as we'll reuse the newly released space
        params.has_memory_pressure = false;
      } else if (params.has_memory_pressure) {
        // the oversized buffer is busy and not reusable at the moment. So release it (and potentially its heap
        // container) in allocator, and ARC will later free up its backing memory when the busy command buffer finishes.
        release_buffer(buffer_block, true);
      } else {
        // only if there's no memory pressure, we'll reuse the oversized buffer
        params.buffer_block = buffer_block;
      }
    }
  }

  if (!params.buffer_block) {
    return false; // this will make allocator to allocate a new buffer
  }
  pool.available_buffers.erase(params.buffer_block);
  params.buffer_block->requested_size = params.requested_size;
  params.buffer_block->gc_count = 0;
  pool.available_size -= params.buffer_block->size;

  if ((m_debug_verbosity & DebugVerbosity::RECYCLES) &&
      (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    std::cerr << "Reusing " << ((params.pool->usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((params.pool->usage & UsageFlags::SCALAR) ? " scalar" : "") << " buffer #"
              << params.buffer_block->buf_id << " of size " << format_size(params.buffer_block->size) << " at "
              << params.buffer_block->buffer << " (requested: " << format_size(params.requested_size)
              << ", use#: " << params.buffer_block->use_count + 1 << ", retain#: " << params.buffer_block->retainCount()
              << ")\n";
  }
  return true;
}

BufferBlock* MPSHeapAllocatorImpl::alloc_buffer_block(size_t size, uint32_t usage) {
  TORCH_CHECK(size < m_max_buffer_size, "Invalid buffer size: ", format_size(size));

  size_t alloc_size = get_allocation_size(size, usage);
  auto& pool = get_pool(size, alloc_size, usage);
  AllocParams params(alloc_size, size, &pool);
  // we care about memory pressure if only we're allocating large buffers when the
  // low watermark limit has been reached
  params.has_memory_pressure = !(pool.usage & UsageFlags::SMALL) && getLowWatermarkValue() <= 0;
  params.has_unified_memory = m_device.hasUnifiedMemory;

  // first, try to get a block from the existing pool.
  bool block_found = get_free_buffer(params);
  if (!block_found) {
    // do garbage collection if memory pressure is high and there's enough memory in pool
    if (params.has_memory_pressure && alloc_size < pool.available_size) {
      garbage_collect_cached_buffers(params);
    }

    block_found =
        // Attempt allocate
        alloc_buffer(params) ||
        // Callbacks might release more memory (eg. by forcing a GC in the host language) thus
        // we can retry getting a free buffer in the pool, before trying to alloc again.
        (trigger_memory_callbacks(nullptr, IMpsAllocatorCallback::EventType::ALLOCATION_FAILED) &&
         get_free_buffer(params)) ||
        // Free enough available cached blocks to satisfy alloc and retry alloc.
        (release_available_cached_buffers(params) && alloc_buffer(params)) ||
        // Free all cached buffers and retry alloc.
        (release_cached_buffers() && alloc_buffer(params));
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
          format_size(m_total_allocated_memory),
          ", other allocations: ",
          format_size(current_allocated_size() - m_total_allocated_memory),
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
                  format_size(m_total_allocated_memory),
                  ", other allocations: ",
                  format_size(current_allocated_size() - m_total_allocated_memory),
                  "). Tried to allocate ",
                  format_size(alloc_size),
                  " on ",
                  ((pool.usage & UsageFlags::SHARED) ? "shared" : "private"),
                  " pool.");
    }
  }
  buffer_block->in_use = true;
  buffer_block->use_count++;
  m_current_allocated_memory += buffer_block->size;

  return buffer_block;
}

void MPSHeapAllocatorImpl::free_buffer(BufferBlock* buffer_block) {
  TORCH_INTERNAL_ASSERT(buffer_block->in_use);

  BufferPool& pool = *buffer_block->heap->pool;
  // Makes sure the BufferBlock* isn't already present in the pool we're freeing it back into.
  TORCH_INTERNAL_ASSERT(pool.available_buffers.insert(buffer_block).second);
  pool.available_size += buffer_block->size;
  buffer_block->shape.clear(); // reset shape
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_current_allocated_memory >= buffer_block->size);
  m_current_allocated_memory -= buffer_block->size;
  if (buffer_block->event) {
    // returns the MPSEvent back to MPSEventPool
    buffer_block->event.reset(nullptr);
  }
  buffer_block->in_use = false;
}

BufferBlock* MPSHeapAllocatorImpl::get_allocated_buffer_block(const void* ptr) {
  auto it = m_allocated_buffers.find(ptr);
  if (it == m_allocated_buffers.end()) {
    return nullptr;
  }
  return it->second;
}

bool MPSHeapAllocatorImpl::release_buffer(BufferBlock* buffer_block, bool remove_empty_heap) {
  HeapBlock* heap_block = buffer_block->heap;
  BufferPool& pool = *heap_block->pool;
  pool.allocated_size -= buffer_block->size;
  pool.available_size -= buffer_block->size;
  m_allocated_buffers.erase(buffer_block->buffer);
  pool.available_buffers.erase(buffer_block);
  pool.n_buffers--;
  // will re-insert later to keep the heaps list sorted based on heap's new available size (if heap not empty)
  pool.heaps.erase(heap_block);
  uint32_t retainCount = heap_block->releaseMTLBuffer(buffer_block->buffer);

  if ((m_debug_verbosity & DebugVerbosity::RELEASES) &&
      (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    std::cerr << "Released buffer #" << buffer_block->buf_id << " of size " << format_size(buffer_block->size)
              << " from heap #" << heap_block->heap_id << " (heap size: " << format_size(heap_block->size.available)
              << ", use#: " << buffer_block->use_count << ", retain#: " << retainCount
              << ", gc#: " << buffer_block->gc_count << ")\n";
  }
  delete buffer_block;

  if (remove_empty_heap && heap_block->n_buffers == 0) {
    pool.heaps_pending_update.erase(heap_block);
    m_total_allocated_memory -= heap_block->size.total;
    retainCount = heap_block->releaseMTLHeap();
    if (m_debug_verbosity & DebugVerbosity::RELEASES) {
      std::cerr << "Released heap #" << heap_block->heap_id << " of size " << format_size(heap_block->size.total)
                << " (current allocated: " << format_size(current_allocated_size()) << ", retain#: " << retainCount
                << ")\n";
    }
    delete heap_block;
    return true;
  } else {
    pool.heaps.insert(heap_block);
    // if heap wasn't released and its released buffer is still busy in command buffer, the available
    // size of the heap cannot be updated and we should defer updating until command buffer finishes.
    if (retainCount > 1) {
      pool.heaps_pending_update.insert(heap_block);
      m_mutex.unlock();
      m_stream->addCompletedHandler(^(id<MTLCommandBuffer>) {
        std::lock_guard<std::recursive_mutex> lock(m_mutex);
        // check if the heap block still exists
        if (pool.heaps_pending_update.find(heap_block) != pool.heaps_pending_update.end()) {
          pool.heaps_pending_update.erase(heap_block);
          pool.heaps.erase(heap_block);
          heap_block->updateAvailableSize();
          pool.heaps.insert(heap_block);
        }
      });
      m_mutex.lock();
    }
  }
  return false;
}

void MPSHeapAllocatorImpl::release_buffers(BufferPool& pool) {
  if (pool.available_buffers.empty()) {
    return;
  }
  if ((m_debug_verbosity & DebugVerbosity::RELEASES)) {
    std::cerr << "Releasing " << pool.available_buffers.size() << " buffers from "
              << ((pool.usage & UsageFlags::SMALL) ? "small " : "large ")
              << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((pool.usage & UsageFlags::SCALAR) ? " scalar" : "")
              << " pool (total size: " << format_size(pool.allocated_size) << ", #buffers: " << pool.n_buffers << ")\n";
  }
  auto it = pool.available_buffers.begin();
  while (it != pool.available_buffers.end()) {
    BufferBlock* buffer_block = *it;
    ++it;
    release_buffer(buffer_block);
  }
}

bool MPSHeapAllocatorImpl::release_available_cached_buffers(AllocParams& params) {
  BufferPool& pool = *params.pool;

  if (pool.available_buffers.empty()) {
    return false;
  }
  auto it = pool.available_buffers.lower_bound(&params.search_key);
  if (it == pool.available_buffers.end()) {
    size_t totalReleased = 0;
    --it;
    while (totalReleased < params.search_key.size) {
      auto cur = it;
      totalReleased += (*it)->size;
      if (it != pool.available_buffers.begin()) {
        --it;
        release_buffer(*cur);
      } else {
        release_buffer(*cur);
        break;
      }
    }
    if (totalReleased < params.search_key.size) {
      return false;
    }
  } else {
    release_buffer(*it);
  }
  return true;
}

bool MPSHeapAllocatorImpl::release_cached_buffers() {
  if (m_debug_verbosity >= DebugVerbosity::PROFILING) {
    std::cerr << "Attempting to release cached buffers (MPS allocated: " << format_size(m_total_allocated_memory)
              << ", other allocations: " << format_size(current_allocated_size() - m_total_allocated_memory) << ")\n";
  }
  // before releasing the buffers make sure the command buffer has finished.
  // we need to release the lock temporarily as synchronizing may cause deadlock with completion handlers.
  m_mutex.unlock();
  auto stream = getDefaultMPSStream();
  dispatch_sync_with_rethrow(stream->queue(), ^() {
    stream->synchronize(SyncType::COMMIT_AND_WAIT);
  });
  m_mutex.lock();
  // Free all cached blocks to system allocator
  for (const auto& poolIt : m_pools) {
    BufferPool& pool = *poolIt.second;
    release_buffers(pool);
  }
  return true;
}

void MPSHeapAllocatorImpl::garbage_collect_cached_buffers(AllocParams& params) {
  // skip garbage collection if memory pressure has already relieved
  if (current_allocated_size() < m_low_watermark_limit) {
    return;
  }
  // attempt to collect garbage until we reach below low watermark limit
  const auto target_size = current_allocated_size() - m_low_watermark_limit;
  const BufferPool& pool = *params.pool;
  // calculate the total age of the free-able blocks. We'll use it later to get the average age threshold.
  double total_age = 0.0;
  unsigned int freeable_block_count = 0, freed_count = 0;
  size_t gc_reclaimed = 0;

  for (auto& b : pool.available_buffers) {
    if (b->retainCount() <= 1) {
      total_age += b->gc_count;
      ++freeable_block_count;
    }
  }
  if (freeable_block_count == 0) {
    return;
  }
  // repeat GC until we reach reclaim > target size.
  bool block_freed = true;
  while (gc_reclaimed < target_size && block_freed && freeable_block_count > 0) {
    // free blocks exceeding this age threshold first.
    double age_threshold = total_age / freeable_block_count;
    // stop iteration if we can no longer free a block.
    block_freed = false;
    // free blocks of > avg age. Stop garbage collection if we reach below the
    // low watermark limit since re-allocation or fragmentation could be costly.
    auto it = pool.available_buffers.begin();
    while (it != pool.available_buffers.end() && gc_reclaimed < target_size) {
      BufferBlock* buffer_block = *it++;
      if (buffer_block->gc_count >= age_threshold && buffer_block->retainCount() <= 1) {
        block_freed = true;
        gc_reclaimed += buffer_block->size;
        total_age -= buffer_block->gc_count;
        freeable_block_count--;
        freed_count++;
        release_buffer(buffer_block, !buffer_block->heap->is_split);
      }
    }
  }
  if (m_debug_verbosity & DebugVerbosity::RELEASES) {
    std::cerr << "Garbage collected " << freed_count << " buffers from large "
              << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private")
              << " pool (total reclaimed: " << format_size(gc_reclaimed)
              << ", #buffers: " << pool.available_buffers.size() << ")\n";
  }
}

// public interface to MPSAllocator
id<MTLBuffer> MPSHeapAllocatorImpl::malloc(size_t size, uint32_t usage) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  BufferBlock* buffer_block = alloc_buffer_block(size, usage);
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

    buffer_block = alloc_buffer_block(size, UsageFlags::SCALAR);
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
    const BufferPool& pool = *buffer_block->heap->pool;
    if (!(pool.usage & UsageFlags::SCALAR)) {
      free_buffer(buffer_block);
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
  explicit MPSAllocator(uint32_t Usage)
      : m_has_unified_memory(_getAllocImpl().Device().hasUnifiedMemory), m_usage(Usage) {
    if (_getAllocImpl().getDebugVerbosity()) {
      if (!(m_usage & HeapAllocator::UsageFlags::SHARED) || m_has_unified_memory) {
        std::cerr << "Initializing " << ((m_usage & HeapAllocator::UsageFlags::SHARED) ? "shared" : "private")
                  << " heap allocator on " << (m_has_unified_memory ? "unified" : "discrete")
                  << " device memory of size "
                  << _getAllocImpl().format_size(_getAllocImpl().Device().recommendedMaxWorkingSetSize) << "\n";
      }
    }
  }

  ~MPSAllocator() override {
    _getAllocImpl().emptyCache();
  }
  DeleterFnPtr raw_deleter() const override {
    return &Delete;
  }

  DataPtr allocate(const size_t nbytes) override {
    __block id<MTLBuffer> buf = nbytes > 0 ? _getAllocImpl().malloc(nbytes, m_usage) : nullptr;
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
  bool isSharedBuffer(const void* ptr) const override {
    return _getAllocImpl().isSharedBuffer(ptr);
  }
  bool isSharedStorageSupported() const override {
    return m_has_unified_memory;
  }
  void emptyCache() const override {
    _getAllocImpl().emptyCache();
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
  bool m_has_unified_memory;
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

MPSAllocator& _getPrivateAllocator() {
  static MPSAllocator s_mps_private_alloc(HeapAllocator::UsageFlags::PRIVATE);
  return s_mps_private_alloc;
}
} // anonymous namespace

IMPSAllocator* getIMPSAllocator(bool sharedAllocator) {
  if (!sharedAllocator) {
    return &_getPrivateAllocator();
  }
  auto& sa = _getSharedAllocator();
  if (sa.isSharedStorageSupported()) {
    return &sa;
  }
  return nullptr;
}

// torch.is_pinned() implementation
// Pinned memory will be helpful on Apple Silicon Macs with Unified memory as we
// will be able to use SharedStorageMode for MTLBuffer allocations. This will
// avoid extra copies on DataLoading operations.
bool isMPSPinnedPtr(const void* data) {
  return at::mps::_getSharedAllocator().isSharedBuffer(data);
}

} // namespace at::mps
