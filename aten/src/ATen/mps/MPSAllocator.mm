//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <ATen/CPUFunctions.h>
#include <iostream>

namespace at {
namespace mps {

C10_DEFINE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);

namespace HeapAllocator {

uint64_t BufferBlock::buffer_counter = 0;
uint64_t HeapBlock::heap_counter = 0;

void MPSHeapAllocatorImpl::init_allocator()
{
  // debug verbosity flags (see DebugVerbosity enum)
  static const char *verbosity_str = getenv("PYTORCH_DEBUG_MPS_ALLOCATOR");
  m_debug_verbosity = verbosity_str ? strtol(verbosity_str, nullptr, 0) : DebugVerbosity::SILENT;

  // on unified memory, we set the allowed upper bound to twice the size of recommendedMaxWorkingSetSize.
  const double high_watermark_upper_bound =  m_device.hasUnifiedMemory ? 2.0 : 1.0;

  static const char *high_watermark_ratio_str = getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO");
  m_high_watermark_ratio = high_watermark_ratio_str ? strtod(high_watermark_ratio_str, nullptr) : default_high_watermark_ratio;
  TORCH_CHECK(m_high_watermark_ratio >= 0.0 && m_high_watermark_ratio <= high_watermark_upper_bound,
              "invalid high watermark ratio ", m_high_watermark_ratio);

  m_max_total_allowed_size = (m_high_watermark_ratio == 0.0) ? std::numeric_limits<size_t>::max() :
                              static_cast<size_t>(m_high_watermark_ratio * (double)max_device_size());
  // used for comparison with lower_watermark_ratio
  const double high_watermark_limit = m_high_watermark_ratio == 0.0 ? high_watermark_upper_bound : m_high_watermark_ratio;
  const double default_low_watermark_ratio =  m_device.hasUnifiedMemory ? default_low_watermark_ratio_unified :
                                                                          default_low_watermark_ratio_discrete;
  static const char *low_watermark_ratio_str = getenv("PYTORCH_MPS_LOW_WATERMARK_RATIO");
  m_low_watermark_ratio = low_watermark_ratio_str ? strtod(low_watermark_ratio_str, nullptr) : default_low_watermark_ratio;
  TORCH_CHECK(m_low_watermark_ratio >= 0.0 && m_low_watermark_ratio <= high_watermark_limit,
              "invalid low watermark ratio ", m_low_watermark_ratio);
  // we use this to detect if there's memory pressure
  m_low_watermark_limit = (m_low_watermark_ratio == 0.0) ? std::numeric_limits<size_t>::max() :
                          static_cast<size_t>(m_low_watermark_ratio * (double)max_device_size());
}

HeapBlock* MPSHeapAllocatorImpl::get_free_heap(AllocParams& params)
{
  BufferPool& pool = *params.pool;
  HeapBlock *heap_block = nullptr;
  HeapBlock search_key(params.size());

  auto it = pool.heaps.lower_bound(&search_key);
  if (it == pool.heaps.end()) {
    heap_block = HeapBlock::createHeapBlock(params, pool.device, pool.usage);
    if (heap_block) {
      if (m_debug_verbosity & DebugVerbosity::ALLOCATIONS) {
        std::cerr << "\nAllocated "
                  << ((pool.usage & UsageFlags::SHARED) ? "shared " : "private ")
                  << " heap #" << heap_block->heap_id
                  << " of size " << format_size(heap_block->size.total)
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

bool MPSHeapAllocatorImpl::alloc_buffer(AllocParams& params)
{
  if (m_max_total_allowed_size != std::numeric_limits<size_t>::max() &&
      current_allocated_size() + params.size() > m_max_total_allowed_size)
    return false;

  HeapBlock *heap = get_free_heap(params);
  if (!heap)
    return false; // this will cause releasing pool buffers to free up memory

  BufferPool& pool = *params.pool;

  id<MTLBuffer> buffer = heap->newMTLBuffer(params.size(), pool.usage);
  // this should never happen as the backing memory (i.e., heap) was allocated successfully.
  TORCH_INTERNAL_ASSERT(buffer);
  // insert heap after a buffer was created on it to update the order of heap's set
  pool.heaps.insert(heap);
  params.buffer_block = new BufferBlock(params.size(), params.requested_size, buffer, heap);
  m_allocated_buffers[params.buffer_block->buffer] = params.buffer_block;
  m_total_allocated_memory += params.size();
  pool.allocated_size += params.size();
  pool.n_buffers++;

  if ((m_debug_verbosity & DebugVerbosity::ALLOCATIONS) &&
    (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    std::cerr << "Allocated "
              << ((params.pool->usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((params.pool->usage & UsageFlags::SCALAR) ? " scalar" : "")
              << " buffer #" << params.buffer_block->buf_id
              << " of size " << format_size(params.size())
              << " at " << params.buffer_block->buffer
              << " from heap #" << heap->heap_id
              << " (requested: " << format_size(params.requested_size)
              << ", heap: " << format_size(heap->size.available)
              << ", total: " << format_size(m_total_allocated_memory) << ")\n";
  }
  return true;
}

bool MPSHeapAllocatorImpl::get_free_buffer(AllocParams& params)
{
  // this helps to monitor "implicit" allocations from MPS backend and to prevent OOM and system failure.
  if (m_high_watermark_ratio > 0.0 && current_allocated_size() + params.size() > m_max_total_allowed_size)
    return false;

  BufferPool& pool = *params.pool;
  // track buffer reuse intervals only on large pool when low watermark limit is enabled.
  if (m_low_watermark_ratio > 0.0 && !(pool.usage & UsageFlags::SMALL)) {
    for (auto& b : pool.buffers) {
      ++b->gc_count;
    }
  }
  auto it = pool.buffers.lower_bound(&params.search_key);
  if (it != pool.buffers.end()) {
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
        // the oversized buffer is busy and not reusable at the moment. So release it (and potentially its heap container)
        // in allocator, and ARC will later free up its backing memory when the busy command buffer finishes.
        release_buffer(buffer_block, true);
      } else {
        // only if there's no memory pressure, we'll reuse the oversized buffer
        params.buffer_block = buffer_block;
      }
    }
  }

  if (!params.buffer_block)
    return false; // this will make allocator to allocate a new buffer

  pool.buffers.erase(params.buffer_block);
  params.buffer_block->gc_count = 0;
  pool.available_size -= params.buffer_block->size;

  if ((m_debug_verbosity & DebugVerbosity::RECYCLES) &&
    (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    std::cerr << "Reusing "
              << ((params.pool->usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((params.pool->usage & UsageFlags::SCALAR) ? " scalar" : "")
              << " buffer #" << params.buffer_block->buf_id
              << " of size " << format_size(params.buffer_block->size)
              << " at " << params.buffer_block->buffer
              << " (requested: " << format_size(params.requested_size)
              << ", use#: " << params.buffer_block->use_count + 1
              << ", retain#: " << params.buffer_block->retainCount() << ")\n";
  }
  return true;
}

BufferBlock* MPSHeapAllocatorImpl::alloc_buffer_block(size_t size, uint32_t usage)
{
  TORCH_CHECK(size < m_max_buffer_size, "Invalid buffer size: ", format_size(size));

  size_t alloc_size = get_allocation_size(size, usage);
  auto& pool = get_pool(alloc_size, usage);
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
      TORCH_CHECK(false, "MPS backend out of memory (MPS allocated: ", format_size(m_total_allocated_memory),
                  ", other allocations: ", format_size(current_allocated_size() - m_total_allocated_memory),
                  ", max allowed: ", format_size(m_max_total_allowed_size), "). Tried to allocate ", format_size(alloc_size),
                  " on ", ((pool.usage & UsageFlags::SHARED) ? "shared" : "private"),
                  " pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).");
    } else {
      TORCH_CHECK(false, "MPS backend out of memory (MPS allocated: ", format_size(m_total_allocated_memory),
                  ", other allocations: ", format_size(current_allocated_size() - m_total_allocated_memory),
                  "). Tried to allocate ", format_size(alloc_size),
                  " on ", ((pool.usage & UsageFlags::SHARED) ? "shared" : "private"), " pool.");
    }
  }
  buffer_block->in_use = true;
  buffer_block->use_count++;

  return buffer_block;
}

void MPSHeapAllocatorImpl::free_buffer(BufferBlock* buffer_block)
{
  TORCH_INTERNAL_ASSERT(buffer_block->in_use);

  BufferPool& pool = *buffer_block->heap->pool;
  // Makes sure the BufferBlock* isn't already present in the pool we're freeing it back into.
  TORCH_INTERNAL_ASSERT(pool.buffers.insert(buffer_block).second);
  pool.available_size += buffer_block->size;
  buffer_block->shape.clear(); // reset shape
  buffer_block->in_use = false;
}

BufferBlock* MPSHeapAllocatorImpl::get_allocated_buffer_block(void* ptr)
{
  auto it = m_allocated_buffers.find(ptr);
  if (it == m_allocated_buffers.end())
    return nullptr;

  return it->second;
}

bool MPSHeapAllocatorImpl::release_buffer(BufferBlock* buffer_block, bool remove_empty_heap)
{
  HeapBlock *heap_block = buffer_block->heap;
  BufferPool& pool = *heap_block->pool;
  m_total_allocated_memory -= buffer_block->size;
  pool.allocated_size -= buffer_block->size;
  pool.available_size -= buffer_block->size;
  m_allocated_buffers.erase(buffer_block->buffer);
  pool.buffers.erase(buffer_block);
  pool.n_buffers--;
  // will re-insert later to keep the heaps list sorted based on heap's new available size (if heap not empty)
  pool.heaps.erase(heap_block);
  uint32_t retainCount = heap_block->releaseMTLBuffer(buffer_block->buffer);

  if ((m_debug_verbosity & DebugVerbosity::RELEASES) &&
    (!(m_debug_verbosity & DebugVerbosity::LARGE_ONLY) || !(pool.usage & UsageFlags::SMALL))) {
    std::cerr << "Released buffer #" << buffer_block->buf_id
              << " of size " << format_size(buffer_block->size)
              << " from heap #" << heap_block->heap_id
              << " (heap size: " << format_size(heap_block->size.available)
              << ", use#: " << buffer_block->use_count
              << ", retain#: " << retainCount
              << ", gc#: " << buffer_block->gc_count << ")\n";
  }
  delete buffer_block;

  if (remove_empty_heap && heap_block->n_buffers == 0) {
    pool.heaps_pending_update.erase(heap_block);
    retainCount = heap_block->releaseMTLHeap();
    if (m_debug_verbosity & DebugVerbosity::RELEASES) {
      std::cerr << "Released heap #" << heap_block->heap_id
                << " of size " << format_size(heap_block->size.total)
                << " (current allocated: " << format_size(current_allocated_size())
                << ", retain#: " << retainCount << ")\n";
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
      m_stream->addCompletedHandler(^(id <MTLCommandBuffer>) {
        std::lock_guard<std::mutex> lock(m_mutex);
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

void MPSHeapAllocatorImpl::release_buffers(BufferPool& pool)
{
  if ((m_debug_verbosity & DebugVerbosity::PROFILING) && pool.n_buffers > 0) {
    std::cerr << "Releasing " << pool.n_buffers
              << " buffers from "
              << ((pool.usage & UsageFlags::SMALL ) ? "small " : "large ")
              << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((pool.usage & UsageFlags::SCALAR) ? " scalar" : "")
              << " pool (total size: " << format_size(pool.allocated_size)
              << ", free buffers: " << pool.buffers.size() << ")\n";
  }
  auto it = pool.buffers.begin();
  while (it != pool.buffers.end()) {
    BufferBlock* buffer_block = *it;
    ++it;
    release_buffer(buffer_block);
  }
}

bool MPSHeapAllocatorImpl::release_available_cached_buffers(AllocParams& params)
{
  BufferPool& pool = *params.pool;

  if (pool.buffers.empty())
    return false;

  auto it = pool.buffers.lower_bound(&params.search_key);
  if (it == pool.buffers.end()) {
    size_t totalReleased = 0;
    --it;
    while (totalReleased < params.search_key.size) {
      auto cur = it;
      totalReleased += (*it)->size;
      if (it != pool.buffers.begin()) {
        --it;
        release_buffer(*cur);
      } else {
        release_buffer(*cur);
        break;
      }
    }
    if (totalReleased < params.search_key.size)
      return false;
  } else {
    release_buffer(*it);
  }
  return true;
}

bool MPSHeapAllocatorImpl::release_cached_buffers()
{
  if (m_debug_verbosity >= DebugVerbosity::PROFILING) {
    std::cerr << "Releasing buffer pools (MPS allocated: " << format_size(m_total_allocated_memory)
              << ", other allocations: " << format_size(current_allocated_size() - m_total_allocated_memory) << ")\n";
  }
  // before releasing the buffers make sure the command buffer has finished.
  // we need to release the lock temporarily as synchronizing may cause deadlock with completion handlers.
  m_mutex.unlock();
  m_stream->synchronize(SyncType::COMMIT_AND_WAIT);
  m_mutex.lock();
  // Free all cached blocks to system allocator
  release_buffers(m_large_pool_private);
  release_buffers(m_large_pool_shared);
  release_buffers(m_small_pool_private);
  release_buffers(m_small_pool_shared);
  release_buffers(m_scalar_pool);
  return true;
}

void MPSHeapAllocatorImpl::garbage_collect_cached_buffers(AllocParams& params)
{
  TORCH_INTERNAL_ASSERT(current_allocated_size() >= m_low_watermark_limit);
  // attempt to collect garbage until we reach below low watermark limit
  const auto target_size = current_allocated_size() - m_low_watermark_limit;
  const BufferPool& pool = *params.pool;
  // calculate the total age of the free-able blocks. We'll use it later to get the average age threshold.
  double total_age = 0.0;
  unsigned int freeable_block_count = 0, freed_count = 0;
  size_t gc_reclaimed = 0;

  for (auto& b : pool.buffers) {
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
    auto it = pool.buffers.begin();
    while (it != pool.buffers.end() && gc_reclaimed < target_size) {
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
    std::cerr << "Garbage collected " << freed_count
              << " buffers from large "
              << ((pool.usage & UsageFlags::SHARED) ? "shared" : "private")
              << " pool (total reclaimed: " << format_size(gc_reclaimed)
              << ", #buffers: " << pool.buffers.size() << ")\n";
  }
}

// public interface to MPSAllocator
id<MTLBuffer> MPSHeapAllocatorImpl::malloc(size_t size, uint32_t usage)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock* buffer_block = alloc_buffer_block(size, usage);
  return buffer_block ? buffer_block->buffer : nullptr;
}

bool MPSHeapAllocatorImpl::isSharedBuffer(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock *buffer_block = get_allocated_buffer_block(ptr);
  // it's OK for the buffer_block to not exist yet
  return buffer_block && (buffer_block->heap->pool->usage & UsageFlags::SHARED);
}

id<MTLBuffer> MPSHeapAllocatorImpl::allocScalarBufferWithValue(void* value, size_t size)
{
  BufferBlock* buffer_block = nullptr;
  {
    std::lock_guard<std::mutex> lock(m_mutex);

    buffer_block = alloc_buffer_block(size, UsageFlags::SCALAR);
    if (!buffer_block)
      return nullptr;
  }
  // buffer is out of the pool, so no mutex lock is needed
  memcpy([buffer_block->buffer contents], value, size);
  return buffer_block->buffer;
}

ssize_t MPSHeapAllocatorImpl::getRequestedBufferSize(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock *buffer_block = get_allocated_buffer_block(ptr);
  if (buffer_block)
    return (ssize_t) buffer_block->requested_size;
  // -1 indicates the passed buffer pointer wasn't found
  return -1;
}

void MPSHeapAllocatorImpl::setBufferShape(void* ptr, const IntArrayRef& shape)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock *buffer_block = get_allocated_buffer_block(ptr);
  TORCH_INTERNAL_ASSERT(buffer_block, "failed to find the buffer ", ptr);
  // note that the IntArrayRef doesn't own the underlying data, and the backing
  // memory for shape data must persist as long as the buffer is in use.
  // So we need to copy to vector.
  buffer_block->shape = shape.vec();
}

IntArrayRef MPSHeapAllocatorImpl::getBufferShape(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock *buffer_block = get_allocated_buffer_block(ptr);
  if (buffer_block && buffer_block->shape.size() > 0)
    return IntArrayRef{buffer_block->shape};

  return IntArrayRef();
}

void MPSHeapAllocatorImpl::free(void* ptr)
{
  BufferBlock *buffer_block = nullptr;
  {
    std::lock_guard<std::mutex> lock(m_mutex);

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
  m_stream->addCompletedHandler(^(id <MTLCommandBuffer>) {
    std::lock_guard<std::mutex> lock(m_mutex);
    free_buffer(buffer_block);
  });
}

void MPSHeapAllocatorImpl::emptyCache()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  release_cached_buffers();
}

ssize_t MPSHeapAllocatorImpl::getLowWatermarkValue()
{
  // check if low watermark limit is disabled
  if (m_low_watermark_ratio == 0.0)
    return std::numeric_limits<ssize_t>::max();
  // current_allocated_size could exceed m_low_watermark_limit (e.g., when swapping to disk)
  return std::max<ssize_t>(0, (ssize_t)(m_low_watermark_limit - current_allocated_size()) / 1048576L);
}

} // namespace HeapAllocator

// Use "at::mps::GetMPSAllocator()" to acquire a handle to MPS Allocator
namespace {
HeapAllocator::MPSHeapAllocatorImpl& _getAllocImpl() {
  static HeapAllocator::MPSHeapAllocatorImpl s_allocatorImpl;
  return s_allocatorImpl;
}
}

// MPS allocator struct to be registered with Pytorch
struct TORCH_API MPSAllocator final : public at::Allocator {
public:
  explicit MPSAllocator(uint32_t Usage) :
      m_has_unified_memory(_getAllocImpl().Device().hasUnifiedMemory), m_usage(Usage)
  {
    if (_getAllocImpl().getDebugVerbosity()) {
      if (!(m_usage & HeapAllocator::UsageFlags::SHARED) || m_has_unified_memory) {
        const size_t max_total_allowed_size = _getAllocImpl().getMaxTotalAllowedSize();
        const size_t low_watermark_limit = _getAllocImpl().getLowWatermarkLimit();
        std::cerr << "Initializing "
                  << ((m_usage & HeapAllocator::UsageFlags::SHARED) ? "shared" : "private")
                  << " heap allocator on "
                  << (m_has_unified_memory ? "unified" : "discrete")
                  << " device memory of size "
                  << _getAllocImpl().Device().recommendedMaxWorkingSetSize / 1048576UL << " MB"
                  << " (max allowed: "
                  << (max_total_allowed_size == std::numeric_limits<size_t>::max() ? "unlimited" :
                     (to_string(max_total_allowed_size / 1048576UL) + " MB"))
                  << ", low watermark: "
                  << (low_watermark_limit == std::numeric_limits<size_t>::max() ? "unlimited" :
                     (to_string(low_watermark_limit / 1048576UL) + " MB"))  << ")\n";
      }
    }
  }

  ~MPSAllocator() override {
    _getAllocImpl().emptyCache();
  }

  DataPtr allocate(const size_t nbytes) const override {
    __block id<MTLBuffer> buf = nbytes > 0 ? _getAllocImpl().malloc(nbytes, m_usage) : nullptr;
    return { buf, buf, &Delete, at::Device(at::DeviceType::MPS, 0)};
  }

  DataPtr allocate_scalar_buffer(void *value, size_t size) const {
    id<MTLBuffer> buf = _getAllocImpl().allocScalarBufferWithValue(value, size);
    return { buf, buf, &Delete, at::Device(at::DeviceType::MPS, 0)};
  }

  DeleterFnPtr raw_deleter() const override { return &Delete; }
  bool is_shared(void* ptr) const { return _getAllocImpl().isSharedBuffer(ptr); }
  bool is_shared_storage_supported() const { return m_has_unified_memory; }

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

at::Allocator* getMPSSharedAllocator()
{
  auto& sa = _getSharedAllocator();
  if (sa.is_shared_storage_supported()) {
    return &sa;
  }

  return nullptr;
}

at::Allocator* getMPSPrivateAllocator() {
  return &_getPrivateAllocator();
}

// TODO: create MPSHooks interface and move these there.
ssize_t get_requested_buffer_size(void* ptr) {
  return _getAllocImpl().getRequestedBufferSize(ptr);
}

void set_buffer_shape(void* ptr, const IntArrayRef& shape) {
  _getAllocImpl().setBufferShape(ptr, shape);
}

IntArrayRef get_buffer_shape(void* ptr) {
  return _getAllocImpl().getBufferShape(ptr);
}

DataPtr allocate_scalar_buffer(void *value, size_t size) {
  return _getPrivateAllocator().allocate_scalar_buffer(value, size);
}

uint32_t get_adaptive_commit_threshold() {
  return _getAllocImpl().getLowWatermarkValue();
}

} // namespace mps

namespace native {

// torch.is_pinned() implementation
// Pinned memory will be helpful on Apple Silicon Macs with Unified memory as we
// will be able to use SharedStorageMode for MTLBuffer allocations. This will
// avoid extra copies on DataLoading operations.
bool is_pinned_mps(const Tensor& self, c10::optional<Device> device)
{
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_mps());
  return at::mps::_getSharedAllocator().is_shared(self.storage().data());
}

// torch.pin_memory() implementation
Tensor _pin_memory_mps(const Tensor& self, c10::optional<Device> device)
{
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!device.has_value() || device->is_mps());
  auto* shared_allocator = at::mps::getMPSSharedAllocator();
  TORCH_CHECK(shared_allocator, "unable to pin memory on a non-unified memory device");

  const size_t storage_size = detail::computeStorageNbytes(self.sizes(), self.strides(), self.dtype().itemsize());
  std::cout << "Pinning memory of size " << storage_size / 1024UL << " KB\n";
  auto storage = Storage(Storage::use_byte_size_t(), storage_size, shared_allocator, false);
  auto tensor = at::cpu::empty({0}, self.options()).set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace native
} // namespace at
