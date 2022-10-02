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

void MPSHeapAllocatorImpl::init_allocator_params()
{
  // debug verbosity flags (see DebugVerbosity enum)
  static const char *verbosity_str = getenv("PYTORCH_DEBUG_MPS_ALLOCATOR");
  m_debug_verbosity = verbosity_str ? strtol(verbosity_str, nullptr, 0) : DebugVerbosity::SILENT;

  static const char *high_watermark_ratio_str = getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO");
  m_high_watermark_ratio = high_watermark_ratio_str ? strtod(high_watermark_ratio_str, nullptr) : default_high_watermark_ratio;
  TORCH_CHECK(m_high_watermark_ratio >= 0.0 && m_high_watermark_ratio <= 1.0,
              "invalid high watermark ratio ", m_high_watermark_ratio);

  m_max_total_allowed_size = (m_high_watermark_ratio == 0.0) ? std::numeric_limits<size_t>::max() :
                              static_cast<size_t>(m_high_watermark_ratio * (double)max_device_size());
}

HeapBlock* MPSHeapAllocatorImpl::get_free_heap(AllocParams& p)
{
  BufferPool *pool = p.pool;
  HeapBlock *heapBlock = nullptr;
  HeapBlock search_key(p.size());

  auto it = pool->heaps.lower_bound(&search_key);
  if (it == pool->heaps.end()) {
    id<MTLHeap> heap = HeapBlock::createMTLHeap(pool->device, p.size(), pool->usage);
    if (heap) {
      size_t heap_size = HeapBlock::heapAvailableSize(heap);
      heapBlock = new HeapBlock(heap_size, heap, pool);

      if (m_debug_verbosity & DebugVerbosity::ALLOCATIONS) {
        static unsigned int heap_counter = 0;
        std::cerr << "\nAllocated "
                  << ((pool->usage & UsageFlags::SMALL) ? "small " : "large ")
                  << ((pool->usage & UsageFlags::SHARED) ? "shared " : "private ")
                  << "heap of size " << format_size(heap_size)
                  << " (#heaps: " << (++heap_counter)
                  << ", current allocated: " << format_size(current_allocated_size()) << ")\n";
      }
    }
  } else {
    heapBlock = *it;
    // remove and re-insert heap in the set later after a buffer is created.
    // this ensures updating the order of heaps based on their new available sizes
    pool->heaps.erase(it);
  }
  return heapBlock;
}

bool MPSHeapAllocatorImpl::alloc_buffer(AllocParams& p)
{
  if (m_max_total_allowed_size != std::numeric_limits<size_t>::max() &&
      current_allocated_size() + p.size() > m_max_total_allowed_size)
    return false;

  HeapBlock *heap = get_free_heap(p);
  if (!heap)
    return false; // this will cause releasing pool buffers to free up memory

  BufferPool& pool = *p.pool;

  id<MTLBuffer> buffer = heap->newMTLBuffer(p.size(), pool.usage);
  // this should never happen as the backing memory (i.e., heap) was allocated successfully.
  TORCH_INTERNAL_ASSERT(buffer);
  // insert heap after a buffer was created on it to update the order of heap's set
  pool.heaps.insert(heap);
  p.buffer_block = new BufferBlock(p.size(), p.requested_size, buffer, heap, m_allocated_buffers.size() + 1);
  m_allocated_buffers[p.buffer_block->buffer] = p.buffer_block;
  m_total_allocated_memory += p.size();
  pool.allocated_size += p.size();
  pool.n_buffers++;

  if (m_debug_verbosity & DebugVerbosity::ALLOCATIONS) {
    std::cerr << "Allocated "
              << ((p.pool->usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((p.pool->usage & UsageFlags::SCALAR) ? " scalar" : "")
              << " buffer #" << p.buffer_block->buf_id
              << " of size " << format_size(p.size())
              << " at " << p.buffer_block->buffer
              << " (requested: " << format_size(p.requested_size)
              << ", heap: " << format_size(heap->size.available)
              << ", total: " << format_size(m_total_allocated_memory)
              << ", pool: " << format_size(pool.allocated_size) << ")\n";
  }
  return true;
}

bool MPSHeapAllocatorImpl::get_free_buffer(AllocParams& p)
{
  // this helps to monitor "implicit" allocations from MPS backend and to prevent OOM and system failure.
  if (m_high_watermark_ratio > 0.0 && current_allocated_size() + p.size() > m_max_total_allowed_size)
    return false;

  BufferPool& pool = *p.pool;
  auto it = pool.buffers.lower_bound(&p.search_key);
  if (it != pool.buffers.end()) {
    p.buffer_block = *it;
    pool.buffers.erase(it);
  }

  if (!p.buffer_block)
    return false; // this will make allocator to allocate a new buffer

  if (m_debug_verbosity & DebugVerbosity::RECYCLES) {
    std::cerr << "Reusing "
              << ((p.pool->usage & UsageFlags::SHARED) ? "shared" : "private")
              << ((p.pool->usage & UsageFlags::SCALAR) ? " scalar" : "")
              << " buffer #" << p.buffer_block->buf_id
              << " of size " << format_size(p.buffer_block->size)
              << " at " << p.buffer_block->buffer
              << " (requested: " << format_size(p.requested_size) << ")\n";
  }
  return true;
}

BufferBlock* MPSHeapAllocatorImpl::alloc_buffer_block(size_t size, uint32_t usage)
{
  TORCH_CHECK(size < m_max_buffer_size, "Invalid buffer size: ", format_size(size));

  size_t alloc_size = get_allocation_size(size, usage);
  auto& pool = get_pool(alloc_size, usage);
  AllocParams params(alloc_size, size, &pool);

  bool block_found =
      // Search pool
      get_free_buffer(params) ||
      // Attempt allocate
      alloc_buffer(params) ||
      // Free enough available cached blocks to satisfy alloc and retry alloc.
      (release_available_cached_buffers(params) && alloc_buffer(params)) ||
      // Free all cached buffers and retry alloc.
      (release_cached_buffers() && alloc_buffer(params));

  BufferBlock* buffer_block = params.buffer_block;

  TORCH_CHECK(block_found && buffer_block, "MPS backend out of memory (currently allocated: ",
              format_size(current_allocated_size()), ", max allowed: ", format_size(m_max_total_allowed_size),
              "). Tried to allocate ", format_size(alloc_size), " on ", ((pool.usage & UsageFlags::SHARED) ? "shared" : "private"),
              " pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).");

  buffer_block->in_use = true;

  return buffer_block;
}

void MPSHeapAllocatorImpl::free_buffer(BufferBlock* buffer_block)
{
  TORCH_INTERNAL_ASSERT(buffer_block->in_use);

  BufferPool& pool = *buffer_block->heap->pool;
  // Makes sure the BufferBlock* isn't already present in the pool we're freeing it back into.
  TORCH_INTERNAL_ASSERT(pool.buffers.insert(buffer_block).second);
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

void MPSHeapAllocatorImpl::release_buffer(BufferBlock* buffer_block, bool remove_empty_heap)
{
  trigger_memory_callbacks(buffer_block, IMpsAllocatorCallback::EventType::RELEASED);

  HeapBlock *heap = buffer_block->heap;
  BufferPool *pool = heap->pool;
  m_total_allocated_memory -= buffer_block->size;
  pool->allocated_size -= buffer_block->size;
  m_allocated_buffers.erase(buffer_block->buffer);
  pool->buffers.erase(buffer_block);
  pool->n_buffers--;
  // will re-insert later to keep the heaps list sorted based on heap's new available size (if heap not empty)
  pool->heaps.erase(heap);
  heap->releaseMTLBuffer(buffer_block->buffer);

  if (m_debug_verbosity & DebugVerbosity::RELEASES) {
    std::cerr << "Released buffer #" << buffer_block->buf_id
              << " of size " << format_size(buffer_block->size)
              << " (heap size: " << format_size(heap->size.available)
              << ", total allocated: " << format_size(m_total_allocated_memory) << ")\n";

  }
  delete buffer_block;

  if (remove_empty_heap && heap->n_buffers == 0) {
    heap->releaseMTLHeap();
    if (m_debug_verbosity & DebugVerbosity::RELEASES) {
      std::cerr << "Released heap of size " << format_size(heap->size.total)
                << " (current allocated: " << format_size(current_allocated_size()) << ")\n";
    }
    delete heap;
  } else {
    pool->heaps.insert(heap);
  }
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

bool MPSHeapAllocatorImpl::release_available_cached_buffers(AllocParams& p)
{
  BufferPool& pool = *p.pool;

  if (pool.buffers.empty())
    return false;

  auto it = pool.buffers.lower_bound(&p.search_key);
  if (it == pool.buffers.end()) {
    size_t totalReleased = 0;
    --it;
    while (totalReleased < p.search_key.size) {
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
    if (totalReleased < p.search_key.size)
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
              << ", other allocations: " << format_size(current_allocated_size()-m_total_allocated_memory) << ")\n";
  }
  // Free all cached blocks to system allocator
  release_buffers(m_large_pool_private);
  release_buffers(m_large_pool_shared);
  release_buffers(m_small_pool_private);
  release_buffers(m_small_pool_shared);
  release_buffers(m_scalar_pool);
  return true;
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
  // before releasing the buffers make sure the command buffer has finished.
  m_stream->synchronize(SyncType::COMMIT_AND_WAIT);

  std::lock_guard<std::mutex> lock(m_mutex);
  release_cached_buffers();
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
        std::cerr << "Initializing "
                  << ((m_usage & HeapAllocator::UsageFlags::SHARED) ? "shared" : "private")
                  << " heap allocator on "
                  << (m_has_unified_memory ? "unified" : "discrete")
                  << " device memory of size "
                  << _getAllocImpl().Device().recommendedMaxWorkingSetSize / 1048576UL << " MB"
                  << " (max allowed: "
                  << (max_total_allowed_size == std::numeric_limits<size_t>::max() ? "unlimited" :
                     (to_string(max_total_allowed_size / 1048576UL) + " MB")) << ")\n";
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
