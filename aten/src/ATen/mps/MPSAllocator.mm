//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocator.h>
#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>
#include <ATen/CPUFunctions.h>

namespace at {
namespace mps {

C10_DEFINE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);

namespace HeapAllocator {

HeapBlock* MPSHeapAllocatorImpl::get_free_heap(AllocParams& p)
{
  BufferPool *pool = p.pool;
  HeapBlock *heapBlock = nullptr;
  HeapBlock search_key(p.size());

  auto it = pool->heaps.lower_bound(&search_key);
  if (it == pool->heaps.end()) {
    id<MTLHeap> heap = HeapBlock::createMTLHeap(pool->device, p.size(), pool->is_shared);
    if (heap) {
      size_t heap_size = HeapBlock::heapAvailableSize(heap);
      heapBlock = new HeapBlock(heap_size, heap, pool);

      if (debug_info_enabled()) {
        static unsigned int heap_counter = 0;
        std::cerr << "\nAllocated "
                  << (pool->is_small ? "small " : "large ")
                  << (pool->is_shared ? "shared " : "private ")
                  << "heap of size " << format_size(heap_size)
                  << " (#heaps: " << (++heap_counter)
                  << ", free memory: " << format_size(max_available_size()) << ")\n";
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
  if (m_set_fraction && m_total_allocated_memory + p.size() > max_available_size())
    return false;

  HeapBlock *heap = get_free_heap(p);
  if (!heap)
    return false; // this will cause releasing pool buffers to free up memory

  id<MTLBuffer> buffer = heap->newMTLBuffer(p.size(), p.pool->is_shared);
  // this should never happen as the backing memory (i.e., heap) was allocated successfully.
  TORCH_INTERNAL_ASSERT(buffer);
  // insert heap after a buffer was created on it to update the order of heap's set
  p.pool->heaps.insert(heap);
  p.buffer_block = new BufferBlock(p.size(), p.requested_size, buffer, heap, m_allocated_buffers.size() + 1);
  m_allocated_buffers[p.buffer_block->buffer] = p.buffer_block;
  m_total_allocated_memory += p.size();

  if (debug_info_enabled()) {
    std::cerr << "Allocated "
              << (p.pool->is_shared ? "shared" : "private")
              << " buffer #" << p.buffer_block->buf_id
              << " of size " << format_size(p.size())
              << " at " << p.buffer_block->buffer
              << " (requested size: " << format_size(p.requested_size)
              << ", heap size: " << format_size(heap->size.available)
              << ", total allocated: " << format_size(m_total_allocated_memory) << ")\n";
  }
  return true;
}

bool MPSHeapAllocatorImpl::get_free_buffer(AllocParams& p)
{
  BufferPool& pool = *p.pool;
  auto it = pool.buffers.lower_bound(&p.search_key);
  if (it == pool.buffers.end())
    return false;
  // do not return an oversized buffer for a large request
  // allow oversized buffer size to be rounded up but within a limit
  if ((p.size() < max_split_size() && (*it)->size >= max_split_size()) ||
     ((p.size() >= max_split_size()) && ((*it)->size >= p.size() + kLargeHeap)))
    return false;

  p.buffer_block = *it;
  pool.buffers.erase(it);
  if (debug_info_enabled()) {
    std::cerr << "Reusing "
              << (p.pool->is_shared ? "shared" : "private")
              << " buffer #" << p.buffer_block->buf_id
              << " of size " << format_size(p.buffer_block->size)
              << " at " << p.buffer_block->buffer
              << " (requested size: " << format_size(p.requested_size) << ")\n";
  }
  return true;
}

id<MTLBuffer> MPSHeapAllocatorImpl::Malloc(size_t size, bool sharedStorage)
{
  TORCH_CHECK(size < m_max_buffer_size, "Invalid buffer size: ", format_size(size));

  std::lock_guard<std::mutex> lock(m_mutex);

  size_t alloc_size = get_allocation_size(size, sharedStorage);
  auto& pool = get_pool(alloc_size, sharedStorage);
  AllocParams params(alloc_size, size, &pool);

  bool block_found =
      // Search pool
      get_free_buffer(params) ||
      // Attempt allocate
      alloc_buffer(params) ||
      // Free enough available cached blocks to satisfy alloc and retry alloc.
      (release_available_cached_buffers(params) && alloc_buffer(params)) ||
      // Free all non-split cached buffers and retry alloc.
      (release_cached_buffers() && alloc_buffer(params));

  BufferBlock* buffer_block = params.buffer_block;
  TORCH_INTERNAL_ASSERT(block_found && buffer_block);
  buffer_block->in_use = true;
  return buffer_block->buffer;
}

void MPSHeapAllocatorImpl::free_buffer(BufferBlock* buffer_block)
{
  TORCH_INTERNAL_ASSERT(buffer_block->in_use);
  trigger_memory_callbacks(buffer_block, IMpsAllocatorCallback::EventType::FREED);
  buffer_block->in_use = false;
  buffer_block->shape.clear(); // reset shape
  BufferPool *pool = buffer_block->heap->pool;
  // Makes sure the BufferBlock* isn't already present in the pool we're freeing it back into.
  TORCH_INTERNAL_ASSERT(pool->buffers.insert(buffer_block).second);
}

BufferBlock* MPSHeapAllocatorImpl::get_allocated_buffer_block(void* ptr)
{
  auto it = m_allocated_buffers.find(ptr);
  if (it == m_allocated_buffers.end())
    return nullptr;

  return it->second;
}

void MPSHeapAllocatorImpl::trigger_memory_callbacks(BufferBlock* buffer_block, IMpsAllocatorCallback::EventType event) {
  for (const auto& name : MPSAllocatorCallbacksRegistry()->Keys()) {
    MPSAllocatorCallbacksRegistry()->Create(name)->executeMPSAllocatorCallback(buffer_block->buffer, event);
  }
}

bool MPSHeapAllocatorImpl::isSharedBuffer(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock *buffer_block = get_allocated_buffer_block(ptr);
  // it's OK for the buffer_block to not exist yet
  return buffer_block && buffer_block->heap->pool->is_shared;
}

ssize_t MPSHeapAllocatorImpl::getRequestedBufferSize(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock *buffer_block = get_allocated_buffer_block(ptr);
  if (buffer_block)
    return (ssize_t) buffer_block->requested_size;
  // this indicates the passed buffer pointer wasn't found
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

void MPSHeapAllocatorImpl::Free(void* ptr)
{
  std::lock_guard<std::mutex> lock(m_mutex);

  BufferBlock *buffer_block = get_allocated_buffer_block(ptr);
  TORCH_INTERNAL_ASSERT(buffer_block);
  free_buffer(buffer_block);
}

void MPSHeapAllocatorImpl::EmptyCache()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  release_cached_buffers();
}

void MPSHeapAllocatorImpl::release_buffer(BufferBlock* buffer_block, bool remove_empty_heap)
{
  trigger_memory_callbacks(buffer_block, IMpsAllocatorCallback::EventType::RELEASED);

  HeapBlock *heap = buffer_block->heap;
  BufferPool *pool = heap->pool;
  m_total_allocated_memory -= buffer_block->size;
  m_allocated_buffers.erase(buffer_block->buffer);
  pool->buffers.erase(buffer_block);
  // will re-insert later to keep the heaps list sorted based on heap's new available size (if heap not empty)
  pool->heaps.erase(heap);
  heap->releaseMTLBuffer(buffer_block->buffer);
  if (debug_info_enabled()) {
    std::cerr << "Released buffer #" << buffer_block->buf_id
              << " of size " << format_size(buffer_block->size)
              << " (heap size: " << format_size(heap->size.available)
              << ", total allocated: " << format_size(m_total_allocated_memory) << ")\n";

  }
  delete buffer_block;

  if (remove_empty_heap && heap->n_buffers == 0) {
    heap->releaseMTLHeap();
    if (debug_info_enabled()) {
      std::cerr << "Released heap of size " << format_size(heap->size.total)
                << " (free memory: " << format_size(max_available_size()) << ")\n";
    }
    delete heap;
  } else {
    pool->heaps.insert(heap);
  }
}

void MPSHeapAllocatorImpl::release_buffers(BufferPool& pool)
{
  auto it = pool.buffers.begin();
  while (it != pool.buffers.end()) {
    BufferBlock* buffer_block = *it;
    ++it;
    release_buffer(buffer_block);
  }
}

bool MPSHeapAllocatorImpl::release_available_cached_buffers(const AllocParams& p)
{
  BufferPool& pool = *p.pool;

  if (max_split_size() == std::numeric_limits<size_t>::max() || pool.buffers.empty())
    return false;

  BufferBlock key = p.search_key;
  key.size = (key.size < max_split_size()) ? max_split_size() : key.size;
  auto it = pool.buffers.lower_bound(&key);
  if (it == pool.buffers.end()) {
    size_t totalReleased = 0;
    --it;
    while ((totalReleased < key.size) && ((*it)->size >= max_split_size())) {
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
    if (totalReleased < key.size)
      return false;
  } else {
    release_buffer(*it);
  }
  return true;
}

bool MPSHeapAllocatorImpl::release_cached_buffers()
{
  // Free all cached blocks to system allocator
  release_buffers(m_large_pool_private);
  release_buffers(m_large_pool_shared);
  release_buffers(m_small_pool_private);
  release_buffers(m_small_pool_shared);
  return true;
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
  explicit MPSAllocator(bool useSharedStorage) :
      m_has_unified_memory(_getAllocImpl().Device().hasUnifiedMemory), m_use_shared_storage(useSharedStorage)
  {
    const bool enable_debug_info = isEnvVarEnabled("PYTORCH_DEBUG_MPS_ALLOCATOR");
    if (enable_debug_info) {
      _getAllocImpl().enable_debug_info();
      if (!m_use_shared_storage || m_has_unified_memory) {
        std::cerr << "Initializing "
                  << (useSharedStorage ? "shared" : "private")
                  << " heap allocator on "
                  << (m_has_unified_memory ? "unified" : "discrete")
                  << " device memory of size "
                  << _getAllocImpl().Device().recommendedMaxWorkingSetSize / 1048576UL << " MB\n";
      }
    }
  }

  ~MPSAllocator() override {
    _getAllocImpl().EmptyCache();
  }

  DataPtr allocate(const size_t nbytes) const override {
    __block id<MTLBuffer> buf = nbytes > 0 ? _getAllocImpl().Malloc(nbytes, m_use_shared_storage) : nullptr;
    return { buf, buf, &Delete, at::Device(at::DeviceType::MPS, 0)};
  }

  DeleterFnPtr raw_deleter() const override { return &Delete; }
  bool is_shared(void* ptr) const { return _getAllocImpl().isSharedBuffer(ptr); }
  bool is_shared_storage_supported() const { return m_has_unified_memory; }

private:
  bool m_has_unified_memory;
  // use shared buffers on unified memory
  bool m_use_shared_storage;

  static void Delete(void* ptr) {
    if (ptr) {
      _getAllocImpl().Free(ptr);
    }
  }

  static bool isEnvVarEnabled(const char *envvar) {
    const char *e = getenv(envvar);
    if (e) {
      char *t = (char*) e;
      long val = strtol(e, &t, 0);
      return (t != e && val != 0);
    }
    return false;
  }
};

namespace {
MPSAllocator& _getSharedAllocator() {
  static MPSAllocator s_mps_shared_alloc(true);
  return s_mps_shared_alloc;
}

MPSAllocator& _getPrivateAllocator() {
  static mps::MPSAllocator s_mps_private_alloc(false);
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

at::Allocator* getMPSStaticAllocator() {
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
};

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
