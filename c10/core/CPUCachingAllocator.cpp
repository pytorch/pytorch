#include <c10/core/CPUCachingAllocator.h>

namespace c10 {

namespace {
thread_local CachingAllocatorInfo caching_allocator_info;
} // namespace

std::mutex CPUCachingAllocator::mutex_;
ska::flat_hash_map<void*, size_t> CPUCachingAllocator::allocation_map_;

inline void* CPUCachingAllocator::allocate_and_cache(const size_t bytes) {
  void* ptr;
  try {
    ptr = c10::alloc_cpu(bytes);
  } catch (c10::Error& e) {
    // If allocation fails, try freeing cached available blocks.
    // For now free all available cached blocks.
    free_cached();
    // Furthermore to consider: If we ever come here running out of memory
    // perhaps it is best to disable caching, since this is likely to happen
    // again.
    // Try again.
    ptr = c10::alloc_cpu(bytes);
    TORCH_CHECK(ptr, "Memory allocation failed for ", bytes, " bytes.");
  }
  allocation_map_[ptr] = bytes;
  return ptr;
}

inline void* CPUCachingAllocator::use_cached(const size_t bytes) {
  void* ptr = available_map_[bytes].pop_back_val();
  return ptr;
}

void* CPUCachingAllocator::allocate(const size_t bytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = available_map_.find(bytes);
  if (it == available_map_.end() || it->second.empty()) {
    return allocate_and_cache(bytes);
  }
  return use_cached(bytes);
}

void CPUCachingAllocator::free(void* ptr) {
  // NB: since we are not really freeing the memory
  // the cases such as quantization code freeing original weights
  // on mobile, will not quite work, as we likely will hold
  // onto that memory.
  // NB: We can also enable max memory cached for better memory
  // management such that free will actually free the memory if
  // we are nearing or above the watermark.
  std::lock_guard<std::mutex> guard(mutex_);
  // If this allocation was done before caching allocator was enabled
  // then free regularly
  if (allocation_map_.find(ptr) == allocation_map_.end()) {
    c10::free_cpu(ptr);
    return;
  }
  const size_t alloc_size = allocation_map_[ptr];
  available_map_[alloc_size].push_back(ptr);
}

void CPUCachingAllocator::record_free(void* ptr) {
  // This function captures the case when the allocated memory
  // is being freed outside the scope of this allocator.
  // At the moment only way to capture this is to have the allocator,
  // that uses this CachingAllocator as the backing allocator,
  // call this function explicity upon freeing memory while
  // outside the scope of caching allocator.
  // If the memory is freed in some other way, then we will likely
  // have undefined behavior or page fault. But this can be
  // the case without caching allocator as well.
  std::lock_guard<std::mutex> guard(mutex_);
  const auto& it = allocation_map_.find(ptr);
  if (it != allocation_map_.end()) {
    allocation_map_.erase(it);
  }
}

void CPUCachingAllocator::free_cached() {
  for (const auto& it : available_map_) {
    for (const auto ptr : it.second) {
      c10::free_cpu(ptr);
      // When cached memory is return to OS, it must be removed
      // from allocation_map.
      allocation_map_.erase(ptr);
    }
  }
  available_map_.clear();
}

CPUCachingAllocator::~CPUCachingAllocator() {
  free_cached();
}

CachingAllocatorInfo& GetThreadLocalCachingAllocatorInfo() {
  return caching_allocator_info;
}

bool CachingAllocatorInfo::enabled() {
  return is_enabled_;
}

void CachingAllocatorInfo::enable() {
  TORCH_CHECK(
      allocator_ != nullptr,
      "Enabling of caching allocator requires allocator "
      "to be initialized via set_allocator");
  is_enabled_ = true;
}

// Unfortunately get_allocator, cannot be inlined
// because tis definition is removed, which result in
// undefined ref.
// Alternatively is to add allocate and free interface to
// CachineAllocatorInfo, but that does not seem clean.
CPUCachingAllocator* CachingAllocatorInfo::get_allocator() {
  return allocator_;
}

void CachingAllocatorInfo::set_allocator(CPUCachingAllocator* allocator) {
  allocator_ = allocator;
}

void CachingAllocatorInfo::disable() {
  is_enabled_ = false;
}

WithCPUCachingAllocatorGuard::WithCPUCachingAllocatorGuard(
    CPUCachingAllocator* allocator, bool enable) {
  prev_info_ = GetThreadLocalCachingAllocatorInfo();
  GetThreadLocalCachingAllocatorInfo().set_allocator(allocator);
  if (enable) {
    GetThreadLocalCachingAllocatorInfo().enable();
  } else {
    GetThreadLocalCachingAllocatorInfo().disable();
  }
}

WithCPUCachingAllocatorGuard::~WithCPUCachingAllocatorGuard() {
  GetThreadLocalCachingAllocatorInfo() = prev_info_;
}

} // namespace c10
