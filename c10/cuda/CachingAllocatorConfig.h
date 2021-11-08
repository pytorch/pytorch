namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

// Currently either "native" or "cudaMallocAsync"
const std::string& allocatorBacked() {
  return CachingAllocatorConfig::allocator_backend();
}

size_t maxSplitSize() {
  return CachingAllocatorConfig::max_split_size()
}

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
