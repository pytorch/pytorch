namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

// Currently either "native" or "cudaMallocAsync"
const std::string& allocatorBackend();

size_t maxSplitSize();

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10
