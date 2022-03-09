#include <c10/macros/Macros.h>
#include <cstdint>

namespace at {
namespace cuda {
namespace detail {
void init_p2p_access_cache(int64_t num_devices);
}

TORCH_CUDA_CPP_API bool get_p2p_access(int source_dev, int dest_dev);

}}  // namespace at::cuda
