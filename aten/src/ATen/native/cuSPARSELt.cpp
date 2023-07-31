#if defined(CUDA_VERSION)
#include <ATen/cuda/CUDAConfig.h>
#endif

namespace at {
namespace native {

bool _cslt_available(){
    #if defined(CUDA_VERSION)
        return AT_CUSPARSELT_ENABLED();
    #else
        return false;
    #endif
}

} // namespace native
} // namespace at
