# if defined(CUDA_VERSION)
#include <ATen/cuda/CUDAConfig.h>

namespace at {
namespace native {

bool _cslt_available(){
    return AT_CUSPARSELT_ENABLED();
}

} // namespace native
} // namespace at

#else
namespace at {
namespace native {

bool _cslt_available(){
    return false;
}

} // namespace native
} // namespace at

#endif
