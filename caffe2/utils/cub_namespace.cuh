#include <ATen/cuda/cub_definitions.cuh>

#if USE_GLOBAL_CUB_WRAPPED_NAMESPACE()
namespace caffe2 {
namespace cub = ::CUB_WRAPPED_NAMESPACE::cub;
}
#endif
