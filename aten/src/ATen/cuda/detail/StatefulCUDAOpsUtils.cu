#include <ATen/cuda/StatefulCUDAOpsUtils.cuh>

namespace at {
namespace cuda {
namespace philox {

namespace {
// Updates device-state offset capturably.
// Must be launched with a single thread.
__global__ void update_offset_kernel(PhiloxCudaState arg) {
  *arg.offset_.ptr += arg.increment_;
}
} // anonymous namespace

void update_offset(PhiloxCudaState arg) {
  update_offset_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(arg);
}

} // namespace philox
} // namespace cuda
} // namespace at
