#include <ATen/cuda/PeerToPeerAccess.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <vector>
#include <algorithm>

namespace at {
namespace cuda {
namespace detail {

static std::vector<int8_t> p2pAccessEnabled;
static int64_t num_devices = -1;

void init_p2p_access_cache(int64_t num_devices) {
  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  p2pAccessEnabled.clear();
  p2pAccessEnabled.resize(num_devices * num_devices, -1);

  for (int64_t i = 0; i < num_devices; ++i) {
    p2pAccessEnabled[i * num_devices + i] = 1;
  }
}

bool get_p2p_access(int source_dev, int dest_dev) {
  TORCH_CHECK(source_dev >= 0 || source_dev < num_devices,
              source_dev, " is not a device");
  TORCH_CHECK(dest_dev >= 0 || dest_dev < num_devices,
              dest_dev, " is not a device");
  TORCH_INTERNAL_ASSERT(num_devices >= 0, "p2p access cache not initialized");

  auto &cache = p2pAccessEnabled[source_dev * num_devices + dest_dev];

  if (cache != -1) {
    return cache;
  }

  c10::cuda::CUDAGuard device_guard(source_dev);

  int access = 0;
  C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&access, source_dev, dest_dev));
  if (access) {
    cudaError_t err = cudaDeviceEnablePeerAccess(dest_dev, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
      // ignore and clear the error if access was already enabled
      cudaGetLastError();
    } else {
      C10_CUDA_CHECK(err);
    }
    cache = 1;
  } else {
    cache = 0;
  }
  return cache;
}

}}}  // namespace at::cuda::detail
