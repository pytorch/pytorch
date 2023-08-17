#include <ATen/cuda/PeerToPeerAccess.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <vector>

namespace at::cuda {

static std::vector<int8_t> p2pAccessEnabled_;
static int64_t num_devices_ = -1;

namespace detail {

void init_p2p_access_cache(int64_t num_devices) {
  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  p2pAccessEnabled_.clear();
  p2pAccessEnabled_.resize(num_devices * num_devices, -1);
  num_devices_ = num_devices;

  for (const auto i : c10::irange(num_devices)) {
    p2pAccessEnabled_[i * num_devices + i] = 1;
  }
}

}  // namespace detail

bool get_p2p_access(int dev, int dev_to_access) {
  TORCH_CHECK(dev >= 0 || dev < num_devices_,
              dev, " is not a device");
  TORCH_CHECK(dev_to_access >= 0 || dev_to_access < num_devices_,
              dev_to_access, " is not a device");
  TORCH_INTERNAL_ASSERT(num_devices_ >= 0, "p2p access cache not initialized");

  auto &cache = p2pAccessEnabled_[dev * num_devices_ + dev_to_access];

  if (cache != -1) {
    return cache;
  }

  int result;
  C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&result, dev, dev_to_access));
  cache = result ? 1 : 0;
  if (cache) {
    CUDACachingAllocator::enablePeerAccess(dev, dev_to_access);
  }

  return cache;
}

}  // namespace at::cuda::detail
