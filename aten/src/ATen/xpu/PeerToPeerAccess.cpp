#include <ATen/xpu/PeerToPeerAccess.h>

#include <ATen/xpu/XPUContext.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUCachingAllocator.h>

#include <vector>

namespace at::xpu {

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

  for (const auto i : c10::irange(num_devices)) {
    p2pAccessEnabled_[i * num_devices + i] = 1;
  }
}

} // namespace detail

bool get_p2p_access(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);

  check_device_index(dev);
  check_device_index(dev_to_access);

  auto& cache = p2pAccessEnabled_[dev * num_devices_ + dev_to_access];

  if (cache != -1) {
    return cache;
  }

  cache = static_cast<int8_t>(
      c10::xpu::get_raw_device(dev).ext_oneapi_can_access_peer(
          c10::xpu::get_raw_device(dev_to_access),
          sycl::ext::oneapi::peer_access::access_supported));

  if (cache) {
    XPUCachingAllocator::enablePeerAccess(dev, dev_to_access);
  }

  return cache;
}

} // namespace at::xpu
