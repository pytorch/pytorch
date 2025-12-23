#include <ATen/xpu/PeerToPeerAccess.h>
#include <ATen/xpu/XPUContext.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUCachingAllocator.h>

namespace at::xpu {

// p2pAccessEnabled_ is a flattened 2D matrix of size [num_devices x
// num_devices].
// Each element represents whether device[i] can access device[j]:
//   1  -> access allowed
//   0  -> access not allowed
//  -1  -> unknown (not yet queried)
static std::vector<int8_t> p2pAccessEnabled_;

namespace detail {

// Initializes the peer-to-peer (P2P) access capability cache.
void init_p2p_access_cache(c10::DeviceIndex num_devices) {
  // By default, each device can always access itself (diagonal entries = 1).
  // For simplicity, all entries are initialized to -1 except the diagonal.
  static bool once [[maybe_unused]] = [num_devices]() {
    p2pAccessEnabled_.clear();
    p2pAccessEnabled_.resize(num_devices * num_devices, -1);

    for (const auto i : c10::irange(num_devices)) {
      p2pAccessEnabled_[i * num_devices + i] = 1;
    }
    return true;
  }();
}

} // namespace detail

bool get_p2p_access(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);

  check_device_index(dev);
  check_device_index(dev_to_access);

  auto& cache =
      p2pAccessEnabled_[dev * c10::xpu::device_count() + dev_to_access];

  if (cache != -1) {
    return static_cast<bool>(cache);
  }

  // Query the hardware to determine if P2P access is supported
  cache = static_cast<int8_t>(
      c10::xpu::get_raw_device(dev).ext_oneapi_can_access_peer(
          c10::xpu::get_raw_device(dev_to_access),
          sycl::ext::oneapi::peer_access::access_supported));

  if (cache) {
    XPUCachingAllocator::enablePeerAccess(dev, dev_to_access);
  }

  return static_cast<bool>(cache);
}

} // namespace at::xpu
