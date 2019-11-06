#include <string>

#include <c10/util/Registry.h>
#include <gloo/config.h>
#include <gloo/transport/device.h>

namespace c10d {

class GlooDeviceFactory {
 public:
  // Create new device instance for specific interface.
  static std::shared_ptr<::gloo::transport::Device> makeDeviceForInterface(
      const std::string& interface);

  // Create new device instance for specific hostname or address.
  static std::shared_ptr<::gloo::transport::Device> makeDeviceForHostname(
      const std::string& hostname);
};

C10_DECLARE_SHARED_REGISTRY(
    GlooDeviceRegistry,
    ::gloo::transport::Device,
    const std::string&, /* interface */
    const std::string& /* hostname */);

} // namespace c10d
