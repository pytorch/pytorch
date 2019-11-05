#include <string>

#include <gloo/config.h>
#include <gloo/transport/device.h>

namespace c10d {

class GlooDeviceFactory {
 public:
  // Create new device instance for specific interface.
  static std::shared_ptr<::gloo::transport::Device> makeDeviceForInterface(
      const std::string& interface,
      const std::string& transport = "");

  // Create new device instance for specific hostname or address.
  static std::shared_ptr<::gloo::transport::Device> makeDeviceForHostname(
      const std::string& hostname,
      const std::string& transport = "");

  // Create new device instance.
  // It tries to resolve this machine's hostname and bind to that address.
  // If that fails (i.e. the hostname doesn't resolve to an address), it
  // falls back to binding to the loopback address.
  static std::shared_ptr<::gloo::transport::Device> makeDefaultDevice(
      const std::string& transport = "");

 private:
#ifdef __linux__
  inline static bool isTCPTransport(const std::string& transport) {
    return transport == "" || transport == "tcp";
  }

  inline static bool isTLSTransport(const std::string& transport) {
    return transport == "tls";
  }
#endif

#ifdef __APPLE__
  inline static bool isUVTransport(const std::string& transport) {
    return transport == "" || transport == "uv";
  }
#endif
};

} // namespace c10d
