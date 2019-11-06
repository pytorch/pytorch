#include <c10d/GlooDeviceFactory.hpp>

#include <c10/util/Exception.h>

#if GLOO_HAVE_TRANSPORT_TCP
#include <gloo/transport/tcp/device.h>
#endif

#if GLOO_HAVE_TRANSPORT_UV
#include <gloo/transport/uv/device.h>
#endif

// On Linux, check that the tcp transport is available.
#ifdef __linux__
#if !GLOO_HAVE_TRANSPORT_TCP
#error "Expected the tcp transport to be available on Linux."
#endif
#endif

// On macOS, check that the uv transport is available.
#ifdef __APPLE__
#if !GLOO_HAVE_TRANSPORT_UV
#error "Expected the uv transport to be available on macOS."
#endif
#endif

namespace c10d {

C10_DEFINE_SHARED_REGISTRY(
    GlooDeviceRegistry,
    ::gloo::transport::Device,
    const std::string& /* interface */,
    const std::string& /* hostname */);

#if GLOO_HAVE_TRANSPORT_TCP
static std::shared_ptr<::gloo::transport::Device> makeTCPDevice(
    const std::string& interface,
    const std::string& hostname) {
  TORCH_CHECK(
      !interface.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeTCPDevice(): interface or hostname "
      "can't be empty");

  ::gloo::transport::tcp::attr attr;
  if (!interface.empty()) {
    attr.iface = interface;
  } else {
    attr.hostname = hostname;
  }
  return ::gloo::transport::tcp::CreateDevice(attr);
}

C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP, makeTCPDevice);
#endif

#if GLOO_HAVE_TRANSPORT_UV
static std::shared_ptr<::gloo::transport::Device> makeUVDevice(
    const std::string& interface,
    const std::string& hostname) {
  TORCH_CHECK(
      !interface.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeUVDevice(): interface or hostname "
      "can't be empty");

  ::gloo::transport::uv::attr attr;
  if (!interface.empty()) {
    attr.iface = interface;
  } else {
    attr.hostname = hostname;
  }
  return ::gloo::transport::uv::CreateDevice(attr);
}

C10_REGISTER_CREATOR(GlooDeviceRegistry, UV, makeUVDevice);
#endif

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForInterface(const std::string& interface) {
#ifdef __linux__
  return GlooDeviceRegistry()->Create("TCP", interface, "");
#endif

#ifdef __APPLE__
  return GlooDeviceRegistry()->Create("UV", interface, "");
#endif

  return nullptr;
}

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForHostname(const std::string& hostname) {
#ifdef __linux__
  return GlooDeviceRegistry()->Create("TCP", "", hostname);
#endif

#ifdef __APPLE__
  return GlooDeviceRegistry()->Create("UV", "", hostname);
#endif

  return nullptr;
}

} // namespace c10d
