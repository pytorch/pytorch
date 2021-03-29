#include <c10d/GlooDeviceFactory.hpp>

#include <stdlib.h>

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

C10_DEFINE_SHARED_REGISTRY_WITHOUT_WARNING(
    GlooDeviceRegistry,
    ::gloo::transport::Device,
    const std::string& /* interface */,
    const std::string& /* hostname */);

#if GLOO_HAVE_TRANSPORT_TCP
static std::shared_ptr<::gloo::transport::Device> makeTCPDevice(
    const std::string& interfaceName,
    const std::string& hostname) {
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeTCPDevice(): interface or hostname "
      "can't be empty");

  ::gloo::transport::tcp::attr attr;
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    attr.hostname = hostname;
  }
  return ::gloo::transport::tcp::CreateDevice(attr);
}

// Registry priority is per key identifier. We register TCP to `LINUX` for
// the flexibility of other application to override by priority. Register
// TCP to `TCP` for env "GLOO_DEVICE_TRANSPORT" override.
C10_REGISTER_CREATOR(GlooDeviceRegistry, LINUX, makeTCPDevice);
C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP, makeTCPDevice);
#endif

#if GLOO_HAVE_TRANSPORT_UV
static std::shared_ptr<::gloo::transport::Device> makeUVDevice(
    const std::string& interfaceName,
    const std::string& hostname) {
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeUVDevice(): interface or hostname "
      "can't be empty");

  ::gloo::transport::uv::attr attr;
  if (!interfaceName.empty()) {
    attr.iface = interfaceName;
  } else {
    attr.hostname = hostname;
  }
  return ::gloo::transport::uv::CreateDevice(attr);
}

// Registry priority is per key identifier. We register UV to `APPLE` for
// the flexibility of other application to override by priority. Register
// UV to `UV` for env "GLOO_DEVICE_TRANSPORT" override.
C10_REGISTER_CREATOR(GlooDeviceRegistry, APPLE, makeUVDevice);
C10_REGISTER_CREATOR(GlooDeviceRegistry, WIN32, makeUVDevice);
C10_REGISTER_CREATOR(GlooDeviceRegistry, UV, makeUVDevice);
#endif

static const char* glooDeviceTransport = getenv("GLOO_DEVICE_TRANSPORT");

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForInterface(const std::string& interfaceName) {
  if (glooDeviceTransport) {
    return GlooDeviceRegistry()->Create(glooDeviceTransport, interfaceName, "");
  }

#ifdef __linux__
  return GlooDeviceRegistry()->Create("LINUX", interfaceName, "");
#endif

#ifdef __APPLE__
  return GlooDeviceRegistry()->Create("APPLE", interfaceName, "");
#endif

#ifdef _WIN32
  return GlooDeviceRegistry()->Create("WIN32", interfaceName, "");
#endif

  throw std::runtime_error("makeDeviceForInterface(): unsupported gloo device");
}

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForHostname(const std::string& hostname) {
  if (glooDeviceTransport) {
    return GlooDeviceRegistry()->Create(glooDeviceTransport, "", hostname);
  }

#ifdef __linux__
  return GlooDeviceRegistry()->Create("LINUX", "", hostname);
#endif

#ifdef __APPLE__
  return GlooDeviceRegistry()->Create("APPLE", "", hostname);
#endif

#ifdef _WIN32
  return GlooDeviceRegistry()->Create("WIN32", "", hostname);
#endif

  throw std::runtime_error("makeDeviceForHostname(): unsupported gloo device");
}

} // namespace c10d
