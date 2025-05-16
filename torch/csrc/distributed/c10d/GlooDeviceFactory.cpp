#include <torch/csrc/distributed/c10d/GlooDeviceFactory.hpp>

#include <torch/csrc/distributed/c10d/Utils.hpp>

#ifdef USE_C10D_GLOO

#include <cstdlib>

#include <c10/util/Exception.h>
#include <c10/util/env.h>

#if GLOO_HAVE_TRANSPORT_TCP
#include <gloo/transport/tcp/device.h>
#endif

#if GLOO_HAVE_TRANSPORT_TCP_TLS
#include <gloo/transport/tcp/tls/device.h>
#endif

#if GLOO_HAVE_TRANSPORT_UV
#include <gloo/transport/uv/device.h>
#endif

#if GLOO_HAVE_TRANSPORT_IBVERBS
#include <gloo/transport/ibverbs/device.h>
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
    const std::string& /* hostname */,
    bool /* lazyInit */)

#if GLOO_HAVE_TRANSPORT_TCP
static std::shared_ptr<::gloo::transport::Device> makeTCPDevice(
    const std::string& interfaceName,
    const std::string& hostname,
    bool lazyInit) {
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
  if (lazyInit) {
    return ::gloo::transport::tcp::CreateLazyDevice(attr);
  } else {
    return ::gloo::transport::tcp::CreateDevice(attr);
  }
}

// Registry priority is per key identifier. We register TCP to `LINUX` for
// the flexibility of other application to override by priority. Register
// TCP to `TCP` for env "GLOO_DEVICE_TRANSPORT" override.
C10_REGISTER_CREATOR(GlooDeviceRegistry, LINUX, makeTCPDevice)
C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP, makeTCPDevice)
#endif

#if GLOO_HAVE_TRANSPORT_TCP_TLS
static std::shared_ptr<::gloo::transport::Device> makeTCPTLSDevice(
    const std::string& interface,
    const std::string& hostname,
    bool lazyInit) {
  TORCH_CHECK(
      !interface.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeTCPTLSDevice(): interface or hostname "
      "can't be empty");

  TORCH_CHECK(!lazyInit, "TCP_TLS transport does not support lazy init");

  ::gloo::transport::tcp::attr attr;
  if (!interface.empty()) {
    attr.iface = interface;
  } else {
    attr.hostname = hostname;
  }
  const auto pkey_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY");
  const auto pkey = pkey_env.has_value() ? pkey_env.value() : std::string();
  const auto cert_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT");
  const auto cert = cert_env.has_value() ? cert_env.value() : std::string();
  const auto caFile_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE");
  const auto caFile =
      caFile_env.has_value() ? caFile_env.value() : std::string();
  const auto caPath_env =
      c10::utils::get_env("GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_PATH");
  const auto caPath =
      caPath_env.has_value() ? caPath_env.value() : std::string();
  return ::gloo::transport::tcp::tls::CreateDevice(
      attr, pkey, cert, caFile, caPath);
}

C10_REGISTER_CREATOR(GlooDeviceRegistry, TCP_TLS, makeTCPTLSDevice)
#endif

#if GLOO_HAVE_TRANSPORT_UV
static std::shared_ptr<::gloo::transport::Device> makeUVDevice(
    const std::string& interfaceName,
    const std::string& hostname,
    bool lazyInit) {
  TORCH_CHECK(
      !interfaceName.empty() || !hostname.empty(),
      "GlooDeviceFactory::makeUVDevice(): interface or hostname "
      "can't be empty");

  TORCH_CHECK(!lazyInit, "UV transport does not support lazy init");

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
C10_REGISTER_CREATOR(GlooDeviceRegistry, APPLE, makeUVDevice)
C10_REGISTER_CREATOR(GlooDeviceRegistry, WIN32, makeUVDevice)
C10_REGISTER_CREATOR(GlooDeviceRegistry, UV, makeUVDevice)
#endif

#if GLOO_HAVE_TRANSPORT_IBVERBS
static std::shared_ptr<::gloo::transport::Device> makeIBVerbsDevice(
    const std::string& interface,
    const std::string& hostname,
    bool lazyInit) {
  if (!hostname.empty()) {
    TORCH_WARN(
        "ibverbs transport does not support hostname, defaulting to any");
  }

  TORCH_CHECK(!lazyInit, "transport does not support lazy init");

  ::gloo::transport::ibverbs::attr attr;
  attr.name = getCvarString(
      {
          "TORCH_GLOO_IBV_NAME",
      },
      "");
  attr.port = getCvarInt(
      {
          "TORCH_GLOO_IBV_PORT",
      },
      1);
  attr.index = getCvarInt(
      {
          "TORCH_GLOO_IBV_INDEX",
      },
      0);

  if (!interface.empty()) {
    attr.name = interface;
  }

  // use global port
  attr.port = 1;

  return ::gloo::transport::ibverbs::CreateDevice(attr);
}

C10_REGISTER_CREATOR(GlooDeviceRegistry, IBVERBS, makeIBVerbsDevice)
#endif

namespace {
std::shared_ptr<::gloo::transport::Device> makeGlooDevice(
    const std::string& interfaceName,
    const std::string& hostName,
    bool lazyInit) {
  static auto transportName = c10::utils::get_env("GLOO_DEVICE_TRANSPORT");
  if (transportName.has_value()) {
    return GlooDeviceRegistry()->Create(
        transportName.value().c_str(), interfaceName, hostName, lazyInit);
  }

#ifdef __linux__
  return GlooDeviceRegistry()->Create(
      "LINUX", interfaceName, hostName, lazyInit);
#endif

#ifdef __APPLE__
  return GlooDeviceRegistry()->Create(
      "APPLE", interfaceName, hostName, lazyInit);
#endif

#ifdef _WIN32
  return GlooDeviceRegistry()->Create(
      "WIN32", interfaceName, hostName, lazyInit);
#endif

  return nullptr;
}
} // anonymous namespace

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForInterface(const std::string& interfaceName, bool lazyInit) {
  auto device = makeGlooDevice(interfaceName, "", lazyInit);
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForInterface(): unsupported gloo device");
  }
  return device;
}

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForHostname(const std::string& hostname, bool lazyInit) {
  auto device = makeGlooDevice("", hostname, lazyInit);
  if (!device) {
    TORCH_CHECK(false, "makeDeviceForHostname(): unsupported gloo device");
  }
  return device;
}

} // namespace c10d

#endif // USE_C10D_GLOO
