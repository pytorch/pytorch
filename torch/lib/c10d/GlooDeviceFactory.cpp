#include <c10d/GlooDeviceFactory.hpp>

#include <limits.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

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

namespace {

// Gloo assumes that this machine's hostname can always be resolved
// to an address. If it doesn't it throws a runtime error saying
// that it can't be resolved. Instead of catching it, we choose
// to proactively check if an address can be resolved, so we can
// gracefully fall back to an alternative if it doesn't.
static bool doesHostnameResolveToUsableAddress(const std::string& hostname) {
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* result;
  auto rv = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
  if (rv < 0) {
    return false;
  }
  struct addrinfo* rp;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }
    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
    close(fd);
    if (rv == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  return rp != nullptr;
}

const auto kLoopbackAddress = "127.0.0.1";

} // namespace

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForInterface(
        const std::string& interface,
        const std::string& transport) {
#ifdef __linux__
  if (isTCPTransport(transport)) {
    ::gloo::transport::tcp::attr attr;
    attr.iface = interface;
    return ::gloo::transport::tcp::CreateDevice(attr);
  }
#endif

#ifdef __APPLE__
  if (isTCPTransport(transport)) {
    ::gloo::transport::uv::attr attr;
    attr.iface = interface;
    return ::gloo::transport::uv::CreateDevice(attr);
  }
#endif

  throw std::invalid_argument(
      "GlooDeviceFactory::makeDeviceForInterface: "
      "unsupported transpot type: " +
      transport);

  return nullptr;
}

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::
    makeDeviceForHostname(
        const std::string& hostname,
        const std::string& transport) {
#ifdef __linux__
  if (isTCPTransport(transport)) {
    ::gloo::transport::tcp::attr attr;
    attr.hostname = hostname;
    return ::gloo::transport::tcp::CreateDevice(attr);
  }
#endif

#ifdef __APPLE__
  if (isTCPTransport(transport)) {
    ::gloo::transport::uv::attr attr;
    attr.hostname = hostname;
    return ::gloo::transport::uv::CreateDevice(attr);
  }
#endif

  throw std::invalid_argument(
      "GlooDeviceFactory::makeDeviceForHostname: "
      "unsupported transpot type: " +
      transport);

  return nullptr;
}

std::shared_ptr<::gloo::transport::Device> GlooDeviceFactory::makeDefaultDevice(
    const std::string& transport) {
#ifdef __linux__
  if (isTCPTransport(transport)) {
    ::gloo::transport::tcp::attr attr;
    // Use the hostname to resolve the network address to
    // use. Note: if the hostname does not resolve to an
    // address (e.g. because of misconfigured /etc/hosts
    // file), this will not work.
    std::array<char, HOST_NAME_MAX> buffer{};
    auto rv = gethostname(buffer.data(), buffer.size());
    if (rv != 0) {
      throw std::system_error(errno, std::system_category());
    }
    attr.hostname = buffer.data();

    // Use this machine's hostname if it resolves to an address.
    if (doesHostnameResolveToUsableAddress(attr.hostname)) {
      return ::gloo::transport::tcp::CreateDevice(attr);
    }

    // Otherwise, use the loopback address.
    TORCH_WARN_ONCE(
        "Unable to resolve hostname to a (local) address. ",
        "Using the loopback address as fallback. ",
        "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
    return makeDeviceForHostname(kLoopbackAddress, transport);
  }
#endif

#ifdef __APPLE__
  if (isTCPTransport(transport)) {
    ::gloo::transport::uv::attr attr;
    // Use the hostname to resolve the network address to
    // use. Note: if the hostname does not resolve to an address (e.g.
    // because of misconfigured /etc/hosts file), this will not work.
    const auto hostNameMax = sysconf(_SC_HOST_NAME_MAX);
    auto buffer = std::unique_ptr<char[]>(new char[hostNameMax]);
    auto rv = gethostname(buffer.get(), hostNameMax);
    if (rv != 0) {
      throw std::system_error(errno, std::system_category());
    }
    attr.hostname = buffer.get();

    // Use this machine's hostname if it resolves to an address.
    if (doesHostnameResolveToUsableAddress(attr.hostname)) {
      return ::gloo::transport::uv::CreateDevice(attr);
    }

    // Otherwise, use the loopback address.
    TORCH_WARN_ONCE(
        "Unable to resolve hostname to a (local) address. ",
        "Using the loopback address as fallback. ",
        "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
    return makeDeviceForHostname(kLoopbackAddress, transport);
  }
#endif

  throw std::invalid_argument(
      "GlooDeviceFactory::makeDeviceForHostname: "
      "unsupported transpot type: " +
      transport);

  return nullptr;
}

} // namespace c10d
