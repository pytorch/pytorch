// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "c10d/socket.h"

#include <cstring>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <mutex>

#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <fmt/format.h>

#include "c10d/detail/error.h"
#include "c10d/detail/logging.h"
#include "c10d/interrupt_exception.h"

namespace c10d {
namespace detail {
namespace {
#ifdef _WIN32

const auto pollFd = ::WSAPoll;

int getSocketOption(SOCKET s, int level, int optname, void* optval, int* optlen) {
  return ::getsockopt(s, level, optname, static_cast<char*>(optval), optlen);
}

int setSocketOption(SOCKET s, int level, int optname, const void* optval, int optlen) {
  return ::setsockopt(s, level, optname, static_cast<const char*>(optval), optlen);
}

std::error_code getSocketError() noexcept {
  int wsa_error = ::WSAGetLastError();

  std::errc err{};
  switch (wsa_error) {
    case WSAEINTR:
      err = std::errc::interrupted;
      break;
    case WSAEINVAL:
      err = std::errc::invalid_argument;
      break;
    case WSAEWOULDBLOCK:
      err = std::errc::operation_would_block;
      break;
    case WSAEINPROGRESS:
      err = std::errc::operation_in_progress;
      break;
    case WSAECONNABORTED:
      err = std::errc::connection_aborted;
      break;
    case WSAECONNRESET:
      err = std::errc::connection_reset;
      break;
    case WSAEISCONN:
      err = std::errc::already_connected;
      break;
    case WSAECONNREFUSED:
      err = std::errc::connection_refused;
      break;

    default:
      return std::error_code{wsa_error, std::generic_category()};
  }
  return std::make_error_code(err);
}

inline void setSocketError(int e) noexcept {
  ::WSASetLastError(e);
}

#else

const auto pollFd = ::poll;

const auto getSocketOption = ::getsockopt;
const auto setSocketOption = ::setsockopt;

inline std::error_code getSocketError() noexcept {
  return lastError();
}

inline void setSocketError(int e) noexcept {
  errno = e;
}

#endif

class SocketListenOp;
class SocketConnectOp;
} // namespace

class SocketImpl {
  friend class SocketListenOp;
  friend class SocketConnectOp;

 public:
#ifdef _WIN32
  using Handle = SOCKET;
#else
  using Handle = int;
#endif

#ifdef _WIN32
  static constexpr Handle invalid_socket = INVALID_SOCKET;
#else
  static constexpr Handle invalid_socket = -1;
#endif

  explicit SocketImpl(Handle hnd) noexcept
      : hnd_{hnd} {}

  SocketImpl(const SocketImpl& other) = delete;

  SocketImpl& operator=(const SocketImpl& other) = delete;

  SocketImpl(SocketImpl&& other) noexcept = delete;

  SocketImpl& operator=(SocketImpl&& other) noexcept = delete;

  ~SocketImpl();

  std::unique_ptr<SocketImpl> accept() const;

  void closeOnExec() noexcept;

  void enableNonBlocking();

  void disableNonBlocking();

  bool enableNoDelay() noexcept;

  bool enableDualStack() noexcept;

  bool enableAddressReuse() noexcept;

  std::uint16_t getPort() const;

  Handle handle() const noexcept {
    return hnd_;
  }

 private:
  bool setSocketFlag(int level, int optname, bool value) noexcept;

  Handle hnd_;
};

SocketImpl::~SocketImpl() {
#ifdef _WIN32
  ::closesocket(hnd_);
#else
  ::close(hnd_);
#endif
}

std::unique_ptr<SocketImpl> SocketImpl::accept() const {
  ::sockaddr_storage addr_s{};

  auto addr_ptr = reinterpret_cast<::sockaddr*>(&addr_s);

  ::socklen_t addr_len = sizeof(addr_s);

  Handle hnd = ::accept(hnd_, addr_ptr, &addr_len);
  if (hnd == invalid_socket) {
    std::error_code err = getSocketError();
    if (err == std::errc::interrupted) {
      throw InterruptException{};
    }

    std::string msg{};
    if (err == std::errc::invalid_argument) {
      msg = fmt::format("The server socket on {} is not listening for connections.", *this);
    } else {
      msg = fmt::format("The server socket on {} has failed to accept a connection {}.", *this, err);
    }

    logError(msg);

    throw SocketError{msg};
  }

  ::addrinfo addr{};
  addr.ai_addr = addr_ptr;
  addr.ai_addrlen = addr_len;

  logInfo("The server socket on {} has accepted a connection from {}.", *this, addr);

  auto impl = std::make_unique<SocketImpl>(hnd);

  impl->closeOnExec();

  if (!impl->enableNoDelay()) {
    logWarning("The no-delay option cannot be enabled for the client socket on {}.", addr);
  }

  return impl;
}

void SocketImpl::closeOnExec() noexcept {
#ifndef _WIN32
  ::fcntl(hnd_, F_SETFD, FD_CLOEXEC);
#endif
}

void SocketImpl::enableNonBlocking() {
#ifdef _WIN32
  unsigned long value = 1;
  if (::ioctlsocket(hnd_, FIONBIO, &value) == 0) {
    return;
  }
#else
  int flg = ::fcntl(hnd_, F_GETFL);
  if (flg != -1) {
    if (::fcntl(hnd_, F_SETFL, flg | O_NONBLOCK) == 0) {
      return;
    }
  }
#endif
  throw SocketError{"The socket cannot be switched to non-blocking mode."};
}

void SocketImpl::disableNonBlocking() {
#ifdef _WIN32
  unsigned long value = 0;
  if (::ioctlsocket(hnd_, FIONBIO, &value) == 0) {
    return;
  }
#else
  int flg = ::fcntl(hnd_, F_GETFL);
  if (flg != -1) {
    if (::fcntl(hnd_, F_SETFL, flg & ~O_NONBLOCK) == 0) {
      return;
    }
  }
#endif
  throw SocketError{"The socket cannot be switched to blocking mode."};
}

bool SocketImpl::enableNoDelay() noexcept {
  return setSocketFlag(IPPROTO_TCP, TCP_NODELAY, true);
}

bool SocketImpl::enableDualStack() noexcept {
  return setSocketFlag(IPPROTO_IPV6, IPV6_V6ONLY, false);
}

bool SocketImpl::enableAddressReuse() noexcept {
  return setSocketFlag(SOL_SOCKET, SO_REUSEADDR, true);
}

std::uint16_t SocketImpl::getPort() const {
  ::sockaddr_storage addr_s{};

  ::socklen_t addr_len = sizeof(addr_s);

  if (::getsockname(hnd_, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len) != 0) {
    throw SocketError{"The port number of the socket cannot be retrieved."};
  }

  if (addr_s.ss_family == AF_INET) {
    return ntohs(reinterpret_cast<::sockaddr_in*>(&addr_s)->sin_port);
  } else {
    return ntohs(reinterpret_cast<::sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

bool SocketImpl::setSocketFlag(int lvl, int optname, bool value) noexcept {
#ifdef _WIN32
  auto buf = value ? TRUE : FALSE;
#else
  auto buf = value ? 1 : 0;
#endif
  return setSocketOption(hnd_, lvl, optname, &buf, sizeof(buf)) == 0;
}

namespace {

struct addrinfo_delete {
  void operator()(::addrinfo* addr) const noexcept {
    ::freeaddrinfo(addr);
  }
};

using addrinfo_ptr = std::unique_ptr<::addrinfo, addrinfo_delete>;

class SocketListenOp {
 public:
  SocketListenOp(std::uint16_t port, bool prefer_ipv6);

  std::unique_ptr<SocketImpl> run();

 private:
  bool tryListen(int family);

  bool tryListen(const ::addrinfo& addr);

  template <typename... Args>
  void recordError(fmt::string_view format, Args&&... args) {
    auto msg = fmt::format(format, std::forward<Args>(args)...);

    logWarning(msg);

    errors_.emplace_back(std::move(msg));
  }

  std::string port_;
  bool prefer_ipv6_;
  std::vector<std::string> errors_{};
  std::unique_ptr<SocketImpl> socket_{};
};

SocketListenOp::SocketListenOp(std::uint16_t port, bool prefer_ipv6)
    : port_{fmt::to_string(port)}, prefer_ipv6_{prefer_ipv6} {}

std::unique_ptr<SocketImpl> SocketListenOp::run() {
  if (prefer_ipv6_) {
    logInfo("The server socket will attempt to listen on an IPv6 address.");
    if (tryListen(AF_INET6)) {
      return std::move(socket_);
    }

    logInfo("The server socket will attempt to listen on an IPv4 address.");
    if (tryListen(AF_INET)) {
      return std::move(socket_);
    }
  } else {
    if (tryListen(AF_UNSPEC)) {
      return std::move(socket_);
    }
  }

  constexpr const char* msg = "The server socket has failed to listen on any local network address.";

  logError(msg);

  throw SocketError{fmt::format("{} {}", msg, fmt::join(errors_, " "))};
}

bool SocketListenOp::tryListen(int family) {
  ::addrinfo hints{}, *naked_result = nullptr;

  hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  int r = ::getaddrinfo(nullptr, port_.c_str(), &hints, &naked_result);
  if (r != 0) {
    recordError("The local network addresses cannot be retrieved {}.", ::gai_strerror(r));

    return false;
  }

  addrinfo_ptr result{naked_result};

  for (::addrinfo* addr = naked_result; addr != nullptr; addr = addr->ai_next) {
    logInfo("The server socket is attempting to listen on {}.", *addr);
    if (tryListen(*addr)) {
      return true;
    }
  }

  return false;
}

bool SocketListenOp::tryListen(const ::addrinfo& addr) {
  SocketImpl::Handle hnd = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  if (hnd == SocketImpl::invalid_socket) {
    recordError("The server socket cannot be initialized on {} {}.", addr, getSocketError());

    return false;
  }

  socket_ = std::make_unique<SocketImpl>(hnd);

  if (!socket_->enableAddressReuse()) {
    logWarning("The address reuse option cannot be enabled for the server socket on {}.", addr);
  }

  if (addr.ai_family == AF_INET6 && !socket_->enableDualStack()) {
    logWarning("The server socket does not support IPv4 communication on {}.", addr);
  }

  if (::bind(socket_->handle(), addr.ai_addr, addr.ai_addrlen) != 0) {
    recordError("The server socket has failed to bind to {} {}.", addr, getSocketError());

    return false;
  }

  // NOLINTNEXTLINE(bugprone-argument-comment)
  if (::listen(socket_->handle(), /*backlog=*/2048) != 0) {
    recordError("The server socket has failed to listen on {} {}.", addr, getSocketError());

    return false;
  }

  socket_->closeOnExec();

  logInfo("The server socket has started to listen on {}.", addr);

  return true;
}

class SocketConnectOp {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::steady_clock::duration;
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

 public:
  SocketConnectOp(const std::string& host, std::uint16_t port, bool prefer_ipv6);

  std::unique_ptr<SocketImpl> run();

 private:
  bool tryConnect(int family);

  bool tryConnect(const ::addrinfo& addr);

  int tryConnect(const ::addrinfo& addr, TimePoint deadline);

  template <typename... Args>
  void recordError(fmt::string_view format, Args&&... args) {
    auto msg = fmt::format(format, std::forward<Args>(args)...);

    logWarning(msg);

    errors_.emplace_back(std::move(msg));
  }

  const char* host_;
  std::string port_;
  bool prefer_ipv6_;
  std::vector<std::string> errors_{};
  std::unique_ptr<SocketImpl> socket_{};
};

SocketConnectOp::SocketConnectOp(const std::string& host, std::uint16_t port, bool prefer_ipv6)
    : host_{host.c_str()}, port_{fmt::to_string(port)}, prefer_ipv6_{prefer_ipv6} {}

std::unique_ptr<SocketImpl> SocketConnectOp::run() {
  if (prefer_ipv6_) {
    logInfo("The client socket will attempt to connect to an IPv6 address on ({}, {}).", host_, port_);
    if (tryConnect(AF_INET6)) {
      return std::move(socket_);
    }

    logInfo("The client socket will attempt to connect to an IPv4 address on ({}, {}).", host_, port_);
    if (tryConnect(AF_INET)) {
      return std::move(socket_);
    }
  } else {
    if (tryConnect(AF_UNSPEC)) {
      return std::move(socket_);
    }
  }

  auto msg = fmt::format(
      "The client socket has failed to connect to any network address on ({}, {}).", host_, port_);

  logError(msg);

  throw SocketError{fmt::format("{} {}", msg, fmt::join(errors_, " "))};
}

bool SocketConnectOp::tryConnect(int family) {
  ::addrinfo hints{}, *naked_result = nullptr;

  hints.ai_flags = AI_ADDRCONFIG | AI_V4MAPPED | AI_ALL | AI_NUMERICSERV;
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  int r = ::getaddrinfo(host_, port_.c_str(), &hints, &naked_result);
  if (r != 0) {
    const char* gai = ::gai_strerror(r);

    recordError("The network addresses of ({}, {}) cannot be retrieved {}.", host_, port_, gai);

    return false;
  }

  addrinfo_ptr result{naked_result};

  for (::addrinfo* addr = naked_result; addr != nullptr; addr = addr->ai_next) {
    logInfo("The client socket is attempting to connect to {}.", *addr);
    if (tryConnect(*addr)) {
      return true;
    }
  }

  return false;
}

bool SocketConnectOp::tryConnect(const ::addrinfo& addr) {
  std::chrono::seconds timeout{300};

  TimePoint deadline = Clock::now() + timeout;

  while (Clock::now() < deadline) {
    SocketImpl::Handle hnd = ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
    if (hnd == SocketImpl::invalid_socket) {
      std::error_code err = getSocketError();
      recordError("The client socket cannot be initialized to connect to {} {}.", addr, err);

      return false;
    }

    socket_ = std::make_unique<SocketImpl>(hnd);

    socket_->enableNonBlocking();

    int r = tryConnect(addr, deadline);
    if (r == -1) {
      std::error_code err = getSocketError();
      if (err == std::errc::interrupted) {
        throw InterruptException{};
      }

      // Retry if the server is not yet listening or if its backlog is exhausted.
      if (err == std::errc::connection_refused ||
          err == std::errc::connection_reset ||
          err == std::errc::connection_aborted) {
        logWarning("The server socket on {} is not yet listening {}, retrying...", addr, err);

        if (Clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        continue;
      }

      recordError("The client socket has failed to connect to {} {}.", addr, err);

      return false;
    }

    // A return value of 0 indicates that the operation has timed out.
    if (r == 0) {
      continue;
    }

    socket_->closeOnExec();

    socket_->disableNonBlocking();

    logInfo("The client socket has connected to {} on {}.", addr, *socket_);

    if (!socket_->enableNoDelay()) {
      logWarning("The no-delay option cannot be enabled for the client socket on {}.", *socket_);
    }

    return true;
  }

  recordError("The client socket has timed out while trying to connect to {}.", addr);

  return false;
}

int SocketConnectOp::tryConnect(const ::addrinfo& addr, TimePoint deadline) {
  Duration remaining = deadline - Clock::now();
  if (remaining <= Duration::zero()) {
    return 0;
  }

  int r = ::connect(socket_->handle(), addr.ai_addr, addr.ai_addrlen);
  if (r == 0) {
    return 1;
  }

  std::error_code err = getSocketError();
  if (err == std::errc::already_connected) {
    return 1;
  }

  if (err != std::errc::operation_in_progress && err != std::errc::operation_would_block) {
    return -1;
  }

  ::pollfd pfd{};
  pfd.fd = socket_->handle();
  pfd.events = POLLOUT;

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);

  r = pollFd(&pfd, 1, static_cast<int>(ms.count()));
  if (r <= 0) {
    return r;
  }

  int e = 0;

  ::socklen_t errlen = sizeof(int);

  r = getSocketOption(socket_->handle(), SOL_SOCKET, SO_ERROR, &e, &errlen);
  if (r != 0) {
    return -1;
  }

  if (e != 0) {
    setSocketError(e);

    return -1;
  }
  return 1;
}

} // namespace

void Socket::initialize() {
#ifdef _WIN32
  static std::once_flag init_flag{};

  // All processes that call socket functions on Windows must first initialize
  // the Winsock library.
  std::call_once(init_flag, []() {
    WSADATA data{};
    if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      throw SocketError{"The initialization of Winsock has failed."};
    }
  });
#endif
}

Socket Socket::listen(std::uint16_t port, bool prefer_ipv6) {
  SocketListenOp op{port, prefer_ipv6};

  return Socket{op.run()};
}

Socket Socket::connect(const std::string& host, std::uint16_t port, bool prefer_ipv6) {
  SocketConnectOp op{host, port, prefer_ipv6};

  return Socket{op.run()};
}

Socket::Socket(Socket&& other) noexcept = default;

Socket& Socket::operator=(Socket&& other) noexcept = default;

Socket::~Socket() = default;

Socket Socket::accept() const {
  if (impl_) {
    return Socket{impl_->accept()};
  }

  throw SocketError{"The socket is not initialized."};
}

int Socket::handle() const noexcept {
  if (impl_) {
    return impl_->handle();
  }
  return SocketImpl::invalid_socket;
}

std::uint16_t Socket::port() const {
  if (impl_) {
    return impl_->getPort();
  }
  return 0;
}

Socket::Socket(std::unique_ptr<SocketImpl>&& impl) noexcept
    : impl_{std::move(impl)} {}

} // namespace detail

SocketError::~SocketError() = default;

} // namespace c10d

namespace fmt {

template <>
struct formatter<::addrinfo> {
  constexpr decltype(auto) parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  decltype(auto) format(const ::addrinfo& addr, FormatContext& ctx) {
    char host[NI_MAXHOST], port[NI_MAXSERV]; // NOLINT

    int r = ::getnameinfo(
        addr.ai_addr, addr.ai_addrlen, host, NI_MAXHOST, port, NI_MAXSERV, NI_NUMERICSERV);
    if (r != 0) {
      return format_to(ctx.out(), "?UNKNOWN?");
    }

    if (addr.ai_addr->sa_family == AF_INET) {
      return format_to(ctx.out(), "{}:{}", host, port);
    } else {
      return format_to(ctx.out(), "[{}]:{}", host, port);
    }
  }
};

template <>
struct formatter<c10d::detail::SocketImpl> {
  constexpr decltype(auto) parse(format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  decltype(auto) format(const c10d::detail::SocketImpl& socket, FormatContext& ctx) {
    ::sockaddr_storage addr_s{};

    auto addr_ptr = reinterpret_cast<::sockaddr*>(&addr_s);

    ::socklen_t addr_len = sizeof(addr_s);

    if (::getsockname(socket.handle(), addr_ptr, &addr_len) != 0) {
      return format_to(ctx.out(), "?UNKNOWN?");
    }

    ::addrinfo addr{};
    addr.ai_addr = addr_ptr;
    addr.ai_addrlen = addr_len;

    return format_to(ctx.out(), "{}", addr);
  }
};

} // namespace fmt
