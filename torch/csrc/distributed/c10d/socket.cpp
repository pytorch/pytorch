// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/util/error.h>
#include <torch/csrc/distributed/c10d/socket.h>

#include <cstring>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <mutex>

#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <torch/csrc/distributed/c10d/error.h>
#include <torch/csrc/distributed/c10d/exception.h>
#include <torch/csrc/distributed/c10d/logging.h>
#include <torch/csrc/distributed/c10d/socket_fmt.h>

namespace c10d::detail {
namespace {
#ifdef _WIN32

// Since Winsock uses the name `WSAPoll` instead of `poll`, we alias it here
// to avoid #ifdefs in the source code.
const auto pollFd = ::WSAPoll;

// Winsock's `getsockopt()` and `setsockopt()` functions expect option values to
// be passed as `char*` instead of `void*`. We wrap them here to avoid redundant
// casts in the source code.
int getSocketOption(
    SOCKET s,
    int level,
    int optname,
    void* optval,
    int* optlen) {
  return ::getsockopt(s, level, optname, static_cast<char*>(optval), optlen);
}

int setSocketOption(
    SOCKET s,
    int level,
    int optname,
    const void* optval,
    int optlen) {
  return ::setsockopt(
      s, level, optname, static_cast<const char*>(optval), optlen);
}

// Winsock has its own error codes which differ from Berkeley's. Fortunately the
// C++ Standard Library on Windows can map them to standard error codes.
inline std::error_code getSocketError() noexcept {
  return std::error_code{::WSAGetLastError(), std::system_category()};
}

inline void setSocketError(int val) noexcept {
  ::WSASetLastError(val);
}

#else

const auto pollFd = ::poll;

const auto getSocketOption = ::getsockopt;
const auto setSocketOption = ::setsockopt;

inline std::error_code getSocketError() noexcept {
  return lastError();
}

inline void setSocketError(int val) noexcept {
  errno = val;
}

#endif

// Suspends the current thread for the specified duration.
void delay(std::chrono::milliseconds d) {
#ifdef _WIN32
  std::this_thread::sleep_for(d);
#else
  ::timespec req{};
  auto ms = d.count();
  req.tv_sec = ms / 1000;
  req.tv_nsec = (ms % 1000) * 1000000;

  // The C++ Standard does not specify whether `sleep_for()` should be signal-
  // aware; therefore, we use the `nanosleep()` syscall.
  if (::nanosleep(&req, nullptr) != 0) {
    std::error_code err = getSocketError();
    // We don't care about error conditions other than EINTR since a failure
    // here is not critical.
    if (err == std::errc::interrupted) {
      C10_THROW_ERROR(DistNetworkError, c10::utils::str_error(err.value()));
    }
  }
#endif
}

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

  explicit SocketImpl(Handle hnd) noexcept : hnd_{hnd} {}

  explicit SocketImpl(Handle hnd, const ::addrinfo& remote);

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

#ifndef _WIN32
  bool enableAddressReuse() noexcept;
#endif

#ifdef _WIN32
  bool enableExclusiveAddressUse() noexcept;
#endif

  std::uint16_t getPort() const;

  Handle handle() const noexcept {
    return hnd_;
  }

  const std::optional<std::string>& remote() const noexcept {
    return remote_;
  }

  bool waitForInput(std::chrono::milliseconds timeout);

 private:
  bool setSocketFlag(int level, int optname, bool value) noexcept;

  Handle hnd_;
  const std::optional<std::string> remote_;
};

std::string formatSockAddr(const struct ::sockaddr* addr, socklen_t len) {
  // It can be be very slow to repeatedly hit DNS resolution failure, but its
  // very helpful to have DNS names in logs by default. So we try to use DNS but
  // if we hit a transient failure we just disable it for the remainder of the
  // job, logging IP addresses instead. See
  // https://github.com/pytorch/pytorch/issues/159007
  static bool disable_getnameinfo = false;

  char host[NI_MAXHOST], port[NI_MAXSERV]; // NOLINT

  if (!disable_getnameinfo) {
    int err = ::getnameinfo(
        addr, len, host, NI_MAXHOST, port, NI_MAXSERV, NI_NUMERICSERV);
    if (err != 0) {
      C10D_WARNING(
          "The hostname of the client socket cannot be retrieved. err={}", err);
      disable_getnameinfo = true;
    }
  }
  // if getnameinfo failed, disable would be set
  if (!disable_getnameinfo) {
    if (addr->sa_family == AF_INET) {
      return fmt::format("{}:{}", host, port);
    }
    return fmt::format("[{}]:{}", host, port);
  }
  // if we can't resolve the hostname, display the IP address
  if (addr->sa_family == AF_INET) {
    struct sockaddr_in* psai = reinterpret_cast<struct sockaddr_in*>(&addr);
    // NOLINTNEXTLINE(*array*)
    char ip[INET_ADDRSTRLEN];
    if (inet_ntop(addr->sa_family, &(psai->sin_addr), ip, INET_ADDRSTRLEN) !=
        nullptr) {
      return fmt::format("{}:{}", ip, psai->sin_port);
    }
  } else if (addr->sa_family == AF_INET6) {
    struct sockaddr_in6* psai = reinterpret_cast<struct sockaddr_in6*>(&addr);
    // NOLINTNEXTLINE(*array*)
    char ip[INET6_ADDRSTRLEN];
    if (inet_ntop(addr->sa_family, &(psai->sin6_addr), ip, INET6_ADDRSTRLEN) !=
        nullptr) {
      return fmt::format("[{}]:{}", ip, psai->sin6_port);
    }
  }
  return "?UNKNOWN?";
}
} // namespace c10d::detail

//
// libfmt formatters for `addrinfo` and `Socket`
//
namespace fmt {

template <>
struct formatter<::addrinfo> {
  constexpr auto parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const ::addrinfo& addr, FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "{}",
        c10d::detail::formatSockAddr(addr.ai_addr, addr.ai_addrlen));
  }
};

template <>
struct formatter<c10d::detail::SocketImpl> {
  constexpr auto parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const c10d::detail::SocketImpl& socket, FormatContext& ctx)
      const {
    ::sockaddr_storage addr_s{};

    auto addr_ptr = reinterpret_cast<::sockaddr*>(&addr_s);

    ::socklen_t addr_len = sizeof(addr_s);

    auto fd = socket.handle();

    if (::getsockname(fd, addr_ptr, &addr_len) != 0) {
      return fmt::format_to(ctx.out(), "?UNKNOWN?");
    }

    ::addrinfo addr{};
    addr.ai_addr = addr_ptr;
    addr.ai_addrlen = addr_len;

    auto const& remote = socket.remote();
    std::string remoteStr = remote ? *remote : "none";

    return fmt::format_to(
        ctx.out(),
        "SocketImpl(fd={}, addr={}, remote={})",
        fd,
        addr,
        remoteStr);
  }
};

} // namespace fmt

namespace c10d::detail {

SocketImpl::SocketImpl(Handle hnd, const ::addrinfo& remote)
    : hnd_{hnd}, remote_{fmt::format("{}", remote)} {}

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
      C10_THROW_ERROR(DistNetworkError, c10::utils::str_error(err.value()));
    }

    std::string msg{};
    if (err == std::errc::invalid_argument) {
      msg = fmt::format(
          "The server socket on {} is not listening for connections.", *this);
    } else {
      msg = fmt::format(
          "The server socket on {} has failed to accept a connection {}.",
          *this,
          err);
    }

    C10D_ERROR(msg);

    C10D_THROW_ERROR(SocketError, msg);
  }

  ::addrinfo addr{};
  addr.ai_addr = addr_ptr;
  addr.ai_addrlen = addr_len;

  C10D_DEBUG(
      "The server socket on {} has accepted a connection from {}.",
      *this,
      addr);

  auto impl = std::make_unique<SocketImpl>(hnd, addr);

  // Make sure that we do not "leak" our file descriptors to child processes.
  impl->closeOnExec();

  if (!impl->enableNoDelay()) {
    C10D_WARNING(
        "The no-delay option cannot be enabled for the client socket on {}.",
        addr);
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
  C10D_THROW_ERROR(
      SocketError, "The socket cannot be switched to non-blocking mode.");
}

// TODO: Remove once we migrate everything to non-blocking mode.
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
  C10D_THROW_ERROR(
      SocketError, "The socket cannot be switched to blocking mode.");
}

bool SocketImpl::enableNoDelay() noexcept {
  return setSocketFlag(IPPROTO_TCP, TCP_NODELAY, true);
}

bool SocketImpl::enableDualStack() noexcept {
  return setSocketFlag(IPPROTO_IPV6, IPV6_V6ONLY, false);
}

#ifndef _WIN32
bool SocketImpl::enableAddressReuse() noexcept {
  return setSocketFlag(SOL_SOCKET, SO_REUSEADDR, true);
}
#endif

#ifdef _WIN32
bool SocketImpl::enableExclusiveAddressUse() noexcept {
  return setSocketFlag(SOL_SOCKET, SO_EXCLUSIVEADDRUSE, true);
}
#endif

std::uint16_t SocketImpl::getPort() const {
  ::sockaddr_storage addr_s{};

  ::socklen_t addr_len = sizeof(addr_s);

  if (::getsockname(hnd_, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len) !=
      0) {
    C10D_THROW_ERROR(
        SocketError, "The port number of the socket cannot be retrieved.");
  }

  if (addr_s.ss_family == AF_INET) {
    return ntohs(reinterpret_cast<::sockaddr_in*>(&addr_s)->sin_port);
  } else {
    return ntohs(reinterpret_cast<::sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

bool SocketImpl::setSocketFlag(int level, int optname, bool value) noexcept {
#ifdef _WIN32
  auto buf = value ? TRUE : FALSE;
#else
  auto buf = value ? 1 : 0;
#endif
  return setSocketOption(hnd_, level, optname, &buf, sizeof(buf)) == 0;
}

bool SocketImpl::waitForInput(std::chrono::milliseconds timeout) {
  using Clock = std::chrono::steady_clock;

  auto deadline = Clock::now() + timeout;
  do {
    ::pollfd pfd{};
    pfd.fd = hnd_;
    pfd.events = POLLIN;

    int res = pollFd(&pfd, 1, static_cast<int>(timeout.count()));
    if (res > 0) {
      return true;
    } else if (res == 0) {
      C10D_WARNING(
          "waitForInput: poll for socket {} returned 0, likely a timeout",
          *this);
      continue;
    }

    std::error_code err = getSocketError();
    if (err == std::errc::operation_in_progress) {
      bool timedout = Clock::now() >= deadline;
      if (timedout) {
        return false;
      }
      C10D_WARNING(
          "waitForInput: poll for socket {} returned operation_in_progress before a timeout",
          *this);
    } else if (err != std::errc::interrupted) {
      C10D_WARNING(
          "waitForInput: poll for socket {} failed with res={}, err={}.",
          *this,
          res,
          err);
      return false;
    }
  } while (Clock::now() < deadline);

  C10D_WARNING(
      "waitForInput: socket {} timed out after {}ms", *this, timeout.count());
  return false;
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
  SocketListenOp(std::uint16_t port, const SocketOptions& opts);

  std::unique_ptr<SocketImpl> run();

 private:
  bool tryListen(int family);

  bool tryListen(const ::addrinfo& addr);

  template <typename... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  void recordError(fmt::string_view format, Args&&... args) {
    auto msg = fmt::vformat(format, fmt::make_format_args(args...));

    C10D_WARNING(msg);

    errors_.emplace_back(std::move(msg));
  }

  std::string port_;
  const SocketOptions* opts_;
  std::vector<std::string> errors_;
  std::unique_ptr<SocketImpl> socket_;
};

SocketListenOp::SocketListenOp(std::uint16_t port, const SocketOptions& opts)
    : port_{fmt::to_string(port)}, opts_{&opts} {}

std::unique_ptr<SocketImpl> SocketListenOp::run() {
  if (opts_->prefer_ipv6()) {
    C10D_DEBUG("The server socket will attempt to listen on an IPv6 address.");
    if (tryListen(AF_INET6)) {
      return std::move(socket_);
    }

    C10D_DEBUG("The server socket will attempt to listen on an IPv4 address.");
    if (tryListen(AF_INET)) {
      return std::move(socket_);
    }
  } else {
    C10D_DEBUG(
        "The server socket will attempt to listen on an IPv4 or IPv6 address.");
    if (tryListen(AF_UNSPEC)) {
      return std::move(socket_);
    }
  }

  constexpr auto* msg =
      "The server socket has failed to listen on any local network address.";

  C10D_ERROR(msg);

  C10D_THROW_ERROR(
      SocketError, fmt::format("{} {}", msg, fmt::join(errors_, " ")));
}

bool SocketListenOp::tryListen(int family) {
  ::addrinfo hints{}, *naked_result = nullptr;

  hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  int r = ::getaddrinfo(nullptr, port_.c_str(), &hints, &naked_result);
  if (r != 0) {
    const char* gai_err = ::gai_strerror(r);

    recordError(
        "The local {}network addresses cannot be retrieved (gai error: {} - {}).",
        family == AF_INET        ? "IPv4 "
            : family == AF_INET6 ? "IPv6 "
                                 : "",
        r,
        gai_err);

    return false;
  }

  addrinfo_ptr result{naked_result};

  for (::addrinfo* addr = naked_result; addr != nullptr; addr = addr->ai_next) {
    C10D_DEBUG("The server socket is attempting to listen on {}.", *addr);
    if (tryListen(*addr)) {
      return true;
    }
  }

  recordError(
      "The server could not be initialized on any address for port={}, family={}",
      port_,
      family);

  return false;
}

bool SocketListenOp::tryListen(const ::addrinfo& addr) {
  SocketImpl::Handle hnd =
      ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  if (hnd == SocketImpl::invalid_socket) {
    C10D_DEBUG(
        "The server socket cannot be initialized on {} {}.",
        addr,
        getSocketError());

    return false;
  }

  socket_ = std::make_unique<SocketImpl>(hnd);

#ifndef _WIN32
  if (!socket_->enableAddressReuse()) {
    C10D_WARNING(
        "The address reuse option cannot be enabled for the server socket on {}.",
        addr);
  }
#endif

#ifdef _WIN32
  // The SO_REUSEADDR flag has a significantly different behavior on Windows
  // compared to Unix-like systems. It allows two or more processes to share
  // the same port simultaneously, which is totally unsafe.
  //
  // Here we follow the recommendation of Microsoft and use the non-standard
  // SO_EXCLUSIVEADDRUSE flag instead.
  if (!socket_->enableExclusiveAddressUse()) {
    C10D_WARNING(
        "The exclusive address use option cannot be enabled for the server socket on {}.",
        addr);
  }
#endif

  // Not all operating systems support dual-stack sockets by default. Since we
  // wish to use our IPv6 socket for IPv4 communication as well, we explicitly
  // ask the system to enable it.
  if (addr.ai_family == AF_INET6 && !socket_->enableDualStack()) {
    C10D_WARNING(
        "The server socket does not support IPv4 communication on {}.", addr);
  }

  if (::bind(socket_->handle(), addr.ai_addr, addr.ai_addrlen) != 0) {
    recordError(
        "The server socket has failed to bind to {} {}.",
        addr,
        getSocketError());

    return false;
  }

  // NOLINTNEXTLINE(bugprone-argument-comment)
  if (::listen(socket_->handle(), -1 /* backlog */) != 0) {
    recordError(
        "The server socket has failed to listen on {} {}.",
        addr,
        getSocketError());

    return false;
  }

  socket_->closeOnExec();

  C10D_INFO("The server socket has started to listen on {}.", addr);

  return true;
}

class SocketListenFromFdOp {
 public:
  SocketListenFromFdOp(int fd, std::uint16_t expected_port);

  std::unique_ptr<SocketImpl> run() const;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int fd_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::uint16_t expected_port_;
};

SocketListenFromFdOp::SocketListenFromFdOp(int fd, std::uint16_t expected_port)
    : fd_(fd), expected_port_(expected_port) {}

std::unique_ptr<SocketImpl> SocketListenFromFdOp::run() const {
  C10D_DEBUG("listenFromFd: fd {}, expected port {}", fd_, expected_port_);

  ::sockaddr_storage addr_storage{};
  ::socklen_t addr_len = sizeof(addr_storage);
  if (::getsockname(
          fd_, reinterpret_cast<::sockaddr*>(&addr_storage), &addr_len) < 0) {
    C10D_THROW_ERROR(
        SocketError,
        fmt::format("getsockname failed for fd {}: {}", fd_, getSocketError()));
  }

  auto socket = std::make_unique<SocketImpl>(fd_);
  const auto port = socket->getPort();

  if (port != expected_port_) {
    C10D_THROW_ERROR(
        SocketError,
        fmt::format(
            "listen fd {} is bound to port {}, expected to be bound to port {}",
            fd_,
            port,
            expected_port_));
  }

  if (::listen(socket->handle(), -1 /* backlog */) != 0) {
    C10D_THROW_ERROR(
        SocketError,
        fmt::format(
            "Failed to listen on socket initialized from fd {}: {}.",
            socket->handle(),
            getSocketError()));
  }

  socket->closeOnExec();

  C10D_INFO(
      "The server has taken over the listening socket with fd {}, address {}",
      fd_,
      *socket);
  return socket;
}

class SocketConnectOp {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::steady_clock::duration;
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  enum class ConnectResult : uint8_t { Success, Error, Retry };

 public:
  SocketConnectOp(
      const std::string& host,
      std::uint16_t port,
      const SocketOptions& opts);

  std::unique_ptr<SocketImpl> run();

 private:
  bool tryConnect(int family);

  ConnectResult tryConnect(const ::addrinfo& addr);

  ConnectResult tryConnectCore(const ::addrinfo& addr);

  [[noreturn]] void throwTimeoutError() const;

  template <typename... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  void recordError(fmt::string_view format, Args&&... args) {
    auto msg = fmt::vformat(format, fmt::make_format_args(args...));

    C10D_WARNING(msg);

    errors_.emplace_back(std::move(msg));
  }

  const char* host_;
  std::string port_;
  const SocketOptions* opts_;
  TimePoint deadline_;
  std::vector<std::string> errors_;
  std::unique_ptr<SocketImpl> socket_;
};

SocketConnectOp::SocketConnectOp(
    const std::string& host,
    std::uint16_t port,
    const SocketOptions& opts)
    : host_{host.c_str()}, port_{fmt::to_string(port)}, opts_{&opts} {}

std::unique_ptr<SocketImpl> SocketConnectOp::run() {
  if (opts_->prefer_ipv6()) {
    C10D_DEBUG(
        "The client socket will attempt to connect to an IPv6 address of ({}, {}).",
        host_,
        port_);

    if (tryConnect(AF_INET6)) {
      return std::move(socket_);
    }

    C10D_DEBUG(
        "The client socket will attempt to connect to an IPv4 address of ({}, {}).",
        host_,
        port_);

    if (tryConnect(AF_INET)) {
      return std::move(socket_);
    }
  } else {
    C10D_DEBUG(
        "The client socket will attempt to connect to an IPv4 or IPv6 address of ({}, {}).",
        host_,
        port_);

    if (tryConnect(AF_UNSPEC)) {
      return std::move(socket_);
    }
  }

  auto msg = fmt::format(
      "The client socket has failed to connect to any network address of ({}, {}).",
      host_,
      port_);

  C10D_ERROR(msg);

  C10D_THROW_ERROR(
      SocketError, fmt::format("{} {}", msg, fmt::join(errors_, " ")));
}

bool SocketConnectOp::tryConnect(int family) {
  ::addrinfo hints{};
  hints.ai_flags = AI_V4MAPPED | AI_ALL | AI_NUMERICSERV;
  hints.ai_family = family;
  hints.ai_socktype = SOCK_STREAM;

  deadline_ = Clock::now() + opts_->connect_timeout();

  bool retry = false;
  do {
    retry = false;

    errors_.clear();

    ::addrinfo* naked_result = nullptr;
    // patternlint-disable cpp-dns-deps
    int r = ::getaddrinfo(host_, port_.c_str(), &hints, &naked_result);
    if (r != 0) {
      const char* gai_err = ::gai_strerror(r);

      recordError(
          "The {}network addresses of ({}, {}) cannot be retrieved (gai error: {} - {}).",
          family == AF_INET        ? "IPv4 "
              : family == AF_INET6 ? "IPv6 "
                                   : "",
          host_,
          port_,
          r,
          gai_err);
      retry = true;
    } else {
      addrinfo_ptr result{naked_result};

      for (::addrinfo* addr = naked_result; addr != nullptr;
           addr = addr->ai_next) {
        C10D_TRACE("The client socket is attempting to connect to {}.", *addr);

        ConnectResult cr = tryConnect(*addr);
        if (cr == ConnectResult::Success) {
          return true;
        }

        if (cr == ConnectResult::Retry) {
          retry = true;
        }
      }
    }

    if (retry) {
      auto connectBackoff = opts_->connect_backoff();
      auto delayDuration = connectBackoff->nextBackoff();

      if (Clock::now() < deadline_ - delayDuration) {
        // Prevent our log output to be too noisy, warn only every 30 seconds.
        static auto lastLog = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if ((now - lastLog) >= std::chrono::seconds(30)) {
          C10D_INFO(
              "No socket on ({}, {}) is listening yet, will retry.",
              host_,
              port_);

          lastLog = now;
        }

        // Wait to avoid choking the server.
        delay(delayDuration);
      } else {
        throwTimeoutError();
      }
    }
  } while (retry);

  return false;
}

SocketConnectOp::ConnectResult SocketConnectOp::tryConnect(
    const ::addrinfo& addr) {
  if (Clock::now() >= deadline_) {
    throwTimeoutError();
  }

  SocketImpl::Handle hnd =
      ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  if (hnd == SocketImpl::invalid_socket) {
    recordError(
        "The client socket cannot be initialized to connect to {} {}.",
        addr,
        getSocketError());

    return ConnectResult::Error;
  }

  socket_ = std::make_unique<SocketImpl>(hnd, addr);

  socket_->enableNonBlocking();

  ConnectResult cr = tryConnectCore(addr);
  if (cr == ConnectResult::Error) {
    std::error_code err = getSocketError();
    if (err == std::errc::interrupted) {
      C10_THROW_ERROR(DistNetworkError, c10::utils::str_error(err.value()));
    }

    // Retry if the server is not yet listening or if its backlog is exhausted.
    if (err == std::errc::connection_refused ||
        err == std::errc::connection_reset) {
      C10D_TRACE(
          "The server socket on {} is not yet listening {}, will retry.",
          addr,
          err);

      return ConnectResult::Retry;
    } else if (err == std::errc::timed_out) {
      C10D_WARNING(
          "The server socket on {} has timed out, will retry.", addr, err);

      return ConnectResult::Retry;
    } else {
      recordError(
          "The client socket has failed to connect to {} {}.", addr, err);

      return ConnectResult::Error;
    }
  }

  socket_->closeOnExec();

  // TODO: Remove once we fully migrate to non-blocking mode.
  socket_->disableNonBlocking();

  C10D_INFO("The client socket has connected to {} on {}.", addr, *socket_);

  if (!socket_->enableNoDelay()) {
    C10D_WARNING(
        "The no-delay option cannot be enabled for the client socket on {}.",
        *socket_);
  }

  return ConnectResult::Success;
}

SocketConnectOp::ConnectResult SocketConnectOp::tryConnectCore(
    const ::addrinfo& addr) {
  int r = ::connect(socket_->handle(), addr.ai_addr, addr.ai_addrlen);
  if (r == 0) {
    return ConnectResult::Success;
  }

  std::error_code err = getSocketError();
  if (err == std::errc::already_connected) {
    return ConnectResult::Success;
  }

  if (err != std::errc::operation_in_progress &&
      err != std::errc::operation_would_block) {
    return ConnectResult::Error;
  }

  Duration remaining = deadline_ - Clock::now();
  if (remaining <= Duration::zero()) {
    throwTimeoutError();
  }

  ::pollfd pfd{};
  pfd.fd = socket_->handle();
  pfd.events = POLLOUT;

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);

  r = pollFd(&pfd, 1, static_cast<int>(ms.count()));
  if (r == 0) {
    throwTimeoutError();
  }
  if (r == -1) {
    return ConnectResult::Error;
  }

  int err_code = 0;

  ::socklen_t err_len = sizeof(int);

  r = getSocketOption(
      socket_->handle(), SOL_SOCKET, SO_ERROR, &err_code, &err_len);
  if (r != 0) {
    return ConnectResult::Error;
  }

  if (err_code != 0) {
    setSocketError(err_code);

    return ConnectResult::Error;
  } else {
    return ConnectResult::Success;
  }
}

void SocketConnectOp::throwTimeoutError() const {
  auto msg = fmt::format(
      "The client socket has timed out after {} while trying to connect to ({}, {}).",
      opts_->connect_timeout(),
      host_,
      port_);

  C10D_ERROR(msg);

  C10D_THROW_ERROR(TimeoutError, msg);
}

} // namespace

void Socket::initialize() {
#ifdef _WIN32
  // All processes that call socket functions on Windows must first initialize
  // the Winsock library.
  static bool init_flag [[maybe_unused]] = []() {
    WSADATA data{};
    if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      C10D_THROW_ERROR(
          SocketError, "The initialization of Winsock has failed.");
    }
    return true;
  }();
#endif
}

Socket Socket::listen(std::uint16_t port, const SocketOptions& opts) {
  SocketListenOp op{port, opts};

  return Socket{op.run()};
}

Socket Socket::listenFromFd(int fd, std::uint16_t expected_port) {
  SocketListenFromFdOp op{fd, expected_port};

  return Socket{op.run()};
}

Socket Socket::connect(
    const std::string& host,
    std::uint16_t port,
    const SocketOptions& opts) {
  SocketConnectOp op{host, port, opts};

  return Socket{op.run()};
}

Socket::Socket(Socket&& other) noexcept = default;

Socket& Socket::operator=(Socket&& other) noexcept = default;

Socket::~Socket() = default;

Socket Socket::accept() const {
  if (impl_) {
    return Socket{impl_->accept()};
  }

  C10D_THROW_ERROR(SocketError, "The socket is not initialized.");
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

bool Socket::waitForInput(std::chrono::milliseconds timeout) {
  return impl_->waitForInput(timeout);
}

std::string Socket::repr() const {
  if (impl_) {
    return fmt::format("{}", *impl_);
  }
  return "Socket(no-impl)";
}

} // namespace c10d::detail
