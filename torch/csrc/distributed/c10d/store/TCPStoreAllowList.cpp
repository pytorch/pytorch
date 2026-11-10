#include <torch/csrc/distributed/c10d/store/TCPStoreAllowList.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <optional>
#include <thread>

#include <c10/util/env.h>
#include <c10/util/thread_name.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/exception.h>
#include <torch/csrc/distributed/c10d/logging.h>
#include <torch/csrc/distributed/c10d/store/TCPStoreBackend.hpp>

#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <netinet/in.h>
#endif

namespace c10d::detail {
namespace {

constexpr const char* kAllowedHostsFileEnv =
    "TORCH_TCPSTORE_ALLOWED_HOSTS_FILE";

// Minimum interval between re-resolutions and between rejection warnings.
constexpr auto kRefreshInterval = std::chrono::seconds(5);

using Addr = TCPStoreAllowList::Addr;

// IPv4 addresses are stored IPv4-mapped so they match both IPv4 peers and
// IPv4-mapped peers of a dual-stack listener.
std::optional<Addr> toAddr(const struct ::sockaddr* sa) {
  Addr addr{};
  if (sa->sa_family == AF_INET) {
    addr[10] = addr[11] = 0xff;
    std::memcpy(
        &addr[12], &reinterpret_cast<const sockaddr_in*>(sa)->sin_addr, 4);
    return addr;
  }
  if (sa->sa_family == AF_INET6) {
    std::memcpy(
        addr.data(), &reinterpret_cast<const sockaddr_in6*>(sa)->sin6_addr, 16);
    return addr;
  }
  return std::nullopt;
}

bool isV4Mapped(const Addr& addr) {
  constexpr std::array<uint8_t, 12> kPrefix{
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
  return std::equal(kPrefix.begin(), kPrefix.end(), addr.begin());
}

bool isLoopback(const Addr& addr) {
  constexpr Addr kIpv6Loopback{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  return addr == kIpv6Loopback || (isV4Mapped(addr) && addr[12] == 127);
}

// Formats IPv4-mapped addresses as IPv4.
std::string formatAddr(const struct ::sockaddr* sa) {
  ::sockaddr_in v4{};
  auto addr = toAddr(sa);
  if (sa->sa_family == AF_INET6 && addr && isV4Mapped(*addr)) {
    v4.sin_family = AF_INET;
    std::memcpy(&v4.sin_addr, &(*addr)[12], 4);
    sa = reinterpret_cast<const ::sockaddr*>(&v4);
  }
  socklen_t len =
      sa->sa_family == AF_INET6 ? sizeof(sockaddr_in6) : sizeof(sockaddr_in);
  char host[NI_MAXHOST]; // NOLINT
  if (::getnameinfo(sa, len, host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST)) {
    return "?UNKNOWN?";
  }
  return host;
}

} // namespace

// Resolves every name serially, so a refresh takes the sum of the lookups; held
// peers are rejected once that exceeds the hold timeout.
TCPStoreAllowList::Hosts TCPStoreAllowList::load(
    const std::string& path,
    const Hosts& previous) {
  std::ifstream file(path);
  C10D_CHECK_WITH(
      DistStoreError,
      file.is_open(),
      "Failed to read TCPStore allowed hosts file ",
      kAllowedHostsFileEnv,
      "=",
      path);

  Hosts hosts;
  std::string line;
  while (std::getline(file, line)) {
    line = line.substr(0, line.find('#'));
    auto begin = line.find_first_not_of(" \t\r");
    if (begin == std::string::npos) {
      continue;
    }
    auto name = line.substr(begin, line.find_last_not_of(" \t\r") - begin + 1);
    if (hosts.contains(name)) {
      continue;
    }
    auto& host = hosts[name];

    ::addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    ::addrinfo* result = nullptr;
    if (int err = ::getaddrinfo(name.c_str(), nullptr, &hints, &result)) {
      // Keep the previous addresses and only warn when the status changes.
      auto it = previous.find(name);
      if (it != previous.end()) {
        host.addrs = it->second.addrs;
      }
      if (it == previous.end() || it->second.resolved) {
        C10D_WARNING(
            "TCPStore allow list: cannot resolve {}: {}; {}.",
            name,
            gai_strerror(err),
            host.addrs.empty() ? "skipping it"
                               : "keeping its previous addresses");
      }
      continue;
    }
    host.resolved = true;
    for (auto* ai = result; ai != nullptr; ai = ai->ai_next) {
      if (auto addr = toAddr(ai->ai_addr)) {
        host.addrs.insert(*addr);
      }
    }
    ::freeaddrinfo(result);
  }
  return hosts;
}

std::unique_ptr<TCPStoreAllowList> TCPStoreAllowList::fromEnv(
    std::chrono::milliseconds timeout) {
  auto path = c10::utils::get_env(kAllowedHostsFileEnv);
  if (!path.has_value() || path->empty()) {
    return nullptr;
  }
  return std::make_unique<TCPStoreAllowList>(std::move(*path), timeout);
}

TCPStoreAllowList::TCPStoreAllowList(
    std::string path,
    std::chrono::milliseconds timeout)
    : path_{std::move(path)},
      holdTimeout_{
          timeout > std::chrono::milliseconds::zero()
              ? std::min<std::chrono::milliseconds>(timeout, kMaxHoldTimeout)
              : kMaxHoldTimeout},
      lastRefresh_{std::chrono::steady_clock::now() - kRefreshInterval},
      lastWarning_{lastRefresh_} {
  apply(load(path_, {}));
}

void TCPStoreAllowList::apply(Hosts hosts) {
  std::set<Addr> addrs;
  for (const auto& [name, host] : hosts) {
    addrs.insert(host.addrs.begin(), host.addrs.end());
  }
  if (addrs.empty() && (!addrs_.empty() || started_ == 0)) {
    C10D_WARNING(
        "TCPStore allow list {} resolved to no addresses; only loopback and same-host connections will be accepted.",
        path_);
  }
  hosts_ = std::move(hosts);
  addrs_ = std::move(addrs);
}

bool TCPStoreAllowList::contains(
    const struct ::sockaddr* peer,
    const struct ::sockaddr* local) const {
  auto addr = toAddr(peer);
  if (!addr) {
    return false;
  }
  if (isLoopback(*addr) || addrs_.contains(*addr)) {
    return true;
  }
  auto localAddr = toAddr(local);
  return localAddr.has_value() && *localAddr == *addr;
}

TCPStoreAllowList::Decision TCPStoreAllowList::check(
    const struct ::sockaddr* peer,
    const struct ::sockaddr* local,
    Ticket& ticket) {
  poll();
  if (contains(peer, local)) {
    return Decision::kAllow;
  }
  if (ticket.refresh == 0) {
    // Wait for a refresh that starts after this peer arrived.
    ticket.refresh = started_ + 1;
    wanted_ = true;
    poll();
  }
  if (!refreshStalled_ && ticket.refresh > completed_) {
    return Decision::kDefer;
  }
  logRejection(peer, refreshStalled_);
  return Decision::kReject;
}

void TCPStoreAllowList::poll() {
  if (pending_.valid() &&
      pending_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    try {
      apply(pending_.get());
      refreshFailed_ = false;
    } catch (const std::exception& e) {
      if (!refreshFailed_) {
        C10D_WARNING(
            "TCPStore allow list refresh failed; keeping previous addresses: {}",
            e.what());
      }
      refreshFailed_ = true;
    }
    completed_ = started_;
    refreshStalled_ = false;
  }
  auto now = std::chrono::steady_clock::now();
  if (pending_.valid() && !refreshStalled_ &&
      now - lastRefresh_ >= holdTimeout_) {
    C10D_WARNING(
        "TCPStore allow list refresh of {} has not completed after {}ms; rejecting unlisted connections until it does.",
        path_,
        holdTimeout_.count());
    refreshStalled_ = true;
  }
  if (wanted_ && !pending_.valid() && now - lastRefresh_ >= kRefreshInterval) {
    startRefresh();
  }
}

void TCPStoreAllowList::startRefresh() {
  ++started_;
  wanted_ = false;
  lastRefresh_ = std::chrono::steady_clock::now();
  std::promise<Hosts> promise;
  auto future = promise.get_future();
  try {
    // Detached so that a hung read or resolution does not block destruction.
    // Captures no `this`: it may outlive the allow list and run during process
    // exit.
    std::thread([promise = std::move(promise),
                 path = path_,
                 hosts = hosts_]() mutable {
      c10::setThreadName("pt_tcpstore_dns");
      try {
        promise.set_value(load(path, hosts));
      } catch (...) {
        promise.set_exception(std::current_exception());
      }
    }).detach();
  } catch (const std::exception& e) {
    C10D_WARNING("TCPStore allow list refresh failed to start: {}", e.what());
    completed_ = started_;
    return;
  }
  pending_ = std::move(future);
}

void TCPStoreAllowList::logRejection(
    const struct ::sockaddr* peer,
    bool stalled) {
  ++rejected_;
  auto now = std::chrono::steady_clock::now();
  if (now - lastWarning_ < kRefreshInterval) {
    return;
  }
  auto reason = stalled
      ? fmt::format(
            "the refresh of {}={} timed out", kAllowedHostsFileEnv, path_)
      : fmt::format("not in {}={}", kAllowedHostsFileEnv, path_);
  C10D_WARNING(
      "TCPStore rejected connection from {}: {} ({} rejected since last warning).",
      formatAddr(peer),
      reason,
      rejected_);
  lastWarning_ = now;
  rejected_ = 0;
}

std::vector<uint8_t> TCPStoreAllowList::rejectionMessage(
    const struct ::sockaddr* peer) {
  auto addr = formatAddr(peer);
  uint64_t size = addr.size();
  std::vector<uint8_t> msg(sizeof(rejectedMagicNumber) + sizeof(size));
  std::memcpy(msg.data(), &rejectedMagicNumber, sizeof(rejectedMagicNumber));
  std::memcpy(msg.data() + sizeof(rejectedMagicNumber), &size, sizeof(size));
  msg.insert(msg.end(), addr.begin(), addr.end());
  return msg;
}

} // namespace c10d::detail
