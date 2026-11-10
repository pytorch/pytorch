#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

struct sockaddr;

namespace c10d::detail {

// Restricts TCPStore server connections to loopback peers, peers whose address
// equals the server's own address on the connection (same-host clients that
// use its hostname) and peers whose IP address belongs to a host listed in the
// file named by TORCH_TCPSTORE_ALLOWED_HOSTS_FILE (one hostname or IP literal
// per line, '#' starts a comment). Hostnames are resolved to all of their IPv4
// and IPv6 addresses.
//
// A peer that misses is held, unread, until a refresh that starts after it
// arrived completes, so hosts added to the file later are admitted. Refreshes
// re-read the file and resolve every name serially on a detached thread, at
// most once per refresh interval, so they never block the server loop or its
// destruction. While a refresh has run longer than the hold timeout (the store
// timeout, capped at kMaxHoldTimeout), e.g. because the file or DNS hangs,
// held and new unlisted peers are rejected at once. A peer is thus held for at
// most about the refresh interval plus twice the hold timeout. Only misses
// trigger refreshes, so removals take effect on the next miss. A failed
// refresh keeps the previous addresses, as does a host that fails to resolve.
//
// A rejected peer is sent rejectionMessage(), which the client reads in place
// of its PING reply. The server then reads and discards until the peer closes
// (or kRejectTimeout passes) before closing, so it never resets a connection
// with unread data, which could discard the message.
//
// This only filters by source IP; it does not authenticate peers.
//
// Not thread-safe: use only from the server's event loop thread.
class TCPStoreAllowList {
 public:
  enum class Decision { kAllow, kReject, kDefer };
  using Addr = std::array<uint8_t, 16>;
  // State of a held peer; value-initialize for a new peer.
  struct Ticket {
    // Refresh the peer waits for; 0 for a new peer.
    uint64_t refresh{0};
  };

  // Interval at which servers call poll() while peers are held or rejected
  // connections are open.
  static constexpr std::chrono::milliseconds kPollInterval{10};
  // Maximum time a rejected connection stays open.
  static constexpr std::chrono::seconds kRejectTimeout{5};
  // Maximum refresh duration before held peers are rejected.
  static constexpr std::chrono::seconds kMaxHoldTimeout{30};

  // Returns nullptr if TORCH_TCPSTORE_ALLOWED_HOSTS_FILE is unset. Throws if
  // the file cannot be read. `timeout` is the store timeout.
  static std::unique_ptr<TCPStoreAllowList> fromEnv(
      std::chrono::milliseconds timeout);

  TCPStoreAllowList(std::string path, std::chrono::milliseconds timeout);

  // `local` is the server's address on the connection. On kDefer, call again
  // with the same ticket after poll() until the decision is final.
  Decision check(
      const struct ::sockaddr* peer,
      const struct ::sockaddr* local,
      Ticket& ticket);

  // Non-blocking: applies a completed refresh and starts a due one.
  void poll();

  // Rejection marker followed by the peer address in sendVector format.
  static std::vector<uint8_t> rejectionMessage(const struct ::sockaddr* peer);

 private:
  struct Host {
    std::set<Addr> addrs;
    bool resolved{false};
  };
  using Hosts = std::map<std::string, Host>;

  static Hosts load(const std::string& path, const Hosts& previous);
  void apply(Hosts hosts);
  void startRefresh();
  bool contains(const struct ::sockaddr* peer, const struct ::sockaddr* local)
      const;
  void logRejection(const struct ::sockaddr* peer, bool stalled);

  const std::string path_;
  const std::chrono::milliseconds holdTimeout_;
  Hosts hosts_;
  std::set<Addr> addrs_;
  std::future<Hosts> pending_;
  uint64_t started_{0};
  uint64_t completed_{0};
  bool wanted_{false};
  bool refreshFailed_{false};
  bool refreshStalled_{false};
  std::chrono::steady_clock::time_point lastRefresh_;
  std::chrono::steady_clock::time_point lastWarning_;
  size_t rejected_{0};
};

} // namespace c10d::detail
