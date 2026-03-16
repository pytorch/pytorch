#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d::openreg {

// Fixed-size header preceding every message on the wire.
// Layout: [tag:4][dtype:1][numel:8] = 13 bytes
#pragma pack(push, 1)
struct WireHeader {
  uint32_t tag;
  uint8_t dtype;
  uint64_t numel;
};
#pragma pack(pop)
static_assert(sizeof(WireHeader) == 13, "WireHeader must be 13 bytes");

// TCP transport layer for OCCL.
//
// Manages a full mesh of TCP connections between ranks, with lazy
// connection establishment. Each rank listens on a port and publishes
// its address via the c10d Store; connections to peers are established
// on first send/recv.
//
// This is the component vendors replace with their hardware-specific
// transport.
class OcclTransport {
 public:
  OcclTransport(
      int rank,
      int worldSize,
      c10::intrusive_ptr<::c10d::Store> store);

  ~OcclTransport();

  OcclTransport(const OcclTransport&) = delete;
  OcclTransport& operator=(const OcclTransport&) = delete;

  // Send tagged data to a peer rank.
  // The caller provides the raw buffer, its size in bytes, the scalar
  // type (for validation on the receiver side), element count, and a
  // tag for matching send/recv pairs.
  void send(
      const void* data,
      size_t nbytes,
      int dstRank,
      uint8_t dtype,
      uint64_t numel,
      uint32_t tag);

  // Receive tagged data from a peer rank.
  // Validates that the incoming message's dtype and numel match the
  // expected values. Blocks until the message arrives.
  void recv(
      void* data,
      size_t nbytes,
      int srcRank,
      uint8_t dtype,
      uint64_t numel,
      uint32_t tag);

  int rank() const {
    return rank_;
  }
  int worldSize() const {
    return worldSize_;
  }

 private:
  struct Connection {
    std::string host;
    int port = 0;
    int fd = -1;
    std::mutex mutex;
    std::condition_variable cv;
  };

  // Ensure we have a connection to the given peer, establishing one
  // lazily if needed.
  void ensureConnected(int peerRank);

  // Publish this rank's listen address to the Store so peers can
  // connect.
  void publishAddress();

  // Retrieve a peer's listen address from the Store.
  std::string getPeerAddress(int peerRank);

  // Background thread: accepts incoming connections from peers.
  void acceptLoop();

  // Low-level helpers for exact-count socket I/O.
  static void sendAll(int fd, const void* buf, size_t len);
  static void recvAll(int fd, void* buf, size_t len);

  int rank_;
  int worldSize_;
  c10::intrusive_ptr<::c10d::Store> store_;

  int listenFd_ = -1;
  int listenPort_ = 0;

  std::vector<Connection> peers_;
  std::thread acceptThread_;
  std::atomic<bool> stop_{false};
};

} // namespace c10d::openreg
