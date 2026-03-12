#include "OcclTransport.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <sstream>

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

namespace c10d::openreg {

OcclTransport::OcclTransport(
    int rank,
    int worldSize,
    c10::intrusive_ptr<::c10d::Store> store)
    : rank_(rank),
      worldSize_(worldSize),
      store_(c10::make_intrusive<::c10d::PrefixStore>("occl", std::move(store))),
      peers_(static_cast<size_t>(worldSize)) {
  TORCH_CHECK(rank >= 0 && rank < worldSize, "Invalid rank: ", rank);
  TORCH_CHECK(worldSize > 0, "Invalid worldSize: ", worldSize);

  // TODO: Create listening socket, publish address, start accept thread
}

OcclTransport::~OcclTransport() {
  stop_.store(true);

  // Close listening socket to unblock accept()
  if (listenFd_ >= 0) {
    ::close(listenFd_);
    listenFd_ = -1;
  }

  if (acceptThread_.joinable()) {
    acceptThread_.join();
  }

  // Close all peer connections
  for (auto& conn : peers_) {
    if (conn.fd >= 0) {
      ::close(conn.fd);
      conn.fd = -1;
    }
  }
}

void OcclTransport::send(
    const void* data,
    size_t nbytes,
    int dstRank,
    uint8_t dtype,
    uint64_t numel,
    uint32_t tag) {
  TORCH_CHECK(false, "OcclTransport::send not yet implemented");
}

void OcclTransport::recv(
    void* data,
    size_t nbytes,
    int srcRank,
    uint8_t dtype,
    uint64_t numel,
    uint32_t tag) {
  TORCH_CHECK(false, "OcclTransport::recv not yet implemented");
}

void OcclTransport::ensureConnected(int peerRank) {
  TORCH_CHECK(false, "OcclTransport::ensureConnected not yet implemented");
}

void OcclTransport::publishAddress() {
  TORCH_CHECK(false, "OcclTransport::publishAddress not yet implemented");
}

std::string OcclTransport::getPeerAddress(int peerRank) {
  TORCH_CHECK(false, "OcclTransport::getPeerAddress not yet implemented");
}

void OcclTransport::acceptLoop() {
  // TODO: Accept incoming connections from peers
}

void OcclTransport::sendAll(int fd, const void* buf, size_t len) {
  TORCH_CHECK(false, "OcclTransport::sendAll not yet implemented");
}

void OcclTransport::recvAll(int fd, void* buf, size_t len) {
  TORCH_CHECK(false, "OcclTransport::recvAll not yet implemented");
}

} // namespace c10d::openreg
