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

  // Create listening socket on ephemeral port
  listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  TORCH_CHECK(listenFd_ >= 0, "Failed to create listen socket: ", strerror(errno));

  int optval = 1;
  ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0; // ephemeral port
  TORCH_CHECK(
      ::bind(listenFd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0,
      "Failed to bind listen socket: ", strerror(errno));

  TORCH_CHECK(
      ::listen(listenFd_, worldSize) == 0,
      "Failed to listen: ", strerror(errno));

  // Learn the assigned port
  socklen_t addrLen = sizeof(addr);
  TORCH_CHECK(
      ::getsockname(listenFd_, reinterpret_cast<struct sockaddr*>(&addr), &addrLen) == 0,
      "Failed to get listen port: ", strerror(errno));
  listenPort_ = ntohs(addr.sin_port);

  // Publish our address and read peers' addresses
  publishAddress();
  for (int i = 0; i < worldSize_; ++i) {
    if (i == rank_) {
      continue;
    }
    std::string peerAddr = getPeerAddress(i);
    auto colonPos = peerAddr.rfind(':');
    TORCH_CHECK(colonPos != std::string::npos, "Malformed peer address: ", peerAddr);
    peers_[i].host = peerAddr.substr(0, colonPos);
    peers_[i].port = std::stoi(peerAddr.substr(colonPos + 1));
  }

  // TODO: Start accept thread
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
  // Use localhost — OCCL is single-machine only
  std::string addr = "127.0.0.1:" + std::to_string(listenPort_);
  std::vector<uint8_t> data(addr.begin(), addr.end());
  store_->set("addr/" + std::to_string(rank_), data);
}

std::string OcclTransport::getPeerAddress(int peerRank) {
  // Blocks until the peer has published its address
  auto data = store_->get("addr/" + std::to_string(peerRank));
  return std::string(data.begin(), data.end());
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
