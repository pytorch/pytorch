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

  acceptThread_ = std::thread(&OcclTransport::acceptLoop, this);
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

static void setSocketTimeout(int fd, int timeoutSec) {
  struct timeval tv{};
  tv.tv_sec = timeoutSec;
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

static constexpr int kDataTimeoutSec = 300; // 5 minutes

void OcclTransport::send(
    const void* data,
    size_t nbytes,
    int dstRank,
    uint8_t dtype,
    uint64_t numel,
    uint32_t tag) {
  TORCH_CHECK(
      dstRank >= 0 && dstRank < worldSize_ && dstRank != rank_,
      "Invalid dstRank: ", dstRank);

  ensureConnected(dstRank);

  auto& conn = peers_[dstRank];
  std::lock_guard<std::mutex> lock(conn.mutex);

  setSocketTimeout(conn.fd, kDataTimeoutSec);

  WireHeader hdr{};
  hdr.tag = tag;
  hdr.dtype = dtype;
  hdr.numel = numel;
  sendAll(conn.fd, &hdr, sizeof(hdr));

  if (nbytes > 0) {
    sendAll(conn.fd, data, nbytes);
  }
}

void OcclTransport::recv(
    void* data,
    size_t nbytes,
    int srcRank,
    uint8_t dtype,
    uint64_t numel,
    uint32_t tag) {
  TORCH_CHECK(
      srcRank >= 0 && srcRank < worldSize_ && srcRank != rank_,
      "Invalid srcRank: ", srcRank);

  ensureConnected(srcRank);

  auto& conn = peers_[srcRank];
  std::lock_guard<std::mutex> lock(conn.mutex);

  setSocketTimeout(conn.fd, kDataTimeoutSec);

  WireHeader hdr{};
  recvAll(conn.fd, &hdr, sizeof(hdr));

  TORCH_CHECK(
      hdr.tag == tag,
      "Tag mismatch from rank ", srcRank,
      ": expected ", tag, " but got ", hdr.tag);
  TORCH_CHECK(
      hdr.dtype == dtype,
      "Dtype mismatch from rank ", srcRank,
      ": expected ", static_cast<int>(dtype),
      " but got ", static_cast<int>(hdr.dtype));
  TORCH_CHECK(
      hdr.numel == numel,
      "Numel mismatch from rank ", srcRank,
      ": expected ", numel, " but got ", hdr.numel);

  if (nbytes > 0) {
    recvAll(conn.fd, data, nbytes);
  }
}

void OcclTransport::ensureConnected(int peerRank) {
  auto& conn = peers_[peerRank];
  std::unique_lock<std::mutex> lock(conn.mutex);

  if (conn.fd >= 0) {
    return;
  }

  if (rank_ < peerRank) {
    // Lower rank initiates the connection
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    TORCH_CHECK(fd >= 0, "Failed to create socket: ", strerror(errno));

    int optval = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(conn.port));
    TORCH_CHECK(
        ::inet_pton(AF_INET, conn.host.c_str(), &addr.sin_addr) == 1,
        "Invalid peer address: ", conn.host);

    TORCH_CHECK(
        ::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0,
        "Failed to connect to rank ", peerRank, ": ", strerror(errno));

    // Handshake: send our rank so the peer knows who connected
    int32_t myRank = rank_;
    sendAll(fd, &myRank, sizeof(myRank));

    conn.fd = fd;
  } else {
    // Higher rank waits for the accept thread to fill in the fd
    conn.cv.wait(lock, [&conn] { return conn.fd >= 0; });
  }
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
  // Set a timeout on the listening socket so we can check stop_ periodically
  struct timeval tv{};
  tv.tv_sec = 1;
  ::setsockopt(listenFd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  while (!stop_.load()) {
    struct sockaddr_in peerAddr{};
    socklen_t peerAddrLen = sizeof(peerAddr);
    int fd = ::accept(listenFd_, reinterpret_cast<struct sockaddr*>(&peerAddr), &peerAddrLen);

    if (fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
        continue; // timeout or interrupt — check stop_ and retry
      }
      if (stop_.load()) {
        break; // listenFd_ was closed during shutdown
      }
      break;
    }

    int optval = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &optval, sizeof(optval));

    // Read handshake: the connecting rank's identity
    int32_t connectingRank = -1;
    recvAll(fd, &connectingRank, sizeof(connectingRank));

    TORCH_CHECK(
        connectingRank >= 0 && connectingRank < worldSize_,
        "Invalid connecting rank in handshake: ", connectingRank);

    auto& conn = peers_[connectingRank];
    {
      std::lock_guard<std::mutex> lock(conn.mutex);
      conn.fd = fd;
    }
    conn.cv.notify_one();
  }
}

void OcclTransport::sendAll(int fd, const void* buf, size_t len) {
  auto ptr = static_cast<const char*>(buf);
  size_t sent = 0;
  while (sent < len) {
    auto n = ::send(fd, ptr + sent, len - sent, 0);
    if (n <= 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        C10_THROW_ERROR(DistBackendError, "OCCL send timed out");
      }
      C10_THROW_ERROR(
          DistBackendError,
          std::string("OCCL send failed: ") + strerror(errno));
    }
    sent += static_cast<size_t>(n);
  }
}

void OcclTransport::recvAll(int fd, void* buf, size_t len) {
  auto ptr = static_cast<char*>(buf);
  size_t received = 0;
  while (received < len) {
    auto n = ::recv(fd, ptr + received, len - received, 0);
    if (n <= 0) {
      if (n == 0) {
        C10_THROW_ERROR(DistBackendError, "OCCL recv: peer closed connection");
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        C10_THROW_ERROR(DistBackendError, "OCCL recv timed out");
      }
      C10_THROW_ERROR(
          DistBackendError,
          std::string("OCCL recv failed: ") + strerror(errno));
    }
    received += static_cast<size_t>(n);
  }
}

} // namespace c10d::openreg
