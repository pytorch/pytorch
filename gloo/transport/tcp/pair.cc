/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/tcp/pair.h"

#include <sstream>

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "gloo/common/logging.h"
#include "gloo/transport/tcp/buffer.h"

namespace gloo {
namespace transport {
namespace tcp {

Pair::Pair(const std::shared_ptr<Device>& dev)
    : dev_(dev),
      state_(INITIALIZING),
      sync_(false),
      busyPoll_(false),
      fd_(-1),
      sendBufferSize_(0) {
  memset(&rx_, 0, sizeof(rx_));
  memset(&tx_, 0, sizeof(tx_));
  listen();
}

Pair::~Pair() {
  // Needs lock so that this doesn't race with read/write of the
  // underlying file descriptor on the device thread.
  std::lock_guard<std::mutex> lock(m_);
  changeState(CLOSED);
}

const Address& Pair::address() const {
  return self_;
}

void Pair::connect(const std::vector<char>& bytes) {
  auto peer = Address(bytes);
  connect(peer);
}

static void setSocketBlocking(int fd, bool enable) {
  auto rv = fcntl(fd, F_GETFL);
  GLOO_ENFORCE_NE(rv, -1);
  if (enable) {
    rv &= ~O_NONBLOCK;
  } else {
    rv |= O_NONBLOCK;
  }
  rv = fcntl(fd, F_SETFL, rv);
  GLOO_ENFORCE_NE(rv, -1);
}

void Pair::setSync(bool sync, bool busyPoll) {
  std::unique_lock<std::mutex> lock(m_);

  GLOO_ENFORCE(sync, "Can only switch to sync mode");

  // Wait for pair to be connected
  while (state_ < CONNECTED) {
    cv_.wait(lock);
  }

  // Unregister from loop and switch socket to blocking mode
  dev_->unregisterDescriptor(fd_);
  setSocketBlocking(fd_, true);

  // If the pair was still flushing a write, finish it.
  if (tx_.buf_ != nullptr) {
    auto rv = write(tx_);
    GLOO_ENFORCE(rv, "Write must always succeed in sync mode");
    tx_.buf_->handleSendCompletion();
    memset(&tx_, 0, sizeof(tx_));
  }

  sync_ = true;
  busyPoll_ = busyPoll;
}

void Pair::listen() {
  std::lock_guard<std::mutex> lock(m_);
  int rv;

  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = dev_->attr_.ai_family;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* result;
  rv = getaddrinfo(dev_->attr_.hostname.data(), nullptr, &hints, &result);
  GLOO_ENFORCE_NE(rv, -1);
  for (auto rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }

    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
    if (rv == -1) {
      close(fd);
      continue;
    }

    // bind(2) successful, keep copy of address
    self_ = Address::fromSockName(fd);

    // listen(2) on socket
    fd_ = fd;
    rv = ::listen(fd_, 1);
    GLOO_ENFORCE_NE(rv, -1, "listen");
    break;
  }

  // Expect listening file descriptor at this point.
  // If there is none, build error message that includes all
  // addresses that we attempted to bind to.
  if (fd_ == -1) {
    std::stringstream err;
    for (auto rp = result; rp != nullptr; rp = rp->ai_next) {
      err << Address(rp->ai_addr, rp->ai_addrlen).str();
      if (rp->ai_next != nullptr) {
        err << ", ";
      }
    }
    GLOO_ENFORCE_NE(fd_, -1, "Attempted to bind to: ", err);
  }

  freeaddrinfo(result);

  // Register with device so we're called when peer connects
  changeState(LISTENING);
  dev_->registerDescriptor(fd_, EPOLLIN, this);

  return;
}

void Pair::connect(const Address& peer) {
  std::unique_lock<std::mutex> lock(m_);
  int rv;
  socklen_t addrlen;

  peer_ = peer;

  // Addresses have to have same family
  GLOO_ENFORCE_EQ(
      self_.ss_.ss_family, peer_.ss_.ss_family, "address family mismatch");

  if (self_.ss_.ss_family == AF_INET) {
    struct sockaddr_in* sa = (struct sockaddr_in*)&self_.ss_;
    struct sockaddr_in* sb = (struct sockaddr_in*)&peer_.ss_;
    addrlen = sizeof(struct sockaddr_in);
    rv = memcmp(&sa->sin_addr, &sb->sin_addr, sizeof(struct in_addr));
    if (rv == 0) {
      rv = sa->sin_port - sb->sin_port;
    }
  } else if (peer_.ss_.ss_family == AF_INET6) {
    struct sockaddr_in6* sa = (struct sockaddr_in6*)&self_.ss_;
    struct sockaddr_in6* sb = (struct sockaddr_in6*)&peer_.ss_;
    addrlen = sizeof(struct sockaddr_in6);
    rv = memcmp(&sa->sin6_addr, &sb->sin6_addr, sizeof(struct in6_addr));
    if (rv == 0) {
      rv = sa->sin6_port - sb->sin6_port;
    }
  } else {
    GLOO_ENFORCE(false, "unknown sa_family");
  }

  GLOO_ENFORCE_NE(rv, 0, "cannot connect to self");

  // self_ < peer_; we are listening side.
  if (rv < 0) {
    while (state_ < CONNECTED) {
      cv_.wait(lock);
    }
    return;
  }

  // self_ > peer_; we are connecting side.
  // First destroy listening socket.
  dev_->unregisterDescriptor(fd_);
  close(fd_);

  // Create new socket to connect to peer.
  fd_ = socket(peer_.ss_.ss_family, SOCK_STREAM | SOCK_NONBLOCK, 0);
  GLOO_ENFORCE_NE(fd_, -1, "socket: ", strerror(errno));

  // Connect to peer
  rv = ::connect(fd_, (struct sockaddr*)&peer_.ss_, addrlen);
  if (rv == -1 && errno != EINPROGRESS) {
    GLOO_ENFORCE_NE(rv, -1, "connect: ", strerror(errno));
  }

  // Register with device so we're called when connection completes.
  changeState(CONNECTING);
  dev_->registerDescriptor(fd_, EPOLLIN | EPOLLOUT, this);

  // Wait for connection to complete
  while (state_ < CONNECTED) {
    cv_.wait(lock);
  }
}

// write is called from:
// 1) the device thread (the handleEvents function)
// 2) a user thread (the send function)
//
// In either case, the lock is held and the write function
// below inherits it.
//
bool Pair::write(Op& op) {
  std::array<struct iovec, 2> iov;
  int ioc = 0;
  int nbytes = 0;

  GLOO_ENFORCE_EQ(state_, CONNECTED);

  // Include preamble if necessary
  if (op.nwritten_ < sizeof(op.preamble_)) {
    iov[ioc].iov_base = ((char*)&op.preamble_) + op.nwritten_;
    iov[ioc].iov_len = sizeof(op.preamble_) - op.nwritten_;
    nbytes += iov[ioc].iov_len;
    ioc++;
  }

  // Include remaining piece of buffer
  int offset = op.preamble_.offset_;
  int length = op.preamble_.length_;
  if (op.nwritten_ > sizeof(op.preamble_)) {
    offset += op.nwritten_ - sizeof(op.preamble_);
    length -= op.nwritten_ - sizeof(op.preamble_);
  }
  iov[ioc].iov_base = ((char*)op.buf_->ptr_) + offset;
  iov[ioc].iov_len = length;
  nbytes += iov[ioc].iov_len;
  ioc++;

  int rv = writev(fd_, iov.data(), ioc);
  if (rv == -1 && errno == EAGAIN) {
    return false;
  }

  GLOO_ENFORCE_NE(rv, -1, "writev: ", strerror(errno));
  op.nwritten_ += rv;
  if (rv < nbytes) {
    return false;
  }

  GLOO_ENFORCE_EQ(rv, nbytes);
  return true;
}

// read is called from:
// 1) the device thread (the handleEvents function).
// 2) a user thread (the recv function) IFF the pair is in sync mode.
//
// In either case, the lock is held and the read function
// below inherits it.
//
bool Pair::read(Op& op) {
  GLOO_ENFORCE_EQ(state_, CONNECTED);

  for (;;) {
    struct iovec iov;

    if (op.nread_ < sizeof(op.preamble_)) {
      // Read preamble
      iov.iov_base = ((char*)&op.preamble_) + op.nread_;
      iov.iov_len = sizeof(op.preamble_) - op.nread_;
    } else {
      // Read payload
      if (op.buf_ == nullptr) {
        op.buf_ = getBuffer(op.preamble_.slot_);
        // Buffer not (yet) registered, leave it for next loop iteration
        if (op.buf_ == nullptr) {
          return false;
        }
      }
      auto offset = op.nread_ - sizeof(op.preamble_);
      iov.iov_base = ((char*)op.buf_->ptr_) + offset;
      iov.iov_len = op.preamble_.length_ - offset;
    }

    // If busy-poll has been requested AND sync mode has been enabled for pair
    // we'll keep spinning calling recv() on socket by supplying MSG_DONTWAIT
    // flag. This is more efficient in terms of latency than allowing the kernel
    // to de-schedule this thread waiting for IO event to happen. The tradeoff
    // is stealing the CPU core just for busy polling.
    int rv = 0;
    for (;;) {
      // Alas, readv does not support flags, so we need to use recv
      rv = ::recv(fd_, iov.iov_base, iov.iov_len, busyPoll_ ? MSG_DONTWAIT : 0);
      if (rv == -1) {
        // EAGAIN happens when there are no more bytes left to read
        if (errno == EAGAIN) {
          if (!sync_) {
            return false;
          }
          // Keep looping on EAGAIN if busy-poll flag has been set
          continue;
        }

        // Retry on EINTR
        if (errno == EINTR) {
          continue;
        }

        // ECONNRESET happens when the remote peer unexpectedly terminates
        if (errno == ECONNRESET) {
          changeState(CLOSED);
          return false;
        }

        // Unexpected error
        GLOO_ENFORCE_EQ(
            errno, 0, "reading from ", peer_.str(), ": ", strerror(errno));
      }
      break;
    }

    // Transition to CLOSED on EOF
    if (rv == 0) {
      changeState(CLOSED);
      return false;
    }

    op.nread_ += rv;

    // Verify the payload is non-empty after reading preamble
    if (op.nread_ == sizeof(op.preamble_)) {
      GLOO_ENFORCE_NE(op.preamble_.length_, 0);
    }

    // Return if op is complete
    if (op.nread_ == sizeof(op.preamble_) + op.preamble_.length_) {
      return true;
    }
  }
}

void Pair::handleEvents(int events) {
  // Try to acquire the pair's lock so the device thread (the thread
  // that ends up calling handleEvents) can mutate the tx and rx op
  // fields of this instance. If the lock cannot be acquired that
  // means some other thread is trying to mutate this pair's state,
  // which in turn might require calling into (and locking) the
  // underlying device (for example, when the pair transitions to the
  // CLOSED state). To avoid deadlocks, attempt to lock the pair and
  // skip handling the events until the next tick if the lock cannot
  // be acquired.
  std::unique_lock<std::mutex> lock(m_, std::try_to_lock);
  if (!lock) {
    return;
  }

  if (state_ == CONNECTED) {
    if (events & EPOLLOUT) {
      GLOO_ENFORCE(
          tx_.buf_ != nullptr,
          "tx_.buf_ cannot be NULL because EPOLLOUT happened");
      if (write(tx_)) {
        tx_.buf_->handleSendCompletion();
        memset(&tx_, 0, sizeof(tx_));
        dev_->registerDescriptor(fd_, EPOLLIN, this);
        cv_.notify_all();
      } else {
        // Write didn't complete, wait for epoll again
      }
    }
    if (events & EPOLLIN) {
      while (read(rx_)) {
        rx_.buf_->handleRecvCompletion();
        memset(&rx_, 0, sizeof(rx_));
      }
    }
    return;
  }

  if (state_ == LISTENING) {
    handleListening();
    return;
  }

  if (state_ == CONNECTING) {
    handleConnecting();
    return;
  }

  GLOO_ENFORCE(false, "Unexpected state: ", state_);
}

void Pair::handleListening() {
  struct sockaddr_storage addr;
  socklen_t addrlen = sizeof(addr);
  int rv;

  rv = accept(fd_, (struct sockaddr*)&addr, &addrlen);
  GLOO_ENFORCE_NE(rv, -1, "accept: ", strerror(errno));

  // Connected, replace file descriptor
  dev_->unregisterDescriptor(fd_);
  close(fd_);
  fd_ = rv;

  // Common connection-made code
  handleConnected();
}

void Pair::handleConnecting() {
  int optval;
  socklen_t optlen = sizeof(optval);
  int rv;

  // Verify that connecting was successful
  rv = getsockopt(fd_, SOL_SOCKET, SO_ERROR, &optval, &optlen);
  GLOO_ENFORCE_NE(rv, -1);
  GLOO_ENFORCE_EQ(optval, 0, "SO_ERROR: ", strerror(optval));

  // Common connection-made code
  handleConnected();
}

void Pair::handleConnected() {
  int rv;

  // Reset addresses
  self_ = Address::fromSockName(fd_);
  peer_ = Address::fromPeerName(fd_);

  // Make sure socket is non-blocking
  setSocketBlocking(fd_, false);

  int flag = 1;
  socklen_t optlen = sizeof(flag);
  rv = setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, (char*)&flag, optlen);
  GLOO_ENFORCE_NE(rv, -1);

  dev_->registerDescriptor(fd_, EPOLLIN, this);
  changeState(CONNECTED);
}

// getBuffer must only be called when holding lock.
Buffer* Pair::getBuffer(int slot) {
  for (;;) {
    auto it = buffers_.find(slot);
    if (it == buffers_.end()) {
      // The remote peer already sent some bytes destined for the
      // buffer at this slot, but this side of the pair hasn't
      // registed it yet.
      //
      // The current strategy is to return and let the the device loop
      // repeatedly call us again until the buffer has been
      // registered. This essentially means busy waiting while
      // yielding to other pairs. This is not a problem as this only
      // happens at initialization time.
      //
      return nullptr;
    }

    return it->second;
  }
}

void Pair::registerBuffer(Buffer* buf) {
  std::lock_guard<std::mutex> lock(m_);
  GLOO_ENFORCE(
      buffers_.find(buf->slot_) == buffers_.end(),
      "duplicate buffer for slot ",
      buf->slot_);
  buffers_[buf->slot_] = buf;
  cv_.notify_all();
}

void Pair::unregisterBuffer(Buffer* buf) {
  std::lock_guard<std::mutex> lock(m_);
  buffers_.erase(buf->slot_);
}

// changeState must only be called when holding lock.
void Pair::changeState(state state) {
  // Clean up file descriptor when transitioning to CLOSED
  if (state_ == CONNECTED && state == CLOSED) {
    if (!sync_) {
      dev_->unregisterDescriptor(fd_);
    }
    close(fd_);
    fd_ = -1;
  }

  state_ = state;
  cv_.notify_all();
}

void Pair::send(Op& op) {
  std::unique_lock<std::mutex> lock(m_);

  // The connect function already wait for the pair to become
  // connected (both in listening and connecting mode).
  // No need to wait again here.
  GLOO_ENFORCE_EQ(
      CONNECTED,
      state_,
      "Pair is not connected (",
      self_.str(),
      " <--> ",
      peer_.str(),
      ")");

  // Try to size the send buffer such that the write below completes
  // synchronously and we don't need to finish the write later.
  auto size = 2 * (sizeof(op.preamble_) + op.preamble_.length_);
  if (sendBufferSize_ < size) {
    int rv;
    int optval = size;
    socklen_t optlen = sizeof(optval);
    rv = setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &optval, optlen);
    GLOO_ENFORCE_NE(rv, -1);
    rv = getsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &optval, &optlen);
    GLOO_ENFORCE_NE(rv, -1);
    sendBufferSize_ = optval;
  }

  // Wait until event loop has finished current write.
  if (!sync_) {
    while (tx_.buf_ != nullptr) {
      cv_.wait(lock);
    }
  }

  // Write to socket
  if (sync_) {
    auto rv = write(op);
    GLOO_ENFORCE(rv, "Write must always succeed in sync mode");
    op.buf_->handleSendCompletion();
  } else {
    // Write immediately without checking socket for writeability.
    if (write(op)) {
      op.buf_->handleSendCompletion();
      return;
    }

    // Write didn't complete; pass to event loop
    tx_ = op;
    dev_->registerDescriptor(fd_, EPOLLIN | EPOLLOUT, this);
  }
}

void Pair::recv() {
  std::unique_lock<std::mutex> lock(m_);

  auto rv = read(rx_);
  GLOO_ENFORCE(rv, "Read must always succeed in sync mode");
  rx_.buf_->handleRecvCompletion();
  memset(&rx_, 0, sizeof(rx_));
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createSendBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

std::unique_ptr<::gloo::transport::Buffer>
Pair::createRecvBuffer(int slot, void* ptr, size_t size) {
  auto buffer = new Buffer(this, slot, ptr, size);
  registerBuffer(buffer);
  return std::unique_ptr<::gloo::transport::Buffer>(buffer);
}

} // namespace tcp
} // namespace transport
} // namespace gloo
