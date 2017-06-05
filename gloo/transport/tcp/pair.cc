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
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "gloo/common/error.h"
#include "gloo/common/logging.h"
#include "gloo/transport/tcp/buffer.h"

#define FD_INVALID (-1)

namespace gloo {
namespace transport {
namespace tcp {

Pair::Pair(const std::shared_ptr<Device>& dev)
    : dev_(dev),
      state_(INITIALIZING),
      sync_(false),
      timeout_(dev_->getTimeout()),
      busyPoll_(false),
      fd_(FD_INVALID),
      sendBufferSize_(0),
      ex_(nullptr) {
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

  if (!sync) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("Can only switch to sync mode");
  }

  // Wait for pair to be connected. No need to wait for timeout here. If
  // necessary, the connect path will timeout and signal this thread.
  waitUntilConnected(lock, false);
  if (state_ == CLOSED) {
    signalIoFailure(
        GLOO_ERROR_MSG("Socket unexpectedly closed ", peer_.str()));
  }

  if (!sync_) {
    // If async, unregister from loop and switch socket to blocking mode
    dev_->unregisterDescriptor(fd_);
    setSocketBlocking(fd_, true);

    // If the pair was still flushing a write, finish it.
    if (tx_.buf_ != nullptr) {
      auto rv = write(tx_);
      GLOO_ENFORCE(rv, "Write must always succeed in sync mode");
      tx_.buf_->handleSendCompletion();
      memset(&tx_, 0, sizeof(tx_));
    }
  }

  sync_ = true;
  busyPoll_ = busyPoll;
}

void Pair::listen() {
  std::lock_guard<std::mutex> lock(m_);
  int rv;

  const auto& attr = dev_->attr_;
  auto fd = socket(attr.ai_family, attr.ai_socktype, attr.ai_protocol);
  if (fd == -1) {
    signalIoFailure(GLOO_ERROR_MSG("socket: ", strerror(errno)));
  }

  // Set SO_REUSEADDR to signal that reuse of the listening port is OK.
  int on = 1;
  rv = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  if (rv == -1) {
    close(fd);
    signalIoFailure(GLOO_ERROR_MSG("setsockopt: ", strerror(errno)));
  }

  rv = bind(fd, (const sockaddr*)&attr.ai_addr, attr.ai_addrlen);
  if (rv == -1) {
    close(fd);
    signalIoFailure(GLOO_ERROR_MSG("bind: ", strerror(errno)));
  }

  // listen(2) on socket
  fd_ = fd;
  rv = ::listen(fd_, 1);
  if (rv == -1) {
    close(fd_);
    fd_ = FD_INVALID;
    signalIoFailure(GLOO_ERROR_MSG("listen: ", strerror(errno)));
  }

  // Keep copy of address
  self_ = Address::fromSockName(fd);

  // Register with device so we're called when peer connects
  changeState(LISTENING);
  dev_->registerDescriptor(fd_, EPOLLIN, this);

  return;
}

void Pair::connect(const Address& peer) {
  std::unique_lock<std::mutex> lock(m_);
  int rv;
  socklen_t addrlen;
  checkErrorState();

  peer_ = peer;

  // Addresses have to have same family
  if (self_.ss_.ss_family != peer_.ss_.ss_family) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("address family mismatch");
  }

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
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("unknown sa_family");
  }

  if (rv == 0) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("cannot connect to self");
  }

  // self_ < peer_; we are listening side.
  if (rv < 0) {
    waitUntilConnected(lock, true);
    return;
  }

  // self_ > peer_; we are connecting side.
  // First destroy listening socket.
  dev_->unregisterDescriptor(fd_);
  close(fd_);

  // Create new socket to connect to peer.
  fd_ = socket(peer_.ss_.ss_family, SOCK_STREAM | SOCK_NONBLOCK, 0);
  if (fd_ == -1) {
    signalIoFailure(GLOO_ERROR_MSG("socket: ", strerror(errno)));
  }

  // Set SO_REUSEADDR to signal that reuse of the source port is OK.
  int on = 1;
  rv = setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  if (rv == -1) {
    close(fd_);
    fd_ = FD_INVALID;
    signalIoFailure(GLOO_ERROR_MSG("setsockopt: ", strerror(errno)));
  }

  // Connect to peer
  rv = ::connect(fd_, (struct sockaddr*)&peer_.ss_, addrlen);
  if (rv == -1 && errno != EINPROGRESS) {
    close(fd_);
    fd_ = FD_INVALID;
    signalIoFailure(GLOO_ERROR_MSG("connect: ", strerror(errno)));
  }

  // Register with device so we're called when connection completes.
  changeState(CONNECTING);
  dev_->registerDescriptor(fd_, EPOLLIN | EPOLLOUT, this);

  // Wait for connection to complete
  waitUntilConnected(lock, true);
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
  int ioc;
  int nbytes;
  int rv;

  verifyConnected();

  for (;;) {
    ioc = 0;
    nbytes = 0;

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

    // Write
    rv = writev(fd_, iov.data(), ioc);
    if (rv == -1) {
      if (errno == EAGAIN) {
        if (sync_) {
          // Blocking call returning with EAGAIN indicates timeout
          signalIoFailure(GLOO_ERROR_MSG("Write timeout ", peer_.str()));
        } else {
          // Async. This write is done.
          return false;
        }
      }

      // Retry on EINTR
      if (errno == EINTR) {
        continue;
      }

      // Unexpected error
      signalIoFailure(
          GLOO_ERROR_MSG("writev ", peer_.str(), ": ", strerror(errno)));
    }

    // From write(2) man page (NOTES section):
    //
    //  If a write() is interrupted by a signal handler before any
    //  bytes are written, then the call fails with the error EINTR;
    //  if it is interrupted after at least one byte has been written,
    //  the call succeeds, and returns the number of bytes written.
    //
    // If rv < nbytes we ALWAYS retry, regardless of sync/async mode,
    // since an EINTR may or may not have happened. If this was not
    // the case, and the kernel buffer is full, the next call to
    // write(2) will return EAGAIN, which is handled appropriately.
    op.nwritten_ += rv;
    if (rv < nbytes) {
      continue;
    }

    GLOO_ENFORCE_EQ(rv, nbytes);
    break;
  }

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
  verifyConnected();

  auto start = std::chrono::steady_clock::now();

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
      iov.iov_base = ((char*)op.buf_->ptr_) + offset + op.preamble_.roffset_;
      iov.iov_len = op.preamble_.length_ - offset;

      // There must always be a non-zero number of bytes to read
      GLOO_ENFORCE_GT(iov.iov_len, 0);

      // Bytes read must be in bounds for target buffer
      GLOO_ENFORCE_LE(
          op.preamble_.roffset_ + op.preamble_.length_,
          op.buf_->size_);
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
        // EAGAIN happens when (1) non-blocking and there are no more bytes left
        // to read or (2) blocking and timeout occurs.
        if (errno == EAGAIN) {
          if (sync_) {
            auto hasTimedOut = [&]{
              return (timeout_ != kNoTimeout) &&
                ((std::chrono::steady_clock::now() - start) >= timeout_);
            };
            if (busyPoll_ && !hasTimedOut()) {
              // Keep looping on EAGAIN if busy-poll flag has been set and the
              // timeout (if set) hasn't been reached
              continue;
            } else {
              // Either timeout on poll or blocking call returning with EAGAIN
              // indicates timeout
              signalIoFailure(
                  GLOO_ERROR_MSG("Read timeout ", peer_.str()));
            }
          } else {
            // Async. This read is done.
            return false;
          }
        }

        // Retry on EINTR
        if (errno == EINTR) {
          continue;
        }

        // ECONNRESET happens when the remote peer unexpectedly terminates
        if (errno == ECONNRESET) {
          changeState(CLOSED);
        }

        // Unexpected error
        signalIoFailure(GLOO_ERROR_MSG(
            "Read error ", peer_.str(), ": ", strerror(errno)));
      }
      break;
    }

    // Transition to CLOSED on EOF
    if (rv == 0) {
      changeState(CLOSED);
      if (sync_) {
        signalIoFailure(GLOO_ERROR_MSG(
            "Remote socket closed during sync read ", peer_.str()));
      } else {
        return false;
      }
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

  try {
    checkErrorState();

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
  } catch (const ::gloo::IoException&) {
    // Catch IO exceptions on the event handling thread. The exception has
    // already been saved and user threads signaled.
  }
}

void Pair::handleListening() {
  struct sockaddr_storage addr;
  socklen_t addrlen = sizeof(addr);
  int rv;

  rv = accept(fd_, (struct sockaddr*)&addr, &addrlen);

  // Close the listening file descriptor whether we've successfully connected
  // or run into an error and will throw an exception.
  dev_->unregisterDescriptor(fd_);
  close(fd_);
  fd_ = FD_INVALID;

  if (rv == -1) {
    signalIoFailure(GLOO_ERROR_MSG("accept: ", strerror(errno)));
  }

  // Connected, replace file descriptor
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
  if (optval != 0) {
    signalIoFailure(
        GLOO_ERROR_MSG("connect ", peer_.str(), ": ", strerror(optval)));
  }

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

  // Set timeout
  struct timeval tv = {};
  tv.tv_sec = timeout_.count() / 1000;
  tv.tv_usec = (timeout_.count() % 1000) * 1000;
  rv = setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  GLOO_ENFORCE_NE(rv, -1);
  rv = setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
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
void Pair::changeState(state nextState) {
  // Ignore nops
  if (nextState == state_) {
    return;
  }

  // State can only move forward
  GLOO_ENFORCE_GT(nextState, state_);

  // Clean up file descriptor when transitioning to CLOSED.
  if (nextState == CLOSED) {
    if (state_ == CONNECTED) {
      if (!sync_) {
        dev_->unregisterDescriptor(fd_);
      }
      close(fd_);
      fd_ = FD_INVALID;
    } else if (state_ == LISTENING) {
      // The pair may be in the LISTENING state when it is destructed.
      if (fd_ != FD_INVALID) {
        dev_->unregisterDescriptor(fd_);
        close(fd_);
        fd_ = FD_INVALID;
      }
    } else if (state_ == CONNECTING) {
      // The pair may be in the CONNECTING state when it is destructed.
      if (fd_ != FD_INVALID) {
        dev_->unregisterDescriptor(fd_);
        close(fd_);
        fd_ = FD_INVALID;
      }
    } else {
      GLOO_ENFORCE(false, "Invalid state: ", state_);
    }
  }

  state_ = nextState;
  cv_.notify_all();
}

void Pair::waitUntilConnected(
    std::unique_lock<std::mutex>& lock,
    bool useTimeout) {
  auto pred = [&] {
    checkErrorState();
    return state_ >= CONNECTED;
  };
  auto timeoutSet = timeout_ != kNoTimeout;
  if (useTimeout && timeoutSet) {
    // Use a longer timeout when waiting for initial connect
    auto done = cv_.wait_for(lock, timeout_ * 5, pred);
    if (!done) {
      signalIoFailure(GLOO_ERROR_MSG("Connect timeout ", peer_.str()));
    }
  } else {
    cv_.wait(lock, pred);
  }
}

void Pair::verifyConnected() {
  // This code path should only be called after reaching the connected state
  GLOO_ENFORCE_GE(
      state_,
      CONNECTED,
      "Pair is not connected (",
      self_.str(),
      " <--> ",
      peer_.str(),
      ")");
  // Check if the socket has been closed. We were unable to tell if this was an
  // error or normal tear down, but now throw since we are trying to do IO.
  if (state_ == CLOSED) {
    signalIoFailure(GLOO_ERROR_MSG("Socket closed ", peer_.str()));
  }
}

void Pair::send(Op& op) {
  std::unique_lock<std::mutex> lock(m_);
  checkErrorState();

  // The connect function already wait for the pair to become
  // connected (both in listening and connecting mode).
  // No need to wait again here.
  verifyConnected();

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

  // Wait until event loop has finished current write. No need to wait for
  // timeout here. If necessary, the ongoing write op will timeout and signal
  // this thread.
  if (!sync_ && tx_.buf_ != nullptr) {
    cv_.wait(lock, [&]{
      checkErrorState();
      return tx_.buf_ == nullptr;
    });
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
  checkErrorState();

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

void Pair::signalIoFailureExternal(const std::string& msg) {
  std::unique_lock<std::mutex> lock(m_);
  signalIoFailure(msg);
};

void Pair::signalIoFailure(const std::string& msg) {
  auto ex = ::gloo::IoException(msg);
  if (ex_ == nullptr) {
    // If we haven't seen an error yet, store the exception to throw on future
    // calling threads.
    ex_ = std::make_exception_ptr(ex);
    // Loop through the buffers and signal that an error has occurred.
    for (auto it = buffers_.begin(); it != buffers_.end(); it++) {
      it->second->signalError(ex_);
    }
    // Signal any threads in the async path
    cv_.notify_all();
  }
  // Finally, throw the exception on this thread.
  throw ex;
};

void Pair::checkErrorState() {
  // If we previously encountered an error, rethrow here.
  if (ex_ != nullptr) {
    std::rethrow_exception(ex_);
  }
}

} // namespace tcp
} // namespace transport
} // namespace gloo
