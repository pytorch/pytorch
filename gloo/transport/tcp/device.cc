/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/tcp/device.h"

#include <netdb.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>

#include "gloo/common/logging.h"
#include "gloo/common/error.h"
#include "gloo/transport/tcp/pair.h"

namespace gloo {
namespace transport {
namespace tcp {

static const std::chrono::seconds kTimeoutDefault = std::chrono::seconds(30);
const std::chrono::milliseconds Device::kNoTimeout =
    std::chrono::milliseconds::zero();

std::shared_ptr<transport::Device> CreateDevice(const struct attr& attr) {
  struct attr x = attr;

  // Initialize hostname to equal this host's name
  if (x.hostname.size() == 0) {
    std::array<char, HOST_NAME_MAX> hostname;
    auto rv = gethostname(hostname.data(), hostname.size());
    GLOO_ENFORCE_EQ(rv, 0);
    x.hostname = hostname.data();
  }

  auto device = std::make_shared<Device>(x);
  return std::shared_ptr<transport::Device>(device);
}

Device::Device(const struct attr& attr)
    : attr_(attr), timeout_(kTimeoutDefault) {
  fd_ = epoll_create(1);
  GLOO_ENFORCE_NE(fd_, -1, "epoll_create: ", strerror(errno));

  done_ = false;
  loop_.reset(new std::thread(&Device::loop, this));
}

Device::~Device() {
  done_ = true;
  loop_->join();

  close(fd_);
}

void Device::setTimeout(const std::chrono::milliseconds& timeout) {
  if (timeout < std::chrono::milliseconds::zero()) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION("Invalid timeout", timeout.count());
  }

  std::unique_lock<std::mutex> lock(m_);
  timeout_ = timeout;
}

std::chrono::milliseconds Device::getTimeout() {
  std::unique_lock<std::mutex> lock(m_);
  return timeout_;
}

std::unique_ptr<transport::Pair> Device::createPair() {
  auto pair = new Pair(shared_from_this());
  return std::unique_ptr<transport::Pair>(pair);
}

void Device::registerDescriptor(int fd, int events, Pair* p) {
  struct epoll_event ev;
  int rv;

  ev.events = events;
  ev.data.ptr = p;

  rv = epoll_ctl(fd_, EPOLL_CTL_ADD, fd, &ev);
  if (rv == -1 && errno == EEXIST) {
    rv = epoll_ctl(fd_, EPOLL_CTL_MOD, fd, &ev);
  }
  GLOO_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

  return;
}

void Device::unregisterDescriptor(int fd) {
  int rv;

  rv = epoll_ctl(fd_, EPOLL_CTL_DEL, fd, nullptr);
  GLOO_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

  // Wait for loop to tick before returning, to make sure the handler
  // for this fd is not called once this function returns.
  if (std::this_thread::get_id() != loop_->get_id()) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock);
  }

  return;
}

void Device::loop() {
  std::array<struct epoll_event, capacity_> events;
  int nfds;

  while (!done_) {
    // Wakeup everyone waiting for a loop tick to finish.
    cv_.notify_all();

    // Wait for something to happen
    nfds = epoll_wait(fd_, events.data(), events.size(), 10);
    if (nfds == 0) {
      continue;
    }
    if (nfds == -1 && errno == EINTR) {
      continue;
    }

    GLOO_ENFORCE_NE(nfds, -1);

    for (int i = 0; i < nfds; i++) {
      Pair* p = reinterpret_cast<Pair*>(events[i].data.ptr);
      p->handleEvents(events[i].events);
    }
  }
}

} // namespace tcp
} // namespace transport
} // namespace gloo
