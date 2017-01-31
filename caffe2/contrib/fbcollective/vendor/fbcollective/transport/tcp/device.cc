#include "fbcollective/transport/tcp/device.h"

#include <string.h>
#include <sys/epoll.h>
#include <unistd.h>

#include <array>

#include "fbcollective/common/logging.h"
#include "fbcollective/transport/tcp/pair.h"

namespace fbcollective {
namespace transport {
namespace tcp {

std::shared_ptr<transport::Device> CreateDevice() {
  auto device = std::make_shared<Device>();
  return std::shared_ptr<transport::Device>(device);
}

Device::Device() {
  fd_ = epoll_create(1);
  FBC_ENFORCE_NE(fd_, -1, "epoll_create: ", strerror(errno));

  done_ = false;
  loop_.reset(new std::thread(&Device::loop, this));
}

Device::~Device() {
  done_ = true;
  loop_->join();

  close(fd_);
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
  FBC_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

  return;
}

void Device::unregisterDescriptor(int fd) {
  int rv;

  rv = epoll_ctl(fd_, EPOLL_CTL_DEL, fd, nullptr);
  FBC_ENFORCE_NE(rv, -1, "epoll_ctl: ", strerror(errno));

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

    FBC_ENFORCE_NE(nfds, -1);

    for (int i = 0; i < nfds; i++) {
      Pair* p = reinterpret_cast<Pair*>(events[i].data.ptr);
      p->handleEvents(events[i].events);
    }
  }
}

} // namespace tcp
} // namespace transport
} // namespace fbcollective
