/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/transport/ibverbs/device.h"

#include <fcntl.h>
#include <poll.h>

#include <array>

#include "gloo/common/error.h"
#include "gloo/common/logging.h"
#include "gloo/transport/ibverbs/pair.h"

namespace gloo {
namespace transport {
namespace ibverbs {

// Scope guard for ibverbs device list.
class ibv_devices {
 public:
  ibv_devices() {
    list_ = ibv_get_device_list(&size_);
    GLOO_ENFORCE(list_);
  }

  ~ibv_devices() {
    ibv_free_device_list(list_);
  }

  int size() {
    return size_;
  }

  struct ibv_device*& operator[](int i) {
    return list_[i];
  }

 protected:
  int size_;
  struct ibv_device** list_;
};

static ibv_context* createContext(const std::string& name) {
  ibv_devices devices;

  // Look for specified device name
  struct ibv_device* dev = nullptr;
  for (int i = 0; i < devices.size(); i++) {
    if (name == devices[i]->name) {
      dev = devices[i];
      break;
    }
  }

  if (dev == nullptr) {
    return nullptr;
  }

  return ibv_open_device(dev);
}

std::shared_ptr<::gloo::transport::Device> CreateDevice(
    const struct attr& attr) {
  auto context = createContext(attr.name);
  if (!context) {
    GLOO_THROW_INVALID_OPERATION_EXCEPTION(
        "Unable to find device named: ", attr.name);
  }
  return std::make_shared<Device>(attr, context);
}

Device::Device(const struct attr& attr, ibv_context* context)
    : attr_(attr), context_(context) {
  int rv;

  pd_ = ibv_alloc_pd(context_);
  GLOO_ENFORCE(pd_);

  // Completion channel
  comp_channel_ = ibv_create_comp_channel(context_);
  GLOO_ENFORCE(comp_channel_);

  // Start thread to poll completion queue and dispatch
  // completions for completed work requests.
  done_ = false;
  loop_.reset(new std::thread(&Device::loop, this));
}

Device::~Device() {
  int rv;

  done_ = true;
  loop_->join();

  rv = ibv_destroy_comp_channel(comp_channel_);
  GLOO_ENFORCE_EQ(rv, 0);

  rv = ibv_dealloc_pd(pd_);
  GLOO_ENFORCE_EQ(rv, 0);

  rv = ibv_close_device(context_);
  GLOO_ENFORCE_EQ(rv, 0);
}

void Device::setTimeout(const std::chrono::milliseconds& /* timeout */) {
  GLOO_ENFORCE(false, "The ibverbs transport does not support setting timeout");
}

std::unique_ptr<transport::Pair> Device::createPair() {
  auto pair = new Pair(shared_from_this());
  return std::unique_ptr<transport::Pair>(pair);
}

void Device::loop() {
  int rv;

  auto flags = fcntl(comp_channel_->fd, F_GETFL);
  GLOO_ENFORCE_NE(flags, -1);

  rv = fcntl(comp_channel_->fd, F_SETFL, flags | O_NONBLOCK);
  GLOO_ENFORCE_NE(rv, -1);

  struct pollfd pfd;
  pfd.fd = comp_channel_->fd;
  pfd.events = POLLIN;
  pfd.revents = 0;

  while (!done_) {
    do {
      rv = poll(&pfd, 1, 10);
    } while ((rv == 0 && !done_) || (rv == -1 && errno == EINTR));
    GLOO_ENFORCE_NE(rv, -1);

    if (done_ && rv == 0) {
      break;
    }

    struct ibv_cq* cq;
    void* cqContext;
    rv = ibv_get_cq_event(comp_channel_, &cq, &cqContext);
    GLOO_ENFORCE_EQ(rv, 0, "ibv_get_cq_event");

    // Completion queue context is a Pair*.
    // Delegate handling of this event to the pair itself.
    Pair* pair = static_cast<Pair*>(cqContext);
    pair->handleCompletionEvent();
  }
}
} // namespace ibverbs
} // namespace transport
} // namespace gloo
