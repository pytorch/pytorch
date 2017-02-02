#include "fbcollective/transport/ibverbs/device.h"

#include <fcntl.h>
#include <poll.h>

#include <array>

#include "fbcollective/common/logging.h"
#include "fbcollective/transport/ibverbs/pair.h"

namespace fbcollective {
namespace transport {
namespace ibverbs {

// Scope guard for ibverbs device list.
class ibv_devices {
 public:
  ibv_devices() {
    list_ = ibv_get_device_list(&size_);
    FBC_ENFORCE(list_);
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

std::shared_ptr<::fbcollective::transport::Device> CreateDevice(
    const struct attr& attr) {
  auto context = createContext(attr.name);
  FBC_ENFORCE(context, "Unable to find device named: ", attr.name);
  return std::make_shared<Device>(attr, context);
}

Device::Device(const struct attr& attr, ibv_context* context)
    : attr_(attr), context_(context) {
  int rv;

  pd_ = ibv_alloc_pd(context_);
  FBC_ENFORCE(pd_);

  // Completion channel
  comp_channel_ = ibv_create_comp_channel(context_);
  FBC_ENFORCE(comp_channel_);

  // Completion queue
  cq_ = ibv_create_cq(context_, 64, nullptr, comp_channel_, 0);
  FBC_ENFORCE(cq_);

  // Arm notification mechanism for completion queue
  // The second argument is named solicited_only and is
  // set to 0 because we want notifications for everything.
  rv = ibv_req_notify_cq(cq_, 0);
  FBC_ENFORCE_NE(rv, -1);

  // Start thread to poll completion queue and dispatch
  // completions for completed work requests.
  done_ = false;
  loop_.reset(new std::thread(&Device::loop, this));
}

Device::~Device() {
  int rv;

  done_ = true;
  loop_->join();

  rv = ibv_destroy_cq(cq_);
  FBC_ENFORCE_NE(rv, -1);

  rv = ibv_destroy_comp_channel(comp_channel_);
  FBC_ENFORCE_NE(rv, -1);

  rv = ibv_dealloc_pd(pd_);
  FBC_ENFORCE_NE(rv, -1);

  rv = ibv_close_device(context_);
  FBC_ENFORCE_NE(rv, -1);
}

std::unique_ptr<transport::Pair> Device::createPair() {
  auto pair = new Pair(shared_from_this());
  return std::unique_ptr<transport::Pair>(pair);
}

void Device::loop() {
  int rv;

  auto flags = fcntl(comp_channel_->fd, F_GETFL);
  FBC_ENFORCE_NE(flags, -1);

  rv = fcntl(comp_channel_->fd, F_SETFL, flags | O_NONBLOCK);
  FBC_ENFORCE_NE(rv, -1);

  struct pollfd pfd;
  pfd.fd = comp_channel_->fd;
  pfd.events = POLLIN;
  pfd.revents = 0;

  // Keep array for completed work requests on stack
  std::array<struct ibv_wc, capacity_> wc;

  while (!done_) {
    do {
      rv = poll(&pfd, 1, 10);
    } while ((rv == 0 && !done_) || (rv == -1 && errno == EINTR));
    FBC_ENFORCE_NE(rv, -1);

    if (done_ && rv == 0) {
      break;
    }

    ibv_cq* cq;
    void* cq_context;
    rv = ibv_get_cq_event(comp_channel_, &cq, &cq_context);
    FBC_ENFORCE_NE(rv, -1);

    // Only handle CQEs from our own CQ
    FBC_ENFORCE_EQ(cq, cq_);
    ibv_ack_cq_events(cq_, 1);

    // Arm notification mechanism for completion queue
    // The second argument is named solicited_only and is
    // set to 0 because we want notifications for everything.
    rv = ibv_req_notify_cq(cq_, 0);
    FBC_ENFORCE_NE(rv, -1);

    // Invoke handler for every work completion.
    auto nwc = ibv_poll_cq(cq_, capacity_, wc.data());
    FBC_ENFORCE_GE(nwc, 0);
    for (int i = 0; i < nwc; i++) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        continue;
      }

      auto h = reinterpret_cast<Handler*>(wc[i].wr_id);
      h->handleCompletion(&wc[i]);
    }
  }
}
} // namespace ibverbs
} // namespace transport
} // namespace fbcollective
