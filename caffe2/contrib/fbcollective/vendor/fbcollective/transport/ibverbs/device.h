#pragma once

#include <atomic>
#include <thread>

#include <infiniband/verbs.h>

#include "fbcollective/transport/device.h"

namespace fbcollective {
namespace transport {
namespace ibverbs {

struct attr {
  std::string name;
  int port;
  int index;
};

std::shared_ptr<::fbcollective::transport::Device> CreateDevice(
    const struct attr&);

// Forward declarations
class Pair;
class Buffer;

// Pure virtual base class for Pair/Buffer.
// Used to dispatch completion handling from device loop.
class Handler {
 public:
  virtual ~Handler() {}
  virtual void handleCompletion(struct ibv_wc* wc) = 0;
};

class Device : public ::fbcollective::transport::Device,
               public std::enable_shared_from_this<Device> {
  static const int capacity_ = 64;

 public:
  Device(const struct attr& attr, ibv_context* context);
  virtual ~Device();

  virtual std::unique_ptr<::fbcollective::transport::Pair> createPair()
      override;

 protected:
  struct attr attr_;
  ibv_context* context_;
  ibv_pd* pd_;
  ibv_comp_channel* comp_channel_;
  ibv_cq* cq_;

  void loop();

  std::atomic<bool> done_;
  std::unique_ptr<std::thread> loop_;

  friend class Pair;
  friend class Buffer;
};

} // namespace ibverbs
} // namespace transport
} // namespace fbcollective
