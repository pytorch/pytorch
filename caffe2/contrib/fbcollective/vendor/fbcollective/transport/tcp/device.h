#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <sys/socket.h>

#include "fbcollective/transport/device.h"

namespace fbcollective {
namespace transport {
namespace tcp {

struct attr {
  attr() {}
  /* implicit */ attr(const char* ptr) : hostname(ptr) {}

  std::string hostname;

  // The address family defaults to AF_UNSPEC such that getaddrinfo(3)
  // will try to find either IPv4 or IPv6 addresses.
  int ai_family = AF_UNSPEC;
};

std::shared_ptr<::fbcollective::transport::Device> CreateDevice(
    const struct attr&);

// Forward declarations
class Pair;
class Buffer;

class Device : public ::fbcollective::transport::Device,
               public std::enable_shared_from_this<Device> {
 public:
  explicit Device(const struct attr& attr);
  virtual ~Device();

  virtual std::unique_ptr<::fbcollective::transport::Pair> createPair()
      override;

 protected:
  void loop();

  void registerDescriptor(int fd, int events, Pair* p);
  void unregisterDescriptor(int fd);

  const struct attr attr_;
  std::atomic<bool> done_;
  std::unique_ptr<std::thread> loop_;

  friend class Pair;
  friend class Buffer;

 private:
  static constexpr auto capacity_ = 64;

  int fd_;

  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace tcp
} // namespace transport
} // namespace fbcollective
