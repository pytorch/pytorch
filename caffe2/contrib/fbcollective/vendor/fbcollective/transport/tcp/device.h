#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "fbcollective/transport/device.h"

namespace fbcollective {
namespace transport {
namespace tcp {

std::shared_ptr<::fbcollective::transport::Device> CreateDevice();

// Forward declarations
class Pair;
class Buffer;

class Device : public ::fbcollective::transport::Device,
               public std::enable_shared_from_this<Device> {
 public:
  Device();
  virtual ~Device();

  virtual std::unique_ptr<::fbcollective::transport::Pair> createPair()
      override;

 protected:
  void loop();

  void registerDescriptor(int fd, int events, Pair* p);
  void unregisterDescriptor(int fd);

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
