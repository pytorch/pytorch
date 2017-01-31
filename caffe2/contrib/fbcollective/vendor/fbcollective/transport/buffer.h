#pragma once

#include <stddef.h>

namespace fbcollective {
namespace transport {

class Buffer {
 public:
  explicit Buffer(int slot, void* ptr, size_t size)
      : slot_(slot), ptr_(ptr), size_(size) {}
  virtual ~Buffer() = 0;

  virtual void send(size_t offset, size_t length) = 0;

  // Send entire buffer by default
  void send() {
    send(0, size_);
  }

  virtual void waitRecv() = 0;
  virtual void waitSend() = 0;

 protected:
  int slot_;
  void* ptr_;
  size_t size_;
};

} // namespace transport
} // namespace fbcollective
