#pragma once

#include <memory>

#include "fbcollective/transport/address.h"
#include "fbcollective/transport/buffer.h"

namespace fbcollective {
namespace transport {

class Pair {
 public:
  virtual ~Pair() = 0;

  virtual const Address& address() const = 0;

  virtual void connect(const std::vector<char>& bytes) = 0;

  virtual std::unique_ptr<Buffer>
  createSendBuffer(int slot, void* ptr, size_t size) = 0;

  virtual std::unique_ptr<Buffer>
  createRecvBuffer(int slot, void* ptr, size_t size) = 0;
};

} // namespace transport
} // namespace fbcollective
