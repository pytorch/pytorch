#pragma once

#include <memory>

#include "fbcollective/transport/pair.h"

namespace fbcollective {
namespace transport {

// Forward declarations
class Pair;
class Buffer;

class Device {
 public:
  virtual ~Device() = 0;

  virtual std::unique_ptr<Pair> createPair() = 0;
};

} // namespace transport
} // namespace fbcollective
