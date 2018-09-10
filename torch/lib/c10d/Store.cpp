#include "Store.hpp"

namespace c10d {

constexpr std::chrono::milliseconds Store::kDefaultTimeout;
constexpr std::chrono::milliseconds Store::kNoTimeout;

// Define destructor symbol for abstract base class.
Store::~Store() {}

// Set timeout function
void Store::setTimeout(const std::chrono::seconds& timeoutSec) {
  if (timeoutSec.count() == 0) {
    timeout_ = kNoTimeout;
  }
  timeout_ = timeoutSec;
}

} // namespace c10d
