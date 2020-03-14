#include <c10d/Store.hpp>

namespace c10d {

constexpr std::chrono::milliseconds Store::kDefaultTimeout;
constexpr std::chrono::milliseconds Store::kNoTimeout;

// Define destructor symbol for abstract base class.
Store::~Store() {}

// Set timeout function
void Store::setTimeout(const std::chrono::milliseconds& timeout) {
  timeout_ = timeout;
}

} // namespace c10d
