#include <c10d/Store.hpp>

namespace c10d {

constexpr std::chrono::milliseconds Store::kDefaultTimeout;
constexpr std::chrono::milliseconds Store::kNoTimeout;

// Define destructor symbol for abstract base class.
Store::~Store() {}

const std::chrono::milliseconds& Store::getTimeout() const noexcept {
    return timeout_;
}

// Set timeout function
void Store::setTimeout(const std::chrono::milliseconds& timeout) {
  timeout_ = timeout;
}

} // namespace c10d
