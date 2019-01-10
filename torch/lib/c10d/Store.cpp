#include "Store.hpp"

namespace c10d {

constexpr std::chrono::milliseconds Store::kDefaultTimeout;
constexpr std::chrono::milliseconds Store::kNoTimeout;

// Define destructor symbol for abstract base class.
Store::~Store() {
}

} // namespace c10d
