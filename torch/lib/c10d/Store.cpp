#include <c10d/Store.hpp>

namespace c10d {

// Define destructor symbol for abstract base class.
Store::~Store() {}

// Set timeout function
void Store::setTimeout(const std::chrono::milliseconds& timeout) {
  timeout_ = timeout;
}

} // namespace c10d
