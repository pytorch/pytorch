#include <torch/csrc/distributed/c10d/Store.hpp>

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

void Store::set(const std::string& key, const std::string& value) {
  set(key, std::vector<uint8_t>(value.begin(), value.end()));
}

std::string Store::compareSet(
    const std::string& key,
    const std::string& currentValue,
    const std::string& newValue) {
  auto value = compareSet(
      key,
      std::vector<uint8_t>(currentValue.begin(), currentValue.end()),
      std::vector<uint8_t>(newValue.begin(), newValue.end()));
  return std::string(value.begin(), value.end());
}

std::string Store::get_to_str(const std::string& key) {
  auto value = get(key);
  return std::string(value.begin(), value.end());
}

} // namespace c10d
