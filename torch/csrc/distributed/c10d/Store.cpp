#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

constexpr std::chrono::milliseconds Store::kDefaultTimeout;
constexpr std::chrono::milliseconds Store::kNoTimeout;

// Define destructor symbol for abstract base class.
Store::~Store() = default;

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

void Store::append(const std::string& key, const std::vector<uint8_t>& value) {
  // This fallback depends on compareSet
  std::vector<uint8_t> expected = value;
  std::vector<uint8_t> current;
  // cannot use get(key) as it might block forever if the key doesn't exist
  current = compareSet(key, current, expected);
  while (current != expected) {
    expected = current;
    expected.insert(expected.end(), value.begin(), value.end());
    current = compareSet(key, current, expected);
  }
}

std::vector<std::vector<uint8_t>> Store::multiGet(
    const std::vector<std::string>& keys) {
  std::vector<std::vector<uint8_t>> result;
  result.reserve(keys.size());
  for (auto& key : keys) {
    result.emplace_back(get(key));
  }
  return result;
}

void Store::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  for (auto i : ::c10::irange(keys.size())) {
    set(keys[i], values[i]);
  }
}

bool Store::hasExtendedApi() const {
  return false;
}

} // namespace c10d
