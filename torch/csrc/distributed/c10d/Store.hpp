#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <c10/macros/Macros.h>
#include <torch/custom_class.h>

namespace c10d {

// callback function will be given arguments (optional<string> oldValue,
// optional<string> newValue)
using WatchKeyCallback =
    std::function<void(c10::optional<std::string>, c10::optional<std::string>)>;

class TORCH_API Store : public torch::CustomClassHolder {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(300);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  Store() : timeout_(kDefaultTimeout) {}

  explicit Store(const std::chrono::milliseconds& timeout)
      : timeout_(timeout) {}

  ~Store() override;

  void set(const std::string& key, const std::string& value);

  virtual void set(
      const std::string& key,
      const std::vector<uint8_t>& value) = 0;

  std::string compareSet(
      const std::string& key,
      const std::string& currentValue,
      const std::string& newValue);

  virtual std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& currentValue,
      const std::vector<uint8_t>& newValue) {
    TORCH_INTERNAL_ASSERT(false, "Not implemented.");
  }

  std::string get_to_str(const std::string& key);

  virtual std::vector<uint8_t> get(const std::string& key) = 0;

  virtual int64_t add(const std::string& key, int64_t value) = 0;

  virtual bool deleteKey(const std::string& key) = 0;

  virtual bool check(const std::vector<std::string>& keys) = 0;

  virtual int64_t getNumKeys() = 0;

  virtual void wait(const std::vector<std::string>& keys) = 0;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) = 0;

  virtual const std::chrono::milliseconds& getTimeout() const noexcept;

  virtual void setTimeout(const std::chrono::milliseconds& timeout);

  // watchKey() takes two arguments: key and callback function. The callback
  // should be run whenever the key is changed (create, update, or delete). The
  // callback function takes two parameters: currentValue and newValue, which
  // are optional depending on how the key is changed. These key updates should
  // trigger the callback as follows:
  // CREATE: callback(c10::nullopt, newValue) // null currentValue
  // UPDATE: callback(currentValue, newValue)
  // DELETE: callback(currentValue, c10::nullopt) // null newValue
  virtual void watchKey(
      const std::string& /* unused */,
      WatchKeyCallback /* unused */) {
    TORCH_CHECK(
        false,
        "watchKey only implemented for TCPStore and PrefixStore that wraps TCPStore.");
  }

 protected:
  std::chrono::milliseconds timeout_;
};

} // namespace c10d
