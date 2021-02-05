#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/custom_class.h>

namespace c10d {

class Store : public torch::CustomClassHolder {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(300);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  Store() : timeout_(kDefaultTimeout) {}

  explicit Store(const std::chrono::milliseconds& timeout)
      : timeout_(timeout) {}

  virtual ~Store();

  virtual void set(
      const std::string& key,
      const std::vector<uint8_t>& value) = 0;

  virtual std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& currentValue,
      const std::vector<uint8_t>& newValue) {
          TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");
      }

  virtual std::vector<uint8_t> get(const std::string& key) = 0;

  virtual int64_t add(const std::string& key, int64_t value) = 0;

  virtual bool deleteKey(const std::string& key) = 0;

  virtual bool check(const std::vector<std::string>& keys) = 0;

  virtual int64_t getNumKeys() = 0;

  virtual void wait(const std::vector<std::string>& keys) = 0;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) = 0;

  void setTimeout(const std::chrono::milliseconds& timeout);

 protected:
  std::chrono::milliseconds timeout_;
};

} // namespace c10d
