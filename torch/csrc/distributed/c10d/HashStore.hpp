#pragma once

#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

class TORCH_API HashStore : public Store {
 public:
  c10::intrusive_ptr<Store> clone() override;

  ~HashStore() override = default;

  void set(const std::string& key, const std::vector<uint8_t>& data) override;

  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;

  std::vector<uint8_t> get(const std::string& key) override;

  void wait(const std::vector<std::string>& keys) override {
    wait(keys, timeout_);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  int64_t add(const std::string& key, int64_t value) override;

  int64_t getNumKeys() override;

  bool check(const std::vector<std::string>& keys) override;

  bool deleteKey(const std::string& key) override;

  void append(const std::string& key, const std::vector<uint8_t>& value)
      override;

  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override;

  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override;

  // Returns true if this store support append, multiGet and multiSet
  bool hasExtendedApi() const override;

  void queuePush(const std::string& key, const std::vector<uint8_t>& value)
      override;

  std::vector<uint8_t> queuePop(const std::string& key, bool block) override;

  int64_t queueLen(const std::string& key) override;

 protected:
  bool checkLocked(
      const std::unique_lock<std::mutex>& lock,
      const std::vector<std::string>& keys);

  void waitLocked(
      std::unique_lock<std::mutex>& lock,
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout);

 protected:
  std::unordered_map<std::string, std::vector<uint8_t>> map_;
  std::unordered_map<std::string, std::deque<std::vector<uint8_t>>> queues_;
  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace c10d
