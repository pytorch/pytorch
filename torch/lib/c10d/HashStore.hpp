#pragma once

#include <sys/types.h>

#include <condition_variable>
#include <mutex>
#include <unordered_map>

#include <c10d/Store.hpp>

namespace c10d {

class HashStore : public Store {
 public:
  ~HashStore() override {}

  void set(const std::string& key, const std::vector<uint8_t>& data) override;

  std::vector<uint8_t> get(const std::string& key) override;

  void wait(const std::vector<std::string>& keys) override {
    wait(keys, Store::kDefaultTimeout);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool check(const std::vector<std::string>& keys) override;

 protected:
  std::unordered_map<std::string, std::vector<uint8_t>> map_;
  std::mutex m_;
  std::condition_variable cv_;
};

} // namespace c10d
