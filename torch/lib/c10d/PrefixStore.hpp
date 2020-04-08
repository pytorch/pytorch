#pragma once

#include <c10d/Store.hpp>
#include <memory>

namespace c10d {

class PrefixStore : public Store {
 public:
  explicit PrefixStore(const std::string& prefix, std::shared_ptr<Store> store);

  virtual ~PrefixStore(){};

  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool check(const std::vector<std::string>& keys) override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

 protected:
  std::string prefix_;
  std::shared_ptr<Store> store_;

  std::string joinKey(const std::string& key);
  std::vector<std::string> joinKeys(const std::vector<std::string>& keys);
};

} // namespace c10d
