#pragma once

#include <c10d/Store.hpp>
#include <memory>

namespace c10d {

class TORCH_API PrefixStore : public Store {
 public:
  explicit PrefixStore(
      const std::string& prefix,
      c10::intrusive_ptr<Store> store);

  virtual ~PrefixStore(){};

  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool deleteKey(const std::string& key) override;

  int64_t getNumKeys() override;

  bool check(const std::vector<std::string>& keys) override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  const std::chrono::milliseconds& getTimeout() const noexcept override;

  void setTimeout(const std::chrono::milliseconds& timeout) override;

  void watchKey(const std::string& key, WatchKeyCallback callback) override;

 protected:
  std::string prefix_;
  c10::intrusive_ptr<Store> store_;

  std::string joinKey(const std::string& key);
  std::vector<std::string> joinKeys(const std::vector<std::string>& keys);
};

} // namespace c10d
