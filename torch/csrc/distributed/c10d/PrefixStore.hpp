#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>
#include <memory>

namespace c10d {

class TORCH_API PrefixStore : public Store {
 public:
  explicit PrefixStore(std::string prefix, c10::intrusive_ptr<Store> store);

  using Store::set;
  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  using Store::compareSet;
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

  void append(const std::string& key, const std::vector<uint8_t>& value)
      override;

  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override;

  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override;

  // Returns true if this store support append, multiGet and multiSet
  bool hasExtendedApi() const override;

  c10::intrusive_ptr<Store> getUnderlyingStore();

  // Recursively to fetch the store before layers of wrapping with PrefixStore.
  c10::intrusive_ptr<Store> getUnderlyingNonPrefixStore();

 protected:
  std::string prefix_;
  c10::intrusive_ptr<Store> store_;

  std::string joinKey(const std::string& key);
  std::vector<std::string> joinKeys(const std::vector<std::string>& keys);
};

} // namespace c10d
