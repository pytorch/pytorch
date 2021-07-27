#pragma once

#include <sys/types.h>

#include <mutex>
#include <unordered_map>

#include <c10d/Store.hpp>

namespace c10d {

class TORCH_API FileStore : public Store {
 public:
  explicit FileStore(const std::string& path, int numWorkers);

  virtual ~FileStore();

  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  int64_t getNumKeys() override;

  bool deleteKey(const std::string& key) override;

  bool check(const std::vector<std::string>& keys) override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  // Returns the path used by the FileStore.
  const std::string& getPath() const noexcept {
    return path_;
  }

 protected:
  int64_t addHelper(const std::string& key, int64_t i);

  std::string path_;
  off_t pos_;

  int numWorkers_;
  const std::string cleanupKey_;
  const std::string regularPrefix_;

  std::unordered_map<std::string, std::vector<uint8_t>> cache_;

  std::mutex activeFileOpLock_;
};

} // namespace c10d
