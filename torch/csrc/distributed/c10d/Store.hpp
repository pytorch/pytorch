#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <c10/macros/Macros.h>
#include <torch/custom_class.h>

namespace c10d {

// callback function will be given arguments (std::optional<string> oldValue,
// std::optional<string> newValue)
using WatchKeyCallback =
    std::function<void(std::optional<std::string>, std::optional<std::string>)>;

class TORCH_API Store : public torch::CustomClassHolder {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(300);
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();

  Store() : timeout_(kDefaultTimeout) {}

  explicit Store(const std::chrono::milliseconds& timeout)
      : timeout_(timeout) {}

  Store(const Store&) = default;
  Store(Store&&) noexcept = default;

  ~Store() override = default;

  // Clone a thread safe copy of this store object that points to the same
  // underlying store.
  virtual c10::intrusive_ptr<Store> clone() = 0;

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
    C10_THROW_ERROR(NotImplementedError, "Not implemented.");
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

  // watchKey() is deprecated and no longer supported.
  virtual void watchKey(
      const std::string& /* unused */,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      WatchKeyCallback /* unused */) {
    C10_THROW_ERROR(
        NotImplementedError,
        "watchKey is deprecated, no implementation support it.");
  }

  virtual void append(
      const std::string& key,
      const std::vector<uint8_t>& value);

  virtual std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys);

  virtual void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values);

  // Returns true if this store support append, multiGet and multiSet
  virtual bool hasExtendedApi() const;

  virtual void queuePush(
      const std::string& key,
      const std::vector<uint8_t>& value) {
    C10_THROW_ERROR(NotImplementedError, "queue support is not implemented.");
  }

  virtual std::vector<uint8_t> queuePop(const std::string& key, bool block) {
    C10_THROW_ERROR(NotImplementedError, "queue support is not implemented.");
  }

  virtual int64_t queueLen(const std::string& key) {
    C10_THROW_ERROR(NotImplementedError, "queue support is not implemented.");
  }

  virtual std::vector<std::string> listKeys() {
    C10_THROW_ERROR(
        NotImplementedError, "listKeys support is not implemented.");
  }

  // Barrier operation that blocks until world_size workers have reached it.
  // This is an optimized operation that combines increment and wait into a
  // single operation, reducing network round trips compared to using
  // separate add() and wait() calls.
  virtual void barrier(
      const std::string& key,
      int64_t world_size,
      const std::chrono::milliseconds& timeout);

  void barrier(const std::string& key, int64_t world_size) {
    barrier(key, world_size, timeout_);
  }

 protected:
  std::chrono::milliseconds timeout_;
};

/*
StoreTimeoutGuard is a RAII guard that will set the store timeout and restore it
when it returns.
*/
class StoreTimeoutGuard {
 public:
  explicit StoreTimeoutGuard(
      Store& store,
      const std::chrono::milliseconds& timeout)
      : store_(store), oldTimeout_(store.getTimeout()) {
    store.setTimeout(timeout);
  }

  ~StoreTimeoutGuard() {
    store_.setTimeout(oldTimeout_);
  }

  /* Disabling copy and move semantics */
  StoreTimeoutGuard(const StoreTimeoutGuard&) = delete;
  StoreTimeoutGuard& operator=(const StoreTimeoutGuard&) = delete;
  StoreTimeoutGuard(StoreTimeoutGuard&&) = delete;
  StoreTimeoutGuard& operator=(StoreTimeoutGuard&&) = delete;

 private:
  Store& store_;
  std::chrono::milliseconds oldTimeout_{};
};

} // namespace c10d
