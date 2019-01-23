#include <atomic>
#include <functional>
#include <mutex>
#include <thread>

namespace c10 {

// LeftRight wait-free readers synchronization primitive
// https://hal.archives-ouvertes.fr/hal-01207881/document
template <typename T>
class LeftRight {
 public:
  LeftRight() {
    counters_[0].store(0);
    counters_[1].store(0);
  }

  template <typename F>
  auto read(F&& readFunc) const -> typename std::result_of<F(const T&)>::type {
    auto localCounterIndex = counterIndex_.load();
    ++counters_[localCounterIndex];
    try {
      auto r = readFunc(data_[dataIndex_.load()]);
      --counters_[localCounterIndex];
      return r;
    } catch (const std::exception& e) {
      --counters_[localCounterIndex];
      throw;
    }
  }

  // Throwing from write would result in invalid state
  template <typename F>
  auto write(F&& writeFunc) -> typename std::result_of<F(T&)>::type {
    std::unique_lock<std::mutex> lock(mutex_);
    return uniqueWrite(std::forward<F&&>(writeFunc));
  }

 private:
  // This function doesn't use any locks for the writers. Use only if you know
  // what you're doing
  template <typename F>
  auto uniqueWrite(F&& writeFunc) -> typename std::result_of<F(T&)>::type {
    try {
      auto localDataIndex = dataIndex_.load();
      writeFunc(data_[localDataIndex ^ 1]);
      dataIndex_ = localDataIndex ^ 1;
      auto localCounterIndex = counterIndex_.load();
      while (counters_[localCounterIndex ^ 1].load()) {
        std::this_thread::yield();
      }
      counterIndex_ = localCounterIndex ^ 1;
      while (counters_[localCounterIndex].load()) {
        std::this_thread::yield();
      }
      return writeFunc(data_[localDataIndex]);
    } catch (const std::exception& e) {
      // rethrow
      throw;
    }
  }

  std::mutex mutex_;
  std::atomic<uint8_t> counterIndex_{0};
  std::atomic<uint8_t> dataIndex_{0};
  mutable std::atomic<int32_t> counters_[2];
  T data_[2];
};

} // namespace c10
