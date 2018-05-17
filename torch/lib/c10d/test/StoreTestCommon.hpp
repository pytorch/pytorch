#pragma once

#include "Store.hpp"

#include <mutex>
#include <condition_variable>
#include <string>
#include <vector>

namespace c10d {
namespace test {

class Semaphore {
 public:
  void post(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    n_ += n;
    cv_.notify_all();
  }

  void wait(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    while (n_ < n) {
      cv_.wait(lock);
    }
    n_ -= n;
  }

 protected:
  int n_ = 0;
  std::mutex m_;
  std::condition_variable cv_;
};


inline void set(Store& store,
                const std::string& key,
                const std::string& value) {
  std::vector<uint8_t> data(value.begin(), value.end());
  store.set(key, data);
}

inline void check(Store& store,
                  const std::string& key,
                  const std::string& expected) {
  auto tmp = store.get(key);
  auto actual = std::string((const char*) tmp.data(), tmp.size());
  if (actual != expected) {
    throw std::runtime_error("Expected " + expected + ", got " + actual);
  }
}

} // namespace test
} // namespace c10d
