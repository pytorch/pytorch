#include <c10d/HashStore.hpp>

#include <errno.h>
#include <stdint.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <system_error>

#include <c10/util/Exception.h>

namespace c10d {

void HashStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  std::unique_lock<std::mutex> lock(m_);
  map_[key] = data;
  cv_.notify_all();
}

std::vector<uint8_t> HashStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  std::unique_lock<std::mutex> lock(m_);
  auto it = map_.find(key);
  if ((it == map_.end() && expectedValue.empty()) ||
      (it != map_.end() && it->second == expectedValue)) {
    // if the key does not exist and currentValue arg is empty or
    // the key does exist and current value is what is expected, then set it
    map_[key] = desiredValue;
    cv_.notify_all();
    return desiredValue;
  } else if (it == map_.end()) {
    // if the key does not exist
    return expectedValue;
  }
  // key exists but current value is not expected
  return it->second;
}

std::vector<uint8_t> HashStore::get(const std::string& key) {
  std::unique_lock<std::mutex> lock(m_);
  auto it = map_.find(key);
  if (it != map_.end()) {
    return it->second;
  }
  // Slow path: wait up to any timeout_.
  auto pred = [&]() { return map_.find(key) != map_.end(); };
  if (timeout_ == kNoTimeout) {
    cv_.wait(lock, pred);
  } else {
    if (!cv_.wait_for(lock, timeout_, pred)) {
      throw std::system_error(
          ETIMEDOUT, std::system_category(), "Wait timeout");
    }
  }
  return map_[key];
}

void HashStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  const auto end = std::chrono::steady_clock::now() + timeout;
  auto pred = [&]() {
    auto done = true;
    for (const auto& key : keys) {
      if (map_.find(key) == map_.end()) {
        done = false;
        break;
      }
    }
    return done;
  };

  std::unique_lock<std::mutex> lock(m_);
  if (timeout == kNoTimeout) {
    cv_.wait(lock, pred);
  } else {
    if (!cv_.wait_until(lock, end, pred)) {
      throw std::system_error(
          ETIMEDOUT, std::system_category(), "Wait timeout");
    }
  }
}

int64_t HashStore::add(const std::string& key, int64_t i) {
  std::unique_lock<std::mutex> lock(m_);
  const auto& value = map_[key];
  int64_t ti = i;
  if (!value.empty()) {
    auto buf = reinterpret_cast<const char*>(value.data());
    auto len = value.size();
    ti += std::stoll(std::string(buf, len));
  }

  auto str = std::to_string(ti);
  const uint8_t* strB = reinterpret_cast<const uint8_t*>(str.c_str());
  map_[key] = std::vector<uint8_t>(strB, strB + str.size());
  return ti;
}

int64_t HashStore::getNumKeys() {
  std::unique_lock<std::mutex> lock(m_);
  return map_.size();
}

bool HashStore::deleteKey(const std::string& key) {
  std::unique_lock<std::mutex> lock(m_);
  auto numDeleted = map_.erase(key);
  return (numDeleted == 1);
}

bool HashStore::check(const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> lock(m_);
  for (const auto& key : keys) {
    if (map_.find(key) == map_.end()) {
      return false;
    }
  }
  return true;
}

} // namespace c10d
