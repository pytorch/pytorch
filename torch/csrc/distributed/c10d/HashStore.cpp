#include <torch/csrc/distributed/c10d/HashStore.hpp>

#include <unistd.h>
#include <cerrno>
#include <cstdint>

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
      C10_THROW_ERROR(DistBackendError, "Wait timeout");
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
      C10_THROW_ERROR(DistBackendError, "Wait timeout");
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

void HashStore::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  std::unique_lock<std::mutex> lock(m_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    map_[key] = value;
  } else {
    it->second.insert(it->second.end(), value.begin(), value.end());
  }
  cv_.notify_all();
}

std::vector<std::vector<uint8_t>> HashStore::multiGet(
    const std::vector<std::string>& keys) {
  std::unique_lock<std::mutex> lock(m_);
  auto deadline = std::chrono::steady_clock::now() + timeout_;
  std::vector<std::vector<uint8_t>> res;
  res.reserve(keys.size());

  for (auto& key : keys) {
    auto it = map_.find(key);
    if (it != map_.end()) {
      res.emplace_back(it->second);
    } else {
      auto pred = [&]() { return map_.find(key) != map_.end(); };
      if (timeout_ == kNoTimeout) {
        cv_.wait(lock, pred);
      } else {
        if (!cv_.wait_until(lock, deadline, pred)) {
          C10_THROW_ERROR(DistBackendError, "Wait timeout");
        }
      }
      res.emplace_back(map_[key]);
    }
  }
  return res;
}

void HashStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  std::unique_lock<std::mutex> lock(m_);

  for (auto i : ::c10::irange(keys.size())) {
    map_[keys[i]] = values[i];
  }
  cv_.notify_all();
}

bool HashStore::hasExtendedApi() const {
  return true;
}

} // namespace c10d
