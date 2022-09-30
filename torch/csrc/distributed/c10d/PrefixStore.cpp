#include <torch/csrc/distributed/c10d/PrefixStore.hpp>

namespace c10d {

PrefixStore::PrefixStore(
    const std::string& prefix,
    c10::intrusive_ptr<Store> store)
    : prefix_(prefix), store_(store) {}

std::string PrefixStore::joinKey(const std::string& key) {
  return prefix_ + "/" + key;
}

std::vector<std::string> PrefixStore::joinKeys(
    const std::vector<std::string>& keys) {
  std::vector<std::string> joinedKeys;
  joinedKeys.reserve(keys.size());
  for (const auto& key : keys) {
    joinedKeys.emplace_back(joinKey(key));
  }
  return joinedKeys;
}

void PrefixStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->set(joinKey(key), value);
}

std::vector<uint8_t> PrefixStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  return store_->compareSet(joinKey(key), expectedValue, desiredValue);
}

std::vector<uint8_t> PrefixStore::get(const std::string& key) {
  return store_->get(joinKey(key));
}

int64_t PrefixStore::add(const std::string& key, int64_t value) {
  return store_->add(joinKey(key), value);
}

bool PrefixStore::deleteKey(const std::string& key) {
  return store_->deleteKey(joinKey(key));
}

void PrefixStore::watchKey(const std::string& key, WatchKeyCallback callback) {
  return store_->watchKey(joinKey(key), std::move(callback));
}

int64_t PrefixStore::getNumKeys() {
  return store_->getNumKeys();
}

bool PrefixStore::check(const std::vector<std::string>& keys) {
  auto joinedKeys = joinKeys(keys);
  return store_->check(joinedKeys);
}

void PrefixStore::wait(const std::vector<std::string>& keys) {
  auto joinedKeys = joinKeys(keys);
  store_->wait(joinedKeys);
}

void PrefixStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  auto joinedKeys = joinKeys(keys);
  store_->wait(joinedKeys, timeout);
}

const std::chrono::milliseconds& PrefixStore::getTimeout() const noexcept {
  return store_->getTimeout();
}

void PrefixStore::setTimeout(const std::chrono::milliseconds& timeout) {
  store_->setTimeout(timeout);
}

c10::intrusive_ptr<Store> PrefixStore::getUnderlyingStore() {
  return store_;
}

} // namespace c10d
