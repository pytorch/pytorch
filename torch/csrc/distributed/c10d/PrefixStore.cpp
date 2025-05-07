#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <utility>

namespace c10d {

PrefixStore::PrefixStore(std::string prefix, c10::intrusive_ptr<Store> store)
    : prefix_(std::move(prefix)), store_(std::move(store)) {}

c10::intrusive_ptr<Store> PrefixStore::clone() {
  return c10::make_intrusive<PrefixStore>(prefix_, store_->clone());
}

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

void PrefixStore::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->append(joinKey(key), value);
}

std::vector<std::vector<uint8_t>> PrefixStore::multiGet(
    const std::vector<std::string>& keys) {
  std::vector<std::string> prefixed_keys;
  prefixed_keys.reserve(keys.size());
  for (auto& key : keys) {
    prefixed_keys.push_back(joinKey(key));
  }
  return store_->multiGet(prefixed_keys);
}

void PrefixStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  std::vector<std::string> prefixed_keys;
  prefixed_keys.reserve(keys.size());
  for (auto& key : keys) {
    prefixed_keys.push_back(joinKey(key));
  }
  store_->multiSet(prefixed_keys, values);
}

// Returns true if this store support append, multiGet and multiSet
bool PrefixStore::hasExtendedApi() const {
  return store_->hasExtendedApi();
}

void PrefixStore::queuePush(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->queuePush(joinKey(key), value);
}

std::vector<uint8_t> PrefixStore::queuePop(const std::string& key, bool block) {
  return store_->queuePop(joinKey(key), block);
}

int64_t PrefixStore::queueLen(const std::string& key) {
  return store_->queueLen(joinKey(key));
}

c10::intrusive_ptr<Store> PrefixStore::getUnderlyingStore() {
  return store_;
}

c10::intrusive_ptr<Store> PrefixStore::getUnderlyingNonPrefixStore() {
  c10::intrusive_ptr<Store> store = store_;

  while (store) {
    // Attempt to dynamically cast to PrefixStore
    PrefixStore* asPrefixStore = dynamic_cast<PrefixStore*>(store.get());
    if (asPrefixStore) {
      store = asPrefixStore->getUnderlyingStore();
    } else {
      break; // We've reached a non-PrefixStore
    }
  }

  TORCH_CHECK(
      store != nullptr, "Underlying Non-PrefixStore shouldn't be null.");
  return store;
}

} // namespace c10d
