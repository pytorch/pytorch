#include "prefix_store.h"

#include <sstream>

namespace fbcollective {
namespace rendezvous {

PrefixStore::PrefixStore(
    const std::string& prefix,
    std::unique_ptr<Store>& store)
    : prefix_(prefix), store_(std::move(store)) {}

std::string PrefixStore::joinKey(const std::string& key) {
  std::stringstream ss;
  ss << prefix_ << "/" << key;
  return ss.str();
}

void PrefixStore::set(const std::string& key, const std::vector<char>& data) {
  store_->set(joinKey(key), data);
}

std::vector<char> PrefixStore::get(const std::string& key) {
  return store_->get(joinKey(key));
}

void PrefixStore::wait(const std::vector<std::string>& keys) {
  std::vector<std::string> joinedKeys;
  joinedKeys.reserve(keys.size());
  for (const auto& key : keys) {
    joinedKeys.push_back(joinKey(key));
  }
  store_->wait(joinedKeys);
}

} // namespace rendezvous
} // namespace fbcollective
