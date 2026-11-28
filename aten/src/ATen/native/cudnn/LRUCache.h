#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <unordered_map>
#include <utility>

namespace at::native {

// Thread safety and value lifetime beyond a cache operation belong to callers.
// A zero limit is unlimited; a negative limit disables lookup and insertion.
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LRUCache {
 public:
  explicit LRUCache(int64_t limit) : limit_(limit) {}

  LRUCache(const LRUCache&) = delete;
  LRUCache& operator=(const LRUCache&) = delete;
  LRUCache(LRUCache&&) = delete;
  LRUCache& operator=(LRUCache&&) = delete;

  Value* find(const Key& key) {
    if (limit_ < 0) {
      return nullptr;
    }
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }
    if (limit_ > 0) {
      order_.splice(order_.begin(), order_, it->second.second);
    }
    return &it->second.first;
  }

  // Returns whether an entry was evicted.
  template <typename ValueArg>
  bool update(const Key& key, ValueArg&& value) {
    if (limit_ < 0) {
      return false;
    }
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      it->second.first = std::forward<ValueArg>(value);
      if (limit_ > 0) {
        order_.splice(order_.begin(), order_, it->second.second);
      }
      return false;
    }
    if (limit_ == 0) {
      cache_.try_emplace(key, std::forward<ValueArg>(value), order_.end());
      return false;
    }
    const bool evict = cache_.size() >= static_cast<size_t>(limit_);
    if (evict) {
      cache_.erase(order_.back());
      order_.pop_back();
    }
    order_.emplace_front(key);
    cache_.try_emplace(key, std::forward<ValueArg>(value), order_.begin());
    return evict;
  }

  size_t size() const noexcept {
    return cache_.size();
  }

 private:
  using Order = std::list<Key>;
  using Map =
      std::unordered_map<Key, std::pair<Value, typename Order::iterator>, Hash>;

  const int64_t limit_;
  Order order_;
  Map cache_;
};

} // namespace at::native
