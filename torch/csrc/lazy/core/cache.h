/**
 * Cache utils in this file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/cache.h
 */

#pragma once

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace torch::lazy {

// Generic key and object cache with LRU expiration policy. The objects of type
// T will be stored as std::shared_ptr<T> and taken and returned as such, by the
// cache API.
template <
    typename K,
    typename T,
    typename H = std::hash<K>,
    typename E = std::equal_to<K>>
class Cache {
 public:
  using TypePtr = std::shared_ptr<T>;
  using Element = std::pair<K, TypePtr>;

  explicit Cache(size_t max_size) : max_size_(max_size) {}

  // Adds an object to the cache, unless it already exists. If the cache grows
  // beyond the limit set during construction, the oldest used object will be
  // removed from the cache.
  TypePtr Add(K key, TypePtr object) {
    if (!max_size_) {
      return object;
    }
    std::lock_guard<std::mutex> slock(lock_);
    element_list_.emplace_front(Element(std::move(key), std::move(object)));
    auto it = element_list_.begin();
    auto emplace_result = element_map_.emplace(&it->first, it);
    if (!emplace_result.second) {
      element_list_.erase(it);
      DoLRU(emplace_result.first->second);
    } else if (element_list_.size() > max_size_) {
      Element* last = &element_list_.back();
      element_map_.erase(&last->first);
      element_list_.pop_back();
    }
    return emplace_result.first->second->second;
  }

  // Retrieves the existing object if it exists. If it does, its position in
  // the LRU list gets moved to the head of the list.
  // Returns nullptr if no object with the specified key is found within the
  // cache.
  TypePtr Get(const K& key) {
    if (!max_size_) {
      return nullptr;
    }
    std::lock_guard<std::mutex> slock(lock_);
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      return nullptr;
    }
    DoLRU(it->second);
    return it->second->second;
  }

  TypePtr GetLatest() {
    std::lock_guard<std::mutex> g(lock_);
    TORCH_CHECK(!element_list_.empty());
    return element_list_.front().second;
  }

  bool Erase(const K& key) {
    if (!max_size_) {
      return false;
    }
    std::lock_guard<std::mutex> slock(lock_);
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      return false;
    }
    auto lit = it->second;
    element_map_.erase(it);
    element_list_.erase(lit);
    return true;
  }

  void Clear() {
    if (!max_size_) {
      return;
    }
    std::lock_guard<std::mutex> slock(lock_);
    element_map_.clear();
    element_list_.clear();
  }

  int Numel() const {
    if (!max_size_) {
      return 0;
    }
    std::lock_guard<std::mutex> g(lock_);
    TORCH_CHECK(element_map_.size() == element_list_.size());
    return element_map_.size();
  }

 private:
  using ElementList = std::list<Element>;

  struct Hasher {
    size_t operator()(const K* key) const {
      return hasher(*key);
    }

    H hasher;
  };

  struct Equaler {
    bool operator()(const K* k1, const K* k2) const {
      return equaler(*k1, *k2);
    }

    E equaler;
  };

  using ElementMap = std::
      unordered_map<const K*, typename ElementList::iterator, Hasher, Equaler>;

  void DoLRU(typename ElementList::iterator it) {
    element_list_.splice(element_list_.begin(), element_list_, it);
  }

  mutable std::mutex lock_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const size_t max_size_ = 0;
  ElementList element_list_;
  ElementMap element_map_;
};

} // namespace torch::lazy
