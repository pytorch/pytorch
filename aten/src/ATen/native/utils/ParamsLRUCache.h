#pragma once

#include <ATen/native/utils/ParamsHash.h>
#include <c10/util/Exception.h>

#include <list>
#include <unordered_map>
#include <utility>

namespace at::native {

template <typename KeyType, typename ValueType>
struct ParamsLRUCache {
  using KeyWrapper = ParamsWrapper<KeyType>;

  int cache_limit;
  std::list<KeyWrapper> cache_order;
  std::unordered_map<
      KeyWrapper,
      std::pair<ValueType, typename std::list<KeyWrapper>::iterator>,
      ParamsWrapperHash<KeyWrapper>> cache;

  explicit ParamsLRUCache(int limit) : cache_limit(limit) {}

  ValueType* find(const KeyType& key) {
    if (cache_limit < 0) return nullptr;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end()) return nullptr;
    if (cache_limit) {
      cache_order.splice(cache_order.begin(), cache_order, it->second.second);
    }
    return &(it->second.first);
  }

  void update(const KeyType& key, ValueType entry) {
    if (cache_limit < 0) return;
    KeyWrapper wrapped;
    wrapped.pod = key;
    auto it = cache.find(wrapped);
    if (it == cache.end()) {
      if (cache_limit == 0) {
        // Unlimited cache — insert into map only, no LRU tracking
        cache.emplace(wrapped, std::make_pair(std::move(entry), cache_order.end()));
      } else {
        if (static_cast<long>(cache.size()) >= cache_limit) {
          auto count = cache.erase(cache_order.back());
          TORCH_INTERNAL_ASSERT(count == 1, "LRU cache eviction failed to erase key");
          cache_order.pop_back();
        }
        cache_order.emplace_front(wrapped);
        cache.emplace(wrapped, std::make_pair(std::move(entry), cache_order.begin()));
      }
    } else {
      it->second.first = std::move(entry);
      if (cache_limit) {
        cache_order.splice(cache_order.begin(), cache_order, it->second.second);
      }
    }
  }
};

} // namespace at::native
