#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace at::native::onednn {

template <
    class key_t,
    class value_t,
    template <typename...> class map = std::unordered_map>
class lru_cache {
 public:
  class node_t;

  using value_type = typename std::pair<key_t, value_t>;

  // Only need opaque node_t pointer, it'll compile
  using iterator = typename std::list<node_t>::iterator;
  using const_iterator = typename std::list<node_t>::const_iterator;

  using map_it = typename map<key_t, iterator>::iterator;
  using const_map_it = typename map<key_t, iterator>::const_iterator;

  // Only class possible, we can't use typedef or using. Or can we?
  class node_t : public std::pair<map_it, value_t> {
   public:
    node_t(const std::pair<map_it, value_t>& l)
        : std::pair<map_it, value_t>(l) {}
    node_t(std::pair<map_it, value_t>&& l)
        : std::pair<map_it, value_t>(std::move(l)) {}
  };

  using size_type = typename std::list<node_t>::size_type;

  lru_cache(size_type capacity) : capacity_(capacity) {}
  lru_cache() : capacity_(0) {}

  size_type size() const {
    return map_.size();
  }
  size_type max_size() const {
    return capacity_;
  }
  void resize(size_type new_capacity) {
    capacity_ = new_capacity;

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last--;
      map_.erase(last->first);
      vlist_.pop_back();
    }
  }

  iterator begin() noexcept {
    auto it = map_.begin();
    if (it == map_.end()) {
      return vlist_.end();
    }
    return it->second;
  }
  const_iterator begin() const noexcept {
    const auto it = map_.begin();
    if (it == map_.end()) {
      return vlist_.end();
    }
    return it->second;
  }
  iterator end() noexcept {
    return vlist_.end();
  }
  const_iterator end() const noexcept {
    return vlist_.end();
  }

  iterator find(const key_t& key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  // Is this feasible?
  const_iterator find(const key_t& key) const {
    const auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  bool empty() const noexcept {
    return vlist_.empty();
  }

  void clear() noexcept {
    vlist_.clear();
    map_.clear();
  }

  // Can we?
  // template <class... Args>
  // std::pair<iterator, bool> emplace(Args&&... args) {
  // }

  std::pair<iterator, bool> insert(const value_type& value) {
    auto map_it = map_.find(value.first);

    if (map_it == map_.end()) {
      vlist_.push_front(std::make_pair(map_it, value.second));
      auto list_it = vlist_.begin();
      auto updated = map_.insert(map_it, std::make_pair(value.first, list_it));
      // Update node to pointer to new map position
      list_it->first = updated;
    } else
      return std::make_pair(map_it->second, false);

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last--;
      map_.erase(last->first);
      vlist_.pop_back();
    }

    return std::make_pair(vlist_.begin(), true);
  }

  iterator erase(iterator pos) {
    auto map_pos = pos->first;
    map_.erase(map_pos);
    return vlist_.erase(pos);
  }

  // Warning: carefully check iterator validity
  void swap(lru_cache& other) {
    std::swap(vlist_, other.vlist_);
    std::swap(map_, other.map_);
    std::swap(capacity_, other.capacity_);
  }

 private:
  std::list<node_t> vlist_;
  map<key_t, iterator> map_;
  size_type capacity_;
};

} // namespace at::native::onednn
