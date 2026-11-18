#pragma once

#include <iterator>
#include <list>
#include <unordered_map>
#include <utility>

namespace at::native::onednn {

template <
    class key_t,
    class value_t,
    template <typename...> class map_t = std::unordered_map>
class lru_cache {
 public:
  using value_type = std::pair<key_t, value_t>;
  using list_type = std::list<value_type>;
  using list_iter = typename list_type::iterator;
  using map_type = map_t<key_t, list_iter>;
  using const_list_iter = typename list_type::const_iterator;
  using size_type = typename list_type::size_type;

  explicit lru_cache(size_type capacity) : capacity_(capacity) {}
  lru_cache() : capacity_(0) {}

  [[nodiscard]] size_type size() const noexcept {
    return map_.size();
  }
  [[nodiscard]] size_type max_size() const noexcept {
    return capacity_;
  }
  [[nodiscard]] bool empty() const noexcept {
    return vlist_.empty();
  }

  void resize(size_type new_capacity) {
    capacity_ = new_capacity;
    trim();
  }

  list_iter begin() noexcept {
    return vlist_.begin();
  }
  const_list_iter begin() const noexcept {
    return vlist_.begin();
  }
  list_iter end() noexcept {
    return vlist_.end();
  }
  const_list_iter end() const noexcept {
    return vlist_.end();
  }

  void clear() noexcept {
    map_.clear();
    vlist_.clear();
  }

  void swap(lru_cache& other) noexcept {
    using std::swap;
    swap(vlist_, other.vlist_);
    swap(map_, other.map_);
    swap(capacity_, other.capacity_);
  }

  list_iter find(const key_t& key) {
    auto it = map_.find(key);
    if (it == map_.end())
      return end();
    vlist_.splice(vlist_.begin(), vlist_, it->second);
    return it->second;
  }

  std::pair<list_iter, bool> insert(const value_type& value) {
    auto it = map_.find(value.first);
    if (it != map_.end()) {
      // Move existing to front
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return {it->second, false};
    }

    // Insert new at front
    vlist_.emplace_front(value);
    map_[value.first] = vlist_.begin();

    trim();

    return {vlist_.begin(), true};
  }

  list_iter erase(list_iter pos) {
    map_.erase(pos->first);
    return vlist_.erase(pos);
  }

 private:
  void trim() {
    while (map_.size() > capacity_) {
      auto last = std::prev(vlist_.end());
      map_.erase(last->first);
      vlist_.pop_back();
    }
  }

  list_type vlist_;
  map_type map_;
  size_type capacity_;
};

} // namespace at::native::onednn
