#ifndef _LRU_CACHE_CPP
#define _LRU_CACHE_CPP

#include <string>
#include <unordered_map>
#include <list>
#ifdef WIN32
#include <intrin.h>
#endif
#include "tensor.hpp"
#include "utils.hpp"

namespace ideep {
namespace utils {

template <class key_t, class value_t,
         template <typename ...> class map = std::unordered_map>
class lru_cache {
public:
  class node_t;

  typedef typename std::pair<key_t, value_t> value_type;

  // Only need opaque node_t pointer, it'll compile
  typedef typename std::list<node_t>::iterator iterator;
  typedef typename std::list<node_t>::const_iterator const_iterator;

  typedef typename map<key_t, iterator>::iterator map_it;
  typedef typename map<key_t, iterator>::const_iterator
    const_map_it;

  // Only class possible, we can't use typedef or using. Or can we?
  class node_t : public std::pair<map_it, value_t> {
  public:
    node_t (const std::pair<map_it, value_t> &l) :
      std::pair<map_it, value_t>(l) {}
    node_t (std::pair<map_it, value_t> &&l) :
      std::pair<map_it, value_t>(std::move(l)) {}
  };

  typedef typename std::list<node_t>::size_type size_type;

  lru_cache(size_type capacity) : capacity_(capacity) {}

  size_type size() const { map_.size(); }
  size_type max_size() const { return capacity_; }
  void resize(size_type new_capacity) {
    capacity_ = new_capacity;

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last --;
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

  iterator find(const key_t &key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  // Is this feasible?
  const_iterator find(const key_t &key) const {
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
      last --;
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
  void swap(lru_cache & other) {
    std::swap(vlist_, other.vlist_);
    std::swap(map_, other.map_);
    std::swap(capacity_, other.capacity_);
  }

private:
  std::list<node_t> vlist_;
  map<key_t, iterator> map_;
  size_type capacity_;
};

template <class key_t, class value_t>
class lru_multicache {
public:
  class node_t;

  typedef typename std::pair<key_t, value_t> value_type;

  // Only need opaque node_t pointer, it'll compile
  typedef typename std::list<node_t>::iterator iterator;
  typedef typename std::list<node_t>::const_iterator const_iterator;

  typedef typename std::unordered_multimap<key_t, iterator>::iterator map_it;
  typedef typename std::unordered_multimap<key_t, iterator>::const_iterator
    const_map_it;

  // Only class possible, we can't use typedef or using. Or can we?
  class node_t : public std::pair<map_it, value_t> {
  public:
    node_t (const std::pair<map_it, value_t> &l) :
      std::pair<map_it, value_t>(l) {}
    node_t (std::pair<map_it, value_t> &&l) :
      std::pair<map_it, value_t>(std::move(l)) {}
  };

  typedef typename std::list<node_t>::size_type size_type;

  lru_multicache(size_type capacity) : capacity_(capacity) {}

  size_type size() const { map_.size(); }
  size_type max_size() const { return capacity_; }
  void resize(size_type new_capacity) {
    capacity_ = new_capacity;

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last --;
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

  iterator find(const key_t &key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      return end();
    } else {
      vlist_.splice(vlist_.begin(), vlist_, it->second);
      return it->second;
    }
  }

  // Is this feasible?
  const_iterator find(const key_t &key) const {
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
      auto updated = map_.insert(std::make_pair(value.first, list_it));
      // Update node to pointer to new map position
      list_it->first = updated;
    } else
      return std::make_pair(map_it->second, false);

    // Trim cache
    while (map_.size() > capacity_) {
      auto last = vlist_.end();
      last --;
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
  void swap(lru_multicache & other) {
    std::swap(vlist_, other.vlist_);
    std::swap(map_, other.map_);
    std::swap(capacity_, other.capacity_);
  }

private:
  std::list<node_t> vlist_;
  std::unordered_multimap<key_t, iterator> map_;
  size_type capacity_;
};

template <class value_t, size_t capacity = 1024, class key_t = std::string>
class computation_cache {
public:
  using iterator = typename lru_cache<key_t, value_t>::iterator;

protected:
  template <typename ...Ts>
  static inline iterator create(const key_t& key, Ts&&... args) {
    auto it = t_store().insert(
        std::make_pair(key,value_t(std::forward<Ts>(args)...)));
    return it.first;
  }

  static inline value_t& fetch(iterator it) {
    return it->second;
  }

  static inline void update(value_t &val, iterator it) {
    it->second = val;
  }

  static inline iterator find(const key_t& key) {
    return t_store().find(key);
  }

  static inline iterator end() {
    return t_store().end();
  }

public:
  template <typename ...Ts>
  static inline value_t& fetch_or_create(const key_t& key, Ts&&... args) {
    return fetch(create(key, std::forward<Ts>(args)...));
  }

  static inline void release(
      const key_t& key, const value_t& computation) {
    // Empty
  }

  static inline void release(
      const key_t& key, value_t&& computation) {
    // Empty
  }

  static inline lru_cache<key_t, value_t> &t_store() {
    static thread_local lru_cache<key_t, value_t> t_store_(capacity);
    return t_store_;
  }
};

// TODO: mutex it
template <class value_t, size_t capacity = 1024, class key_t = std::string>
class computation_gcache {
public:
  using iterator = typename lru_cache<key_t, value_t>::iterator;

protected:
  template <typename ...Ts>
  static inline value_t create(Ts&&... args) {
    return value_t(std::forward<Ts>(args)...);
  }

  static inline value_t& fetch(iterator it) {
    auto comp = std::move(it->second);
    g_store().erase(it);
    return comp;
  }

  static inline iterator find(const key_t& key) {
    return g_store().find(key);
  }

  static inline iterator end() {
    return g_store().end();
  }

public:
  template <typename ...Ts>
  static inline value_t& fetch_or_create(const key_t& key, Ts&&... args) {
    const auto it = g_store().find(key);

    if (it != g_store().end()) {
      value_t comp = std::move(it->second);
      g_store().erase(it);
      return comp;
    }

    return value_t(std::forward<Ts>(args)...);
  }

  static inline void release(
      const key_t& key, const value_t& computation) {
    g_store().insert(std::make_pair(key, computation));
  }

  static inline void release(
      const key_t& key, value_t&& computation) {
    g_store().insert(std::make_pair(key, std::move(computation)));
  }

  static lru_cache<key_t, value_t> &g_store() {
    static lru_cache<key_t, value_t> g_store_(capacity);
    return g_store_;
  }
};

// Possible better performance, but use inside class scope only (private)
#define fetch_or_create_m(op, key, ...)  \
    auto it = find(key);  \
    auto op = it == end() ? fetch(create(key, __VA_ARGS__)) : fetch(it);

template <typename T>
inline std::string to_string(const T arg) {
  return std::to_string(arg);
}

inline std::string to_string(const tensor::dims arg) {
  return std::accumulate(std::next(arg.begin()), arg.end(),
      std::to_string(arg[0]), [](std::string a, int b) {
        return a + 'x' + std::to_string(b);
      });
}

template <typename T, typename ...Ts>
inline std::string to_string(T&& arg, Ts&&... args) {
  return to_string(std::forward<T>(arg)) +
    '*' + to_string(std::forward<Ts>(args)...);
}

inline bytestring to_bytes(const tensor arg) {
  bytestring bytes;
  auto arg_desc = arg.get_mkldnn_memory_desc_t();
  bytes.reserve(sizeof(*arg_desc));
  for (int i = 0; i < arg_desc->ndims; i++) {
    bytes.append(to_bytes(static_cast<uint64_t>(arg_desc->layout_desc.blocking.strides[0][i])));
    bytes.append(to_bytes(static_cast<uint64_t>(arg_desc->layout_desc.blocking.strides[1][i])));
    bytes.append(to_bytes(arg_desc->layout_desc.blocking.block_dims[i]));
    bytes.append(to_bytes(arg_desc->layout_desc.blocking.padding_dims[i]));
    bytes.append(to_bytes(arg_desc->layout_desc.blocking.offset_padding_to_data[i]));
    bytes.append(to_bytes(arg_desc->dims[i]));
  }
  bytes.append(to_bytes(static_cast<uint64_t>(arg_desc->layout_desc.blocking.offset_padding)));
  bytes.append(to_bytes(arg_desc->data_type));
  bytes.append(to_bytes(arg_desc->format));

  return bytes;
}

template <typename T, typename ...Ts>
inline bytestring to_bytes(T&& arg, Ts&&... args) {
  bytestring bytes;

  // over booking
  bytes.reserve((sizeof...(args) + 1) * 8);
  bytes.append(to_bytes(std::forward<T>(arg)));
  bytes.append(1, '*');
  bytes.append(to_bytes(std::forward<Ts>(args)...));

  return bytes;
}

template <typename ...Ts>
inline key_t create_key(Ts&&... args) {
  return to_bytes(std::forward<Ts>(args)...);
}

}
}
#endif
