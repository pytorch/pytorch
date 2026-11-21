#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()

#include <dnnl.hpp>

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <list>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace at::native::mkldnn::utils {

// Ported from ideep::key_t.
// for computation cache
using key_t = std::string;

// Ported from ideep::utils::lru_cache.
template <
    class key_t,
    class value_t,
    template <typename...> class map = std::unordered_map>
class lru_cache {
 public:
  class node_t;

  typedef typename std::pair<key_t, value_t> value_type;
  // Only need opaque node_t pointer, it'll compile
  typedef typename std::list<node_t>::iterator iterator;
  typedef typename std::list<node_t>::const_iterator const_iterator;
  typedef typename map<key_t, iterator>::iterator map_it;
  typedef typename map<key_t, iterator>::const_iterator const_map_it;

  // Only class possible, we can't use typedef or using. Or can we?
  class node_t : public std::pair<map_it, value_t> {
   public:
    node_t(const std::pair<map_it, value_t>& l)
        : std::pair<map_it, value_t>(l) {}
    node_t(std::pair<map_it, value_t>&& l)
        : std::pair<map_it, value_t>(std::move(l)) {}
  };

  typedef typename std::list<node_t>::size_type size_type;

  lru_cache(size_type capacity) : capacity_(capacity) {}

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

  iterator end() noexcept {
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

  void clear() noexcept {
    vlist_.clear();
    map_.clear();
  }

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

 private:
  std::list<node_t> vlist_;
  map<key_t, iterator> map_;
  size_type capacity_;
};

// Ported from ideep::utils::computation_cache.
#ifdef __aarch64__
/* initializing the cache with capacity 1 (as zero is crashig) to
 * disable it by default. And enable it for inference use cases that benefit
 * by setting LRU_CACHE_CAPACITY to appropriate size
 */
template <class value_t, size_t capacity = 1, class key_t = std::string>
#else
template <class value_t, size_t capacity = 1024, class key_t = std::string>
#endif
class computation_cache {
 public:
  using iterator = typename lru_cache<key_t, value_t>::iterator;

 protected:
  template <typename T>
  static inline iterator create(const key_t& key, T&& args) {
    auto it = t_store().insert(std::make_pair(key, std::forward<T>(args)));
    return it.first;
  }

  static inline value_t& fetch(iterator it) {
    return it->second;
  }

  static inline iterator find(const key_t& key) {
    return t_store().find(key);
  }

  static inline iterator end() {
    return t_store().end();
  }

 public:
  static inline value_t& fetch_or_create(
      const key_t& key,
      const std::function<value_t()>& callback) {
    auto it = find(key);
    return it == end() ? fetch(create((key), callback())) : fetch(it);
  }

  static inline lru_cache<key_t, value_t>& t_store() {
    static thread_local lru_cache<key_t, value_t> t_store_(capacity);
    // Only read for its initializer, which applies LRU_CACHE_CAPACITY once.
    [[maybe_unused]] static thread_local int new_capacity = [&](const char* pt) {
      if (pt != NULL) {
#ifndef NDEBUG
        dnnl::error::wrap_c_api(
            std::atoi(pt) > 0 ? dnnl_success : dnnl_invalid_arguments,
            "The LRU_CACHE_CAPACITY should be positive");
#endif
        t_store_.resize(std::atoi(pt));
        return std::atoi(pt);
      } else {
        return 0;
      }
    }(std::getenv("LRU_CACHE_CAPACITY"));
    return t_store_;
  }
};

// Ported from ideep::utils::bytestring.
using bytestring = std::string;

// Ported from ideep::utils::clz.
/* Definitions for builtins unavailable on MSVC */
// see
// https://github.com/llvm/llvm-project/blob/master/compiler-rt/lib/builtins/int_lib.h
#if defined(_MSC_VER) && !defined(__clang__)
inline uint32_t clz(uint32_t x) {
  unsigned long leading_zero = 0;
  if (_BitScanReverse(&leading_zero, x))
    return 31 - leading_zero;
  return 32;
}
#else
inline uint32_t clz(uint32_t x) {
  return __builtin_clz(x);
}
#endif

// Ported from ideep::utils::to_bytes.
inline void to_bytes(bytestring& bytes, const int arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  if (arg == 0)
    return;
  auto len = sizeof(arg) - (clz(arg) / 8);
  bytes.append(as_cstring, len);
}

inline void to_bytes(bytestring& bytes, const bool arg) {
  to_bytes(bytes, arg ? 1 : 0);
  bytes.append(1, 'b');
}

inline void to_bytes(bytestring& bytes, const float arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  bytes.append(as_cstring, sizeof(float));
}

inline void to_bytes(bytestring& bytes, const uint64_t arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  bytes.append(as_cstring, sizeof(uint64_t));
}

inline void to_bytes(bytestring& bytes, const int64_t arg) {
  auto as_cstring = reinterpret_cast<const char*>(&arg);
  bytes.append(as_cstring, sizeof(int64_t));
}

template <typename T>
inline void to_bytes(bytestring& bytes, std::vector<T>& arg) {
  if (arg.size() > 0) {
    for (T elems : arg) {
      to_bytes(bytes, elems);
      bytes.append(1, 'v');
    }
    bytes.pop_back();
  } else {
    bytes.append(1, 'v');
  }
}

template <typename T>
inline void to_bytes(bytestring& bytes, const std::vector<T>& arg) {
  // remove constness, then jumps to `to_bytes(bytestring&, vector<T>&)`
  to_bytes(bytes, const_cast<std::vector<T>&>(arg));
}

template <typename T>
inline void to_bytes(bytestring& bytes, std::vector<T>&& arg) {
  // `arg` is an lval ref now, then jumps to `to_bytes(bytestring&, vector<T>&)`
  to_bytes(bytes, arg);
}

template <
    typename T,
    typename = typename std::enable_if<std::is_enum<T>::value>::type>
inline void to_bytes(bytestring& bytes, T arg) {
  to_bytes(bytes, static_cast<int>(arg));
}

template <
    typename T,
    typename = typename std::enable_if<std::is_class<T>::value>::type,
    typename = void>
inline void to_bytes(bytestring& bytes, const T arg) {
  arg.to_bytes(bytes);
}

template <typename T, typename... Ts>
inline void to_bytes(bytestring& bytes, T&& arg, Ts&&... args) {
  to_bytes(bytes, std::forward<T>(arg));
  bytes.append(1, '*');
  to_bytes(bytes, std::forward<Ts>(args)...);
}

// Ported from ideep::utils::create_key.
template <typename... Ts>
inline key_t create_key(Ts&&... args) {
  key_t k;
  to_bytes(k, std::forward<Ts>(args)...);
  return k;
}

} // namespace at::native::mkldnn::utils

#endif // AT_MKLDNN_ENABLED()
