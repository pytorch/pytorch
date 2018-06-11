#pragma once

#include <unordered_map>
#include <memory>
#include <mutex>

namespace at { namespace native {

namespace {

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Params>
struct ParamsHash {
  size_t operator()(const Params& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < (int)sizeof(Params); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename Params>
struct ParamsEqual {
  bool operator()(const Params& a, const Params& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
  }
};

}

// TODO: Use something less heavy duty than a big honking mutex
template <typename Params, typename T>
struct ParamsMap {
  // Params must be a POD because we read out its memory
  // contenst as char* when hashing and comparing
  static_assert(std::is_pod<Params>::value, "Params is not POD");

  using Map_t = std::unordered_map<Params, T, ParamsHash<Params>, ParamsEqual<Params>>;

  std::mutex mutex;
  Map_t map;

  bool find(const Params& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = it->second;
    return true;
  }

  bool find(const Params& params, T** results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    *results = &(it->second);
    return true;
  }

  void insert(const Params& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }

  template<class ...Args>
  std::pair<typename Map_t::iterator, bool> emplace( Args&&... args ) {
    std::lock_guard<std::mutex> guard(mutex);
    return map.emplace(args...);
  }

};

}}  // at::native
