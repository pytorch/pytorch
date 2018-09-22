#pragma once

#include <c10d/Store.hpp>
#include <c10d/test/TestUtils.hpp>

namespace c10d {
namespace test {

inline void set(
    Store& store,
    const std::string& key,
    const std::string& value) {
  std::vector<uint8_t> data(value.begin(), value.end());
  store.set(key, data);
}

inline void check(
    Store& store,
    const std::string& key,
    const std::string& expected) {
  auto tmp = store.get(key);
  auto actual = std::string((const char*)tmp.data(), tmp.size());
  if (actual != expected) {
    throw std::runtime_error("Expected " + expected + ", got " + actual);
  }
}

} // namespace test
} // namespace c10d
