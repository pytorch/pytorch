#pragma once

#include <c10d/Store.hpp>
#include <c10d/test/TestUtils.hpp>

#include <gtest/gtest.h>

namespace c10d {
namespace test {

inline void set(
    Store& store,
    const std::string& key,
    const std::string& value) {
  std::vector<uint8_t> data(value.begin(), value.end());
  store.set(key, data);
}

inline std::vector<uint8_t> compareSet(
    Store& store,
    const std::string& key,
    const std::string& currentValue,
    const std::string& newValue){
  std::vector<uint8_t> currentData(currentValue.begin(), currentValue.end());
  std::vector<uint8_t> newData(newValue.begin(), newValue.end());
  return store.compareSet(key, currentData, newData);
}

inline void check(
    Store& store,
    const std::string& key,
    const std::string& expected) {
  auto tmp = store.get(key);
  auto actual = std::string((const char*)tmp.data(), tmp.size());
  EXPECT_EQ(actual, expected);
}

} // namespace test
} // namespace c10d
