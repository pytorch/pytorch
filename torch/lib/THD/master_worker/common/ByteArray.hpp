#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace thd { namespace rpc {

struct ByteArray {
  using size_type = std::size_t;

  ByteArray();
  ByteArray(std::size_t size);
  ByteArray(const char* arr, std::size_t size);
  ByteArray(ByteArray&& arr);
  ByteArray(const ByteArray& arr);
  ~ByteArray();

  ByteArray& append(const char* arr, std::size_t size);
  const char* data() const;
  size_type length() const;

  std::string to_string() const;

private:
  std::vector<char> _data;
};

}} // namespace rpc, thd

