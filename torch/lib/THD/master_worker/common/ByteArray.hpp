#pragma once

#include <cstddef>
#include <string>

namespace thd { namespace rpc {

struct ByteArray {
  using size_type = size_t;

  ByteArray();
  ByteArray(size_t size);
  ByteArray(const char* arr, size_t size);
  ByteArray(ByteArray&& arr);
  ByteArray(const ByteArray& arr);
  ~ByteArray();

  ByteArray& append(const char* arr, size_t size);
  const char* data() const;
  size_type length() const;

  std::string to_string() const;

private:
  std::string _data;
};

}} // namespace rpc, thd

