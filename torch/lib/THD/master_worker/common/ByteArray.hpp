#pragma once

#include <cstddef>
#include <string>

namespace thd { namespace rpc {

struct ByteArray {
  using size_type = std::size_t;
  ByteArray();
  ByteArray(std::size_t size);
  ByteArray(char* arr, std::size_t size);
  ByteArray(ByteArray&& arr);
  ByteArray(const ByteArray& arr);
  ~ByteArray();

  ByteArray& append(const char* arr, std::size_t size);
  char* data() const;
  size_type length() const;

  std::string to_string();

  static ByteArray fromData(const char* arr, std::size_t size);

private:
  char* _data;
  size_type _length; // The length of the data.
  size_type _size; // The size of the allocated memory.

  void _realloc(std::size_t new_size);
  void _resize(std::size_t desired_size);
  void _resizeExact(std::size_t desired_size);
};

}} // namespace rpc, thd

