#pragma once

#include <cstddef>

namespace thd { namespace rpc {

struct ByteArray {
  ByteArray();
  ByteArray(std::size_t size);
  ByteArray(char* arr, std::size_t size);
  ByteArray(ByteArray&& arr);
  ByteArray(const ByteArray& arr);
  ~ByteArray();

  ByteArray& append(const char* arr, std::size_t size);
  char* data() const;
  std::size_t length() const;

  static ByteArray fromData(const char* arr, std::size_t size);

private:
  char* _data;
  std::size_t _length; // The length of the data.
  std::size_t _size; // The size of the allocated memory.

  void _realloc(std::size_t new_size);
  void _resize(std::size_t desired_size);
  void _resizeExact(std::size_t desired_size);
};

}} // namespace rpc, thd

