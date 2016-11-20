#pragma once

#include <cstddef>

namespace thd { namespace rpc {

struct ByteArray {
  ByteArray();
  ByteArray(size_t size);
  ByteArray(char* arr, size_t size);
  ByteArray(ByteArray&& arr);
  ByteArray(const ByteArray& arr);
  ~ByteArray();

  ByteArray& append(const char* arr, size_t size);
  char* data() const;
  size_t length() const;

  static ByteArray fromData(const char* arr, size_t size);

private:
  char* _data;
  size_t _length; // The length of the data.
  size_t _size; // The size of the allocated memory.

  void _realloc(size_t new_size);
  void _resize(size_t desired_size);
  void _resizeExact(size_t desired_size);
};

}} // namespace rpc, thd

