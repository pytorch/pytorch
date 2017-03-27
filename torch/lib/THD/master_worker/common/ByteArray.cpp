#include "ByteArray.hpp"

#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <exception>
#include <string>
#include <system_error>
#include <utility>

namespace thd { namespace rpc {

ByteArray::ByteArray()
  : _data(nullptr)
  , _length(0)
  , _size(0)
{}

ByteArray::ByteArray(std::size_t size)
  : _data(nullptr)
  , _length(0)
  , _size(0)
{
  _resize(size);
}

ByteArray::ByteArray(char* arr, std::size_t size)
  : _data(arr)
  , _length(size)
  , _size(_length)
{}

ByteArray::ByteArray(ByteArray&& arr)
  : _data(nullptr)
  , _length(0)
  , _size(0)
{
  std::swap(_data, arr._data);
  std::swap(_length, arr._length);
  std::swap(_size, arr._size);
}

ByteArray::ByteArray(const ByteArray& arr)
  : _data(nullptr)
  , _length(0)
  , _size(0)
{
  _resizeExact(arr._length);
  _length = arr._length;
  std::memcpy(_data, arr._data, _length);
}

ByteArray::~ByteArray() {
  std::free(_data);
}

ByteArray& ByteArray::append(const char* arr, std::size_t size) {
  this->_resize(_length + size);
  std::memcpy(_data + _length, arr, size);
  _length += size;
  return *this;
}

char* ByteArray::data() const {
  return _data;
}

std::size_t ByteArray::length() const {
  return _length;
}

std::string ByteArray::to_string() {
  return std::string(_data, _length);
}

ByteArray ByteArray::fromData(const char* arr, std::size_t size) {
  char* new_arr = static_cast<char*>(std::malloc(size));
  if (new_arr == nullptr) {
    throw std::system_error(errno,
                            std::system_category(),
                            "failed to allocate memory");
  }
  std::memcpy(new_arr, arr, size);
  return ByteArray(new_arr, size);
}

void ByteArray::_realloc(std::size_t new_size) {
  char* new_data = static_cast<char*>(std::realloc(_data, new_size));
  if (new_data == nullptr) {
    throw std::system_error(errno,
                            std::system_category(),
                            "failed to realloc when trying to resize an array "
                            "of size " + std::to_string(_size) + " to the "
                            "size of " + std::to_string(new_size));
  }
  _data = new_data;
  _size = new_size;
}

void ByteArray::_resize(std::size_t desired_size) {
  if (desired_size <= _size)
    return;
  std::size_t new_size = _size == 0 ? 1 : _size;
  while (desired_size > new_size) {
      new_size *= 2;
      if (new_size < _size) {
        throw std::length_error("overflow when trying to resize an array of "
                                "size " + std::to_string(_size) + " to the "
                                "size of " + std::to_string(desired_size));
      }
  }
  _realloc(new_size);
}

void ByteArray::_resizeExact(std::size_t desired_size) {
  if (desired_size <= _size)
    return;
  _realloc(desired_size);
}

}} // namespace rpc, thd
