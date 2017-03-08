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
  : _data()
{}

ByteArray::ByteArray(std::size_t size)
  : ByteArray()
{
  _data.reserve(size);
}

ByteArray::ByteArray(const char* arr, std::size_t size)
  : _data(arr, arr + size)
{}

ByteArray::ByteArray(ByteArray&& arr)
{
  std::swap(_data, arr._data);
}

ByteArray::ByteArray(const ByteArray& arr)
  : _data(arr._data)
{}

ByteArray::~ByteArray() {}

ByteArray& ByteArray::append(const char* arr, std::size_t size) {
  if (_data.capacity() < _data.size() + size) {
    _data.reserve(_data.capacity() + size);
  }

  for (std::size_t i = 0; i < size; ++i) {
    _data.push_back(arr[i]);
  }

  return *this;
}

const char* ByteArray::data() const {
  return _data.data();
}

std::size_t ByteArray::length() const {
  return _data.size();
}

std::string ByteArray::to_string() const {
  return std::string(_data.begin(), _data.end());
}

}} // namespace rpc, thd
