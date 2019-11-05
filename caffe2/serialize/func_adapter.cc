#include "caffe2/serialize/func_adapter.h"
#include <c10/util/Exception.h>
#include <iostream>

namespace caffe2 {
namespace serialize {

FuncAdapter::FuncAdapter(ReaderFunc in, SeekerFunc seeker, size_t size)
    : in_(in), seeker_(seeker), size_(size) {}

size_t FuncAdapter::size() const {
  return size_;
}

size_t FuncAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  seeker_(pos);
  return in_(reinterpret_cast<char*>(buf), n);
}

FuncAdapter::~FuncAdapter() {}

} // namespace serialize
} // namespace caffe2
