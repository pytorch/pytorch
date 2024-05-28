#pragma once

#include <assert.h>
#include <stdlib.h>
#include <map>
#include <memory>
#include <vector>

#include "XPUProfilerMacros.h"

namespace c10::xpu {

class XPUActivityBuffer {
 public:
  explicit XPUActivityBuffer(size_t size) : size_(size) {
    buf_.reserve(size);
  }
  XPUActivityBuffer() = delete;
  XPUActivityBuffer& operator=(const XPUActivityBuffer&) = delete;
  XPUActivityBuffer(XPUActivityBuffer&&) = default;
  XPUActivityBuffer& operator=(XPUActivityBuffer&&) = default;

  size_t size() const {
    return size_;
  }

  void setSize(size_t size) {
    assert(size <= buf_.capacity());
    size_ = size;
  }

  uint8_t* data() {
    return buf_.data();
  }

 private:
  std::vector<uint8_t> buf_;
  size_t size_;

  std::vector<std::unique_ptr<const itrace_t>> wrappers_;
};

using XPUActivityBufferMap =
    std::map<uint8_t*, std::unique_ptr<XPUActivityBuffer>>;

} // namespace xpu
