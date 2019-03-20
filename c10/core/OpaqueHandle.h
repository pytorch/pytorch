#pragma once

#include <c10/util/intrusive_ptr.h>

namespace c10 {

/**
 * `OpaqueHandle` wraps a custom storage handle (as template param) of a tensor and inherits
 * `c10::intrusive_ptr_target` so that it can fit in `TensorImpl::opaque_handle_`.
 *
 * It supports several ways of wrapping the custom handle:
 * 1. Default constructor which initializes the custom handle with default constructor.
 *    Then the caller can do further initialization on the custom handle by calling
 *    `get_handle()`.
 * 2. Construct with an existing custom handle by copy/move constructor.
 *
 * See `TensorImpl::opaque_handle_`.
 */
template <typename T>
struct C10_API OpaqueHandle : c10::intrusive_ptr_target {
private:
  T handle_;

public:
  OpaqueHandle() {}
  OpaqueHandle(const T& handle): handle_(handle) {}
  OpaqueHandle(T&& handle): handle_(std::move(handle)) {}

  T& get_handle() {
    return handle_;
  }
};

}
