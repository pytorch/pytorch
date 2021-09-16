#pragma once

// The following code is derived from
// https://github.com/oneapi-src/oneDNN/blob/master/src/common/rw_mutex.hpp

// As shared_mutex was introduced only in C++17
// a custom implementation of read-write lock pattern is used
#include <memory>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct rw_mutex_t {
  rw_mutex_t();
  void lock_read();
  void lock_write();
  void unlock_read();
  void unlock_write();
  ~rw_mutex_t();
  rw_mutex_t(const rw_mutex_t& other) = delete;
  rw_mutex_t& operator=(const rw_mutex_t& other) = delete;

 private:
  struct rw_mutex_impl_t;
  std::unique_ptr<rw_mutex_impl_t> rw_mutex_impl_;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch