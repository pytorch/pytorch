#pragma once

#include <memory>

// The following code is derived from
// https://github.com/oneapi-src/oneDNN/blob/dev-graph-preview2/src/common/rw_mutex.hpp

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

    rw_mutex_t(const rw_mutex_t &) = delete;
    rw_mutex_t &operator=(const rw_mutex_t &) = delete;

private:
    struct rw_mutex_impl_t;
    std::unique_ptr<rw_mutex_impl_t> rw_mutex_impl_;
};

struct lock_read_t {
    explicit lock_read_t(rw_mutex_t &rw_mutex);
    ~lock_read_t();

    lock_read_t(const lock_read_t &) = delete;
    lock_read_t &operator=(const lock_read_t &) = delete;

private:
    rw_mutex_t &rw_mutex_;
};

struct lock_write_t {
    explicit lock_write_t(rw_mutex_t &rw_mutex_t);
    ~lock_write_t();

    lock_write_t(const lock_write_t &) = delete;
    lock_write_t &operator=(const lock_write_t &) = delete;

private:
    rw_mutex_t &rw_mutex_;
};

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
