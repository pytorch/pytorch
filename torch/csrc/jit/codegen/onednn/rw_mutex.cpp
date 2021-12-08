#include <shared_mutex>

#include <torch/csrc/jit/codegen/onednn/utils.hpp>
#include <torch/csrc/jit/codegen/onednn/rw_mutex.hpp>

// The following code is derived from
// https://github.com/oneapi-src/oneDNN/blob/dev-graph-preview2/src/common/rw_mutex.cpp

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct rw_mutex_t::rw_mutex_impl_t {
    using rwlock_t = std::shared_timed_mutex;
    rwlock_t &impl() { return impl_; }

private:
    rwlock_t impl_;
};

rw_mutex_t::rw_mutex_t() {
    rw_mutex_impl_ = impl::utils::make_unique<rw_mutex_impl_t>();
}

void rw_mutex_t::lock_read() {
    auto &impl = rw_mutex_impl_->impl();
    impl.lock_shared();
}

void rw_mutex_t::lock_write() {
    auto &impl = rw_mutex_impl_->impl();
    impl.lock();
}

void rw_mutex_t::unlock_read() {
    auto &impl = rw_mutex_impl_->impl();
    impl.unlock_shared();
}

void rw_mutex_t::unlock_write() {
    auto &impl = rw_mutex_impl_->impl();
    impl.unlock();
}

rw_mutex_t::~rw_mutex_t() {}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
