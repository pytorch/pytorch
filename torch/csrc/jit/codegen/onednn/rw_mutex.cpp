#include <pthread.h>

#include "rw_mutex.hpp"
#include "utils.hpp"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct rw_mutex_t::rw_mutex_impl_t {
  using rwlock_t = pthread_rwlock_t;
  rwlock_t& impl() {
    return impl_;
  }

 private:
  rwlock_t impl_;
};

rw_mutex_t::rw_mutex_t() {
  rw_mutex_impl_ = impl::utils::make_unique<rw_mutex_impl_t>();
  auto& impl = rw_mutex_impl_->impl();
  pthread_rwlock_init(&impl, nullptr);
}

void rw_mutex_t::lock_read() {
  auto& impl = rw_mutex_impl_->impl();
  pthread_rwlock_rdlock(&impl);
}

void rw_mutex_t::lock_write() {
  auto& impl = rw_mutex_impl_->impl();
  pthread_rwlock_wrlock(&impl);
}

void rw_mutex_t::unlock_read() {
  auto& impl = rw_mutex_impl_->impl();
  pthread_rwlock_unlock(&impl);
}

void rw_mutex_t::unlock_write() {
  auto& impl = rw_mutex_impl_->impl();
  pthread_rwlock_unlock(&impl);
}

rw_mutex_t::~rw_mutex_t() {
  auto& impl = rw_mutex_impl_->impl();
  pthread_rwlock_destroy(&impl);
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch