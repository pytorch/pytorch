#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "rw_mutex.hpp"
#include "utils.hpp"

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

struct rw_mutex_t::rw_mutex_impl_t {
#ifdef _WIN32
  using rwlock_t = SRWLOCK;
#else
  using rwlock_t = pthread_rwlock_t;
#endif
  rwlock_t& impl() {
    return impl_;
  }

  private:
    rwlock_t impl_;
};

rw_mutex_t::rw_mutex_t() {
  rw_mutex_impl_ = impl::utils::make_unique<rw_mutex_impl_t>();
  auto& impl = rw_mutex_impl_->impl();
#ifdef _WIN32
  InitializeSRWLock(&impl);
#else
  pthread_rwlock_init(&impl, nullptr);
#endif
}

void rw_mutex_t::lock_read() {
  auto& impl = rw_mutex_impl_->impl();
#ifdef _WIN32
  AcquireSRWLockShared(&impl);
#else
  pthread_rwlock_rdlock(&impl);
#endif
}

void rw_mutex_t::lock_write() {
  auto& impl = rw_mutex_impl_->impl();
#ifdef _WIN32
  AcquireSRWLockExclusive(&impl);
#else
  pthread_rwlock_wrlock(&impl);
#endif
}

void rw_mutex_t::unlock_read() {
  auto& impl = rw_mutex_impl_->impl();
#ifdef _WIN32
  ReleaseSRWLockShared(&impl);
#else
  pthread_rwlock_unlock(&impl);
#endif
}

void rw_mutex_t::unlock_write() {
  auto& impl = rw_mutex_impl_->impl();
#ifdef _WIN32
  ReleaseSRWLockExclusive(&impl);
#else
  pthread_rwlock_unlock(&impl);
#endif
}

rw_mutex_t::~rw_mutex_t() {
// SRW locks do not need to be explicitly destroyed
#ifndef _WIN32
  auto& impl = rw_mutex_impl_->impl();
  pthread_rwlock_destroy(&impl);
#endif
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
