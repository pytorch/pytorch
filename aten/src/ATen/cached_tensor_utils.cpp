#include <ATen/ATen.h>

#include <ATen/cached_tensor_utils.h>


namespace at {
namespace caching {


using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;

// Some systems (just cudagraphs currently) will persist a static tensor output whose TensorImpl
// does not change across iterations. For these tensors caching dtype conversions
// is invalid. If we are not using these systems the enabled
// flag will be false and we will avoid the hash lookup. Like `cached_casts` in,
// autocast_mode, we hash on the TensorImpl* and keep the pointer alive with a weakref value.
bool cached_tensorimpls_enabled = false;
ska::flat_hash_map<TensorImpl*, weakref_type> cached_tensorimpls;
std::mutex cached_tensorimpl_mutex;


bool is_cached_tensor(const at::Tensor& t) {
  if (!cached_tensorimpls_enabled) {
    return false;
  }
  const std::lock_guard<std::mutex> lock(cached_tensorimpl_mutex);
  return cached_tensorimpls.count(t.unsafeGetTensorImpl());
}

void add_cached_tensor(const at::Tensor& t) {
  TORCH_INTERNAL_ASSERT(cached_tensorimpls_enabled);
  const std::lock_guard<std::mutex> lock(cached_tensorimpl_mutex);
  cached_tensorimpls.emplace(t.unsafeGetTensorImpl(), weakref_type(t.getIntrusivePtr()));
}

void remove_cached_tensor(const at::Tensor& t) {
  TORCH_INTERNAL_ASSERT(cached_tensorimpls_enabled);
  const std::lock_guard<std::mutex> lock(cached_tensorimpl_mutex);
  cached_tensorimpls.erase(t.unsafeGetTensorImpl());
}

void set_cached_tensors_enabled(bool enabled) {
  cached_tensorimpls_enabled = enabled;
}

size_t adjusted_use_count(const at::Tensor& t) {
  return t.use_count() - (is_cached_tensor(t) ? 1 : 0);
}

}
}
