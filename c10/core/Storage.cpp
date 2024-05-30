#include <c10/core/RefcountedDeleter.h>
#include <c10/core/Storage.h>

#include <array>

namespace c10 {

// For two different StorageImpls to be considered aliases of each other, they
// must have the same deleter function and deleter context. Also, the deleter
// must be one of the expected types.
bool isSharedStorageAlias(const Storage& storage0, const Storage& storage1) {
  std::array<c10::DeleterFnPtr, 2> allowed_deleters = {
      &c10::refcounted_deleter, &c10::impl::cow::cowsim_deleter};
  c10::DeleterFnPtr deleter0 = storage0.data_ptr().get_deleter();
  c10::DeleterFnPtr deleter1 = storage1.data_ptr().get_deleter();

  if (deleter0 != deleter1) {
    return false;
  }

  bool is_deleter_allowed = false;

  for (c10::DeleterFnPtr deleter_check : allowed_deleters) {
    if (deleter0 == deleter_check) {
      is_deleter_allowed = true;
      break;
    }
  }

  if (!is_deleter_allowed) {
    return false;
  }

  return storage0.data_ptr().get_context() == storage1.data_ptr().get_context();
}

} // namespace c10
