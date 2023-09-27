#include <c10/core/RefcountedDeleter.h>
#include <c10/core/Storage.h>
#include <c10/util/UniqueVoidPtr.h>

namespace c10 {

bool isSharedStorageAlias(const Storage& storage0, const Storage& storage1) {
  c10::DeleterFnPtr deleter0 = storage0.data_ptr().get_deleter();
  c10::DeleterFnPtr deleter1 = storage1.data_ptr().get_deleter();

  if (deleter0 != deleter1 ||
      (deleter0 != &c10::refcounted_deleter &&
       deleter0 != &c10::detail::deleteNothing)) {
    return false;
  }

  return storage0.data_ptr().get_context() == storage1.data_ptr().get_context();
}

} // namespace c10
