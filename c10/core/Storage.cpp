#include <c10/core/RefcountedDeleter.h>
#include <c10/core/Storage.h>

namespace c10 {

bool isSharedStorageAlias(const Storage& storage0, const Storage& storage1) {
  c10::DeleterFnPtr deleter_expected = &c10::refcounted_deleter;
  c10::DeleterFnPtr deleter0 = storage0.data_ptr().get_deleter();
  c10::DeleterFnPtr deleter1 = storage1.data_ptr().get_deleter();

  if ((deleter0 != deleter_expected) || (deleter1 != deleter_expected)) {
    return false;
  }

  return storage0.data_ptr().get_context() == storage1.data_ptr().get_context();
}

} // namespace c10
