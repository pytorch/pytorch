#include <c10/core/DispatchKeySet.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>

namespace c10 {
namespace impl {

void PythonDispatcherTLS::set_state(bool state) {
  if (state) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, true);
  } else {
    PythonDispatcherTLS::reset_state();
  }
}

bool PythonDispatcherTLS::get_state() {
  return c10::impl::tls_is_dispatch_key_included(DispatchKey::PythonDispatcher);
}

void PythonDispatcherTLS::reset_state() {
  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::PythonDispatcher, false);
}

} // namespace impl
} // namespace c10
