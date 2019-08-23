#include <ATen/core/ATenDispatch.h>

namespace at {

ATenDispatch & globalATenDispatch() {
  static ATenDispatch singleton;
  return singleton;
}

} // namespace at
