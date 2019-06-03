#include <ATen/core/ATenDispatch.h>

#include <ATen/Context.h>

namespace at {

ATenDispatch & globalATenDispatch() {
  static ATenDispatch singleton;
  return singleton;
}

void ATenDispatch::initCuda() {
  globalContext().lazyInitCUDA();
}

} // namespace at
