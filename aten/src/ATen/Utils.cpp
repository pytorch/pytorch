#include <ATen/Utils.h>
#include <stdarg.h>
#include <stdexcept>
#include <typeinfo>
#include <cstdlib>

namespace at {

int _crash_if_asan(int arg) {
  volatile char x[3];
  x[arg] = 0;
  return x[0];
}

} // at
