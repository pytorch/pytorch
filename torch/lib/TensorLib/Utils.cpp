#include "TensorLib/Utils.h"
#include <cstdio>
#include <stdexcept>

namespace tlib {

void runtime_error(const char *format, ...) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];

  std::sprintf(error_buf, format, ...);
  throw std::runtime_error(error_buf);
}

} // tlib
