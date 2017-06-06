#include "TensorLib/Utils.h"
#include <stdarg.h>
#include <stdexcept>
#include <typeinfo>

namespace tlib {

void runtime_error(const char *format, ...) {
  static const size_t ERROR_BUF_SIZE = 1024;
  char error_buf[ERROR_BUF_SIZE];

  va_list fmt_args;
  va_start(fmt_args, format);
  vsnprintf(error_buf, ERROR_BUF_SIZE, format, fmt_args);
  va_end(fmt_args);

  throw std::runtime_error(error_buf);
}

} // tlib
