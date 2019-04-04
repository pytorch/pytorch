#pragma once

#include <ATen/core/Macros.h>

namespace torch {
namespace jit {
namespace script {

CAFFE2_API double strtod_c(const char *nptr, char **endptr);
CAFFE2_API float strtof_c(const char *nptr, char **endptr);

}
}
}
