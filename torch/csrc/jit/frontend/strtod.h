
copy: fbcode/caffe2/torch/csrc/jit/frontend/strtod.h
copyrev: d9babb5a74c1fccda2c3b5ac50df717f731fbfcf

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
