#pragma once

#include <ATen/core/Tensor.h>

namespace at::mps {

bool _is_pinned_ptr(const void* data);

}
