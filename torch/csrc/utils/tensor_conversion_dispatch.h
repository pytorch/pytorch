#pragma once

// "Convert" tensor a different type and / or device

#include <ATen/ATen.h>

namespace torch { namespace utils {

at::Tensor dispatch_type_conversion(const at::Tensor & self, const at::Type & type);
at::Tensor dispatch_type_conversion(const at::Tensor & self, const at::Type & type,
                                    int device, bool non_blocking);

}} // namespace torch::utils
