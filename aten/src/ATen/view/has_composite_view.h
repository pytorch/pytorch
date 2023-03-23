#pragma once

#include "c10/macros/Macros.h"

namespace at { class TensorBase; }

namespace at::view {

auto TORCH_API has_composite_view(TensorBase const& tensor) -> bool;

} // namespace at::view
