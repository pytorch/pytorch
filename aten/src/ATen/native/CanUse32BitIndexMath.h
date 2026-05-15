#pragma once
#include <c10/macros/Export.h>
#include <limits>

namespace at {
class TensorBase;
}

namespace at::native {

TORCH_API bool canUse32BitIndexMath(const at::TensorBase &t, int64_t max_elem=std::numeric_limits<int32_t>::max());

}
