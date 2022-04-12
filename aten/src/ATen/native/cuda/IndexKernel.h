#pragma once
#include <c10/core/ScalarType.h>
#include <cstdint>

namespace at {
struct TensorIteratorBase;
class TensorBase;
}

namespace at {
namespace native {
void launch_masked_scatter_kernel(
    const TensorBase &self, const TensorBase &mask,
    const TensorBase &maskPrefixSum, const TensorBase &source);
}}
