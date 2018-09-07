#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER
#pragma once

#include "torch/csrc/jit/code_template.h"

namespace torch { namespace jit { namespace cpufuser {

/*with type_as not checking type of its input, a fusion group can have non-fp32 tensor as input.
Correct code for this case is generated, however, nvrtc does not know how to handle int*_t integer types,
so typedefs help it handle those cases*/

auto type_declarations_template = CodeTemplate(R"(
typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T* data;
  IndexType sizes[N];
  IndexType strides[N];
};
template<typename T>
struct TensorInfo<T, 0> {
  T * data;
};
)");

auto cpu_compilation_unit_template = CodeTemplate(R"(
#include <cstddef>
#include <cstdint>
#include <math.h>
${type_declarations}

#define OMP_THRESHOLD 100000
static void ${kernelName}_kernel(IndexType totalElements, ${formals}) {
  #pragma omp parallel for if(totalElements > OMP_THRESHOLD)
  for (IndexType linearIndex = 0;
        linearIndex < totalElements;
        linearIndex += 1) {
      // Convert `linearIndex` into an offset of tensor:
      ${tensorOffsets}
      // calculate the results
      ${kernelBody}
    }
}

extern "C"
void ${kernelName}(IndexType totalElements, void ** args) {
  ${kernelName}_kernel(totalElements ${,argument_loads});
}
)");

} // namespace cpufuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER
