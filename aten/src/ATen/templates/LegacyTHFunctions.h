#pragma once

// ${generated_comment}

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

namespace c10 {
class Scalar;
}
namespace at {
struct GeneratorImpl;
typedef std::shared_ptr<GeneratorImpl> Generator;
class Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {
namespace legacy {
namespace ${namespace} {

${legacy_th_declarations}

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
