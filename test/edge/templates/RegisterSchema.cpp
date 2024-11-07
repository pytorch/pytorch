// ${generated_comment}
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/library.h>

namespace at {
TORCH_LIBRARY_FRAGMENT(aten, m) {
    ${aten_schema_registrations};
}
$schema_registrations
} // namespace at
