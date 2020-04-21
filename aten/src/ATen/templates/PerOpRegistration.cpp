// ${generated_comment}

#include <ATen/Config.h>
#include <torch/library.h>
#include <ATen/TypeDefault.h>
$extra_headers

namespace at {

TORCH_LIBRARY_FRAGMENT_THIS_API_IS_FOR_PER_OP_REGISTRATION_ONLY(aten, m) {
  ${function_registrations}
}

}  // namespace at
