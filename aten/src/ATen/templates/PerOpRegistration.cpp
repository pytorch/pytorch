// ${generated_comment}

#include <ATen/Config.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/TypeDefault.h>
$extra_headers

namespace at {

#ifndef USE_STATIC_DISPATCH
namespace {
auto registerer = torch::import()
  ${function_registrations};
}
#endif

}  // namespace at
