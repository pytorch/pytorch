#include <ATen/core/dispatch/OpSchema.h>
#include <c10/util/Array.h>
#include <ATen/core/Tensor.h>

using namespace c10;
using at::Tensor;

static_assert(details::is_tensor_arg<Tensor>::value, "");
static_assert(details::is_tensor_arg<const Tensor&>::value, "");
static_assert(details::is_tensor_arg<Tensor&&>::value, "");
static_assert(!details::is_tensor_arg<int>::value, "");

struct SchemaDef final {
  using Signature = bool(int, Tensor, float, Tensor, Tensor, unsigned int);
  static constexpr guts::array<const char*, 6> parameter_names = {{
      "1", "2", "3", "4", "5", "6"
  }};
  static constexpr size_t num_dispatch_args() {return 3;}
  static constexpr size_t num_outputs() {return 0;}
};
static_assert(6 == OpSchema<SchemaDef>::signature::num_args, "");
static_assert(3 == OpSchema<SchemaDef>::signature::num_tensor_args, "");
static_assert(std::is_same<bool, typename OpSchema<SchemaDef>::signature::return_type>::value, "");
static_assert(
    std::is_same<
        guts::typelist::
            typelist<int, Tensor, float, Tensor, Tensor, unsigned int>,
        typename OpSchema<SchemaDef>::signature::parameter_types>::value,
    "");
