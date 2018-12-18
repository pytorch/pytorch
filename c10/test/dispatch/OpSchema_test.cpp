#include <c10/core/dispatch/OpSchema.h>
#include <c10/util/Array.h>

using namespace c10;

static_assert(details::is_tensor_arg<C10Tensor>::value, "");
static_assert(details::is_tensor_arg<const C10Tensor&>::value, "");
static_assert(details::is_tensor_arg<C10Tensor&&>::value, "");
static_assert(!details::is_tensor_arg<int>::value, "");

struct SchemaDef final {
  using Signature = bool(int, C10Tensor, float, C10Tensor, C10Tensor, unsigned int);
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
            typelist<int, C10Tensor, float, C10Tensor, C10Tensor, unsigned int>,
        typename OpSchema<SchemaDef>::signature::parameter_types>::value,
    "");
