#include <operator_registry.h>
#include "Functions.h"

namespace torch {
namespace executor {

namespace {
using OpArrayRef = ::at::ArrayRef<::torch::executor::Operator>;

static Operator operators_to_register[] = {
    ${unboxed_ops} // Generated operators
};

// Explicitly convert to ArrayRef, so that the API can take an empty C array of
// Operators.
static OpArrayRef op_array_ref(
    operators_to_register,
    operators_to_register + sizeof(operators_to_register) / sizeof(Operator));

// Return value not used. Keep the static variable assignment to register
// operators in static initialization time.
static auto success_with_op_reg = register_operators(op_array_ref);
} // namespace
} // namespace executor
} // namespace torch
