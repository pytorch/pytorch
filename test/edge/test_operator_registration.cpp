#include "operator_registry.h"
#include <torch/torch.h>
#include <gtest/gtest.h>

namespace torch {
namespace executor {

// add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
TEST(OperatorRegistrationTest, Add) {
    EValue values[4];
    values[0] = EValue(at::ones({2, 3}));
    values[1] = EValue(at::ones({2, 3}));
    values[2] = EValue(1);

    auto op = getOpsFn("aten::add.out");
    EValue* kernel_values[4];
    for (size_t i = 0; i < 4; i++) {
        kernel_values[i] = &values[i];
    }
    op(kernel_values);
    at::Tensor expected = 2 * at::ones({2, 3});
    ASSERT_EQ(values[3].toTensor(), expected);

}
} // namespace executor
} // namespace torch
