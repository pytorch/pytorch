#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/operator.h"

namespace torch {
namespace jit {

namespace {
RegisterOperators reg({
    Operator(
        "aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("split_with_sizes");
          auto result = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toInt());
          drop(stack, 3);
          pack(stack, std::move(result));
          return 0;
        }),
    Operator(
        "aten::Size(int[] split_sizes) -> int[]",
        [](Stack& stack) {
          auto result = (std::move(peek(stack, 0, 1))).toIntList()->elements();
          drop(stack, 1);
          pack(stack, std::move(result));
          return 0;
        }),
});
}
} // namespace jit
} // namespace torch
