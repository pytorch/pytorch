#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/operator.h"

#include <sstream>

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
        "aten::Size(int[] sizes) -> int[]",
        [](Stack& stack) {
          return 0;
        }),
    Operator(
        FunctionSchema(
          Symbol::fromQualString("aten::format"),
          {Argument("self", StringType::get()), Argument("args", StringType::get())},
          {Argument("", StringType::get())},
          true),
        [](Node* node) {
          size_t num_inputs = node->inputs().size();
          return [num_inputs](Stack& stack) {
            std::stringstream ss;

            auto format = peek(stack, 0, num_inputs).toStringRef();
            auto args = last(stack, num_inputs - 1);
            size_t current_arg = 0;
            bool prev_was_curly_left = false;

            // Iterate over string until '{}' pair, then try to get a value for
            // it from the varargs
            for (char& c : format) {
              if (prev_was_curly_left) {
                if (c == '}') {
                  // write arg
                  if (current_arg >= args.size()) {
                    AT_ERROR("Not enough args for format string!");
                  }
                  ss << args[current_arg];
                  prev_was_curly_left = false;
                  ++current_arg;
                  continue;
                } else {
                  // Skipped writing the '{', so do it now
                  ss << '{';
                }
              }

              prev_was_curly_left = c == '{';
              if (!prev_was_curly_left) {
                ss << c;
              }
            }

            drop(stack, num_inputs);
            stack.push_back(ss.str());
            std::cout << std::endl;
            return 0;
          };
        })
});
}
} // namespace jit
} // namespace torch
