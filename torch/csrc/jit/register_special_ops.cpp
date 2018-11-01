#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/operator.h"

#include <sstream>
#include <regex>

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
        "aten::format(str self, *str args) -> str",
        [](const Node* node) {
          size_t num_inputs = node->inputs().size();
          std::regex unsupported_options("\\{(.*)\\}");
          return [num_inputs, unsupported_options](Stack& stack) {
            auto format = peek(stack, 0, num_inputs).toStringRef();

            if (std::regex_search(format, unsupported_options)) {
              AT_WARN("Format options are not supported.");
            }

            auto args = last(stack, num_inputs - 1);

            size_t base = 0;
            std::string format_arg = "{}";
            size_t next = format.find(format_arg, base);
            size_t used_args = 0;
            std::stringstream ss;

            while (next != std::string::npos) {
              if (used_args == args.size()) {
                AT_ERROR("Too few arguments for format string: ", format);
              }
              ss << format.substr(base, next - base);
              ss << args[used_args++];
              base = next + format_arg.length();
              next = format.find(format_arg, base);
            }

            drop(stack, num_inputs);
            stack.push_back(ss.str());
            return 0;
          };
        })
});
}
} // namespace jit
} // namespace torch
