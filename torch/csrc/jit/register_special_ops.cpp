#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/custom_operator.h"
#include "torch/csrc/jit/operator.h"
#include "torch/csrc/api/include/torch/utils.h"
#include "../ATen/ExpandUtils.h"

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
        "aten::size(Tensor self) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          auto result = (std::move(pop(stack))).toTensor().sizes();
          pack(stack, std::move(result));
          return 0;
        }),
    Operator(
        "aten::list_with_default(int[] list, int[] defaults) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          auto list = peek(stack, 0, 2).toIntListRef();
          auto defaults = peek(stack, 1, 2).toIntListRef();
          drop(stack, 2);

          JIT_ASSERT(defaults.size() > list.size());

          // TODO: allow list of optionals to be filled in with defaults
          // i.e. list_with_default([1, 2, None], [1, 2, 3]) -> [1, 2, 3]

          push(stack, list);
          return 0;
        }),
    Operator(
        "aten::format(str self, ...) -> str",
        [](const Node* node) {
          size_t num_inputs = node->inputs().size();
          std::regex unsupported_options("\\{(.*)\\}");
          return [num_inputs, unsupported_options](Stack& stack) {
            auto format = peek(stack, 0, num_inputs).toStringRef();

            if (std::regex_search(format, unsupported_options)) {
              AT_WARN("Format options are not supported.");
            }

            auto args = last(stack, num_inputs - 1);
            std::stringstream ss;
            for(size_t begin = 0, used_args = 0; true; ++used_args) {
              size_t loc = format.find("{}", begin);
              if(loc == std::string::npos) {
                ss << format.substr(begin);
                break;
              }
              ss << format.substr(begin, loc - begin);
              if(used_args >= args.size()) {
                AT_ERROR("Too few arguments for format string: ", format);
              }
              ss << args[used_args];
              begin = loc + 2;
            }

            drop(stack, num_inputs);
            push(stack, ss.str());
            return 0;
          };
        }),
    Operator(
        "aten::_infer_size(int[] a, int[] b) -> int[]",
        [](const Node* node) {
          return [](Stack& stack) {
            auto a = pop(stack).toIntList()->elements();
            auto b = pop(stack).toIntList()->elements();
            push(stack, at::infer_size(a, b));
            return 0;
          };
        }),
    Operator(
      "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
      [](const Node* node) {
        return [](Stack& stack) {
          at::Tensor weight;
          at::Tensor input;
          double max_norm;
          double norm_type;
          pop(stack, weight, input, max_norm, norm_type);

          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor result = at::embedding_renorm_(weight, input, max_norm, norm_type);
          push(stack, result);

          return 0;
        };
      }),

});
}
} // namespace jit
} // namespace torch
