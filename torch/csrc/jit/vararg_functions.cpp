#include "vararg_functions.h"

namespace torch {
namespace jit {
void tensorListConstructFunc(int num_inputs, Stack& stack) {
  const size_t stack_size = stack.size();
  c10::List<at::Tensor> vals;
  vals.reserve(num_inputs);
  for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
    vals.emplace_back(std::move(stack[i]).toTensor());
  }
  drop(stack, num_inputs);
  push(stack, std::move(vals));
}

void tupleUnpackFunc(int num_outputs, Stack& stack) {
  auto tuple = pop(stack).toTuple();
  if (tuple->elements().size() != num_outputs) {
    AT_ERROR(
       "Expected a tuple of ",
       num_outputs,
       " elements, but got ",
       tuple->elements().size());
  }
  stack.insert(
     stack.end(),
     tuple->elements().begin(),
     tuple->elements().end());
}

void formatFunc(int num_inputs, Stack& stack) {
  static const std::regex unsupported_options("\\{(.*?)\\}");
  auto format = peek(stack, 0, num_inputs).toStringRef();
  if (std::regex_search(format, unsupported_options)) {
    AT_WARN("Format options are not supported.");
  }

  auto args = last(stack, num_inputs - 1);
  std::stringstream ss;
  for (size_t begin = 0, used_args = 0; true; ++used_args) {
    size_t loc = format.find("{}", begin);
    if (loc == std::string::npos) {
      ss << format.substr(begin);
      break;
    }
    ss << format.substr(begin, loc - begin);
    if (used_args >= args.size()) {
      AT_ERROR("Too few arguments for format string: ", format);
    }
    ss << args[used_args];
    begin = loc + 2;
  }

  drop(stack, num_inputs);
  push(stack, ss.str());
}
} // namespace jit
} // namespace torch
