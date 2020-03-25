#include "vararg_functions.h"

namespace torch {
namespace jit {

void tupleUnpack(Stack& stack) {
  auto tuple = pop(stack).toTuple();
  stack.insert(
     stack.end(),
     tuple->elements().begin(),
     tuple->elements().end());
}

void format(Stack& stack, size_t num_inputs) {
  // static const std::regex unsupported_options("\\{(.*?)\\}");
  auto format = peek(stack, 0, num_inputs).toStringRef();
  // // Temporally comment out the warning message because of
  // // "StdRegexIsAwful" internal Lint error, to prevent sev
  // // of std::regex from PT mobile.
  // if (std::regex_search(format, unsupported_options)) {
  //   TORCH_WARN("Format options are not supported.");
  // }

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

void listUnpack(Stack& stack, size_t num_outputs) {
  auto list = pop(stack).toList();
  TORCH_CHECK(
      list.size() == num_outputs,
      "Expected ",
      num_outputs,
      " elements in a list but found ",
      list.size());
  stack.insert(stack.end(), list.begin(), list.end());
}

void tupleConstruct(Stack& stack, size_t num_inputs) {
  std::vector<IValue> elems{std::make_move_iterator(stack.end() - num_inputs),
                            std::make_move_iterator(stack.end())};
  drop(stack, num_inputs);
  push(stack, c10::ivalue::Tuple::create(std::move(elems)));
}

void namedTupleConstruct(Stack& stack, at::TupleTypePtr type, size_t num_inputs) {
  std::vector<IValue> elems{std::make_move_iterator(stack.end() - num_inputs),
                            std::make_move_iterator(stack.end())};
  drop(stack, num_inputs);
  push(
      stack,
      c10::ivalue::Tuple::createNamed(std::move(elems), std::move(type)));
}

void listConstruct(Stack& stack, at::ListTypePtr type, size_t num_inputs) {
  c10::List<IValue> vals(type->getElementType());
  vals.reserve(num_inputs);
  for (size_t i = stack.size() - num_inputs; i < stack.size(); ++i) {
    vals.emplace_back(std::move(stack[i]));
  }
  drop(stack, num_inputs);
  push(stack, std::move(vals));
}

void dictConstruct(Stack& stack, at::DictTypePtr type, size_t num_inputs) {
  at::TypePtr key_type = type->getKeyType();
  at::TypePtr value_type = type->getValueType();
  auto vals = c10::impl::GenericDict(key_type, value_type);
  vals.reserve(num_inputs / 2);
  for (size_t i = 0; i < num_inputs; i += 2) {
    auto val = pop(stack);
    auto key = pop(stack);
    vals.insert_or_assign(std::move(key), std::move(val));
  }
  push(stack, std::move(vals));
}

void createObject(Stack& stack, at::ClassTypePtr type) {
  auto userObj = c10::ivalue::Object::create(
      c10::StrongTypePtr(type->compilation_unit(), type),
      type->numAttributes());
  push(stack, std::move(userObj));
}

void isinstance(
    Stack& stack,
    at::ArrayRef<at::TypePtr> types) {
  at::TypePtr ty = pop(stack).type();
  for (const at::TypePtr& candidate : types) {
    if (ty->isSubtypeOf(candidate)) {
      push(stack, true);
      return;
    }
  }

  push(stack, false);
}

void tupleSlice(Stack& stack, size_t begin, size_t end) {
  auto tuple = pop(stack).toTuple();
  std::vector<IValue> output_elems;
  output_elems.reserve(end - begin);
  for (size_t i = begin; i < end; ++i) {
    output_elems.emplace_back(tuple->elements()[i]);
  }
  push(stack, c10::ivalue::Tuple::create(std::move(output_elems)));
}

} // namespace jit
} // namespace torch
