#include <torch/csrc/jit/runtime/vararg_functions.h>

#include <ATen/ATen.h>

namespace torch {
namespace jit {

namespace {
static constexpr int defaultPrecision = 6;

// IValue tags are intentionally private, so we need additional logic to cast
// the IValue type to the specified format.
void addFormattedArg(
    char key,
    const IValue& ival,
    std::stringstream& ss,
    int precision = defaultPrecision) {
  // TODO: Implement precison-based formatting
  std::stringstream tmp;
  switch (key) {
    case 'd':
    case 'i':
      TORCH_CHECK(
          ival.isScalar(),
          "%",
          key,
          " requires a number for formatting, but got ",
          ival.tagKind());
      if (ival.isInt()) {
        ss << ival.toInt();
      } else {
        ss << static_cast<int>(ival.toDouble());
      }
      break;
    case 'e':
    case 'E':
      TORCH_CHECK(
          ival.isScalar(),
          "%",
          key,
          " requires a number for formatting, but got ",
          ival.tagKind());
      tmp << std::setprecision(precision) << std::scientific;
      if (key == 'E') {
        tmp << std::uppercase;
      }
      if (ival.isInt()) {
        tmp << static_cast<float>(ival.toInt());
      } else {
        tmp << static_cast<float>(ival.toDouble());
      }
      ss << tmp.str();
      break;
    case 'f':
    case 'F':
      TORCH_CHECK(
          ival.isScalar(),
          "%",
          key,
          " requires a number for formatting, but got ",
          ival.tagKind());
      tmp << std::setprecision(precision) << std::fixed;
      if (ival.isInt()) {
        tmp << static_cast<float>(ival.toInt());
      } else {
        tmp << static_cast<float>(ival.toDouble());
      }
      ss << tmp.str();
      break;
    case 'c':
      TORCH_CHECK(
          ival.isInt() || (ival.isString() && ival.toStringRef().length() == 1),
          "%",
          key,
          " requires an int or char for formatting, but got ",
          ival.tagKind());
      if (ival.isInt()) {
        ss << static_cast<char>(ival.toInt());
      } else {
        ss << ival.toStringRef();
      }
      break;
    case 's':
      if (ival.isString()) {
        ss << ival.toStringRef();
      } else {
        ss << ival;
      }
      break;
    default:
      TORCH_CHECK(
          false,
          "The specifier %",
          key,
          " is not supported in TorchScript format strings");
  }
}

} // namespace

void tupleUnpack(Stack& stack) {
  auto tuple = pop(stack).toTuple();
  stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());
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

void percentFormat(Stack& stack, size_t num_inputs) {
  auto format_str = peek(stack, 0, num_inputs).toStringRef();
  auto args = last(stack, num_inputs - 1)[0];
  auto args_size = 1; // assumed size
  if (args.isTuple()) {
    args_size = args.toTuple()->elements().size();
  }
  std::stringstream ss;
  size_t used_args = 0;
  size_t begin = 0;
  while (true) {
    size_t percent_idx = format_str.find('%', begin);
    if (percent_idx == std::string::npos) {
      ss << format_str.substr(begin);
      break;
    }
    size_t format_idx = percent_idx + 1;
    TORCH_CHECK(
        percent_idx < format_str.length() - 1, "Incomplete format specifier");
    ss << format_str.substr(begin, percent_idx - begin);
    if (format_str.at(format_idx) == '%') {
      ss << '%';
      begin = percent_idx + 2; // skip the `%` and the format specifier
      continue;
    }
    TORCH_CHECK(used_args < args_size, "Too few arguments for format string");
    char key = format_str.at(format_idx);
    IValue arg;
    if (args.isTuple()) {
      arg = args.toTuple()->elements()[used_args];
    } else {
      arg = args;
    }
    addFormattedArg(key, arg, ss);
    begin = percent_idx + 2;
    ++used_args;
  }
  TORCH_CHECK(used_args == args_size, "Too many arguments for format string");
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
  std::vector<IValue> elems{
      std::make_move_iterator(stack.end() - num_inputs),
      std::make_move_iterator(stack.end())};
  drop(stack, num_inputs);
  push(stack, c10::ivalue::Tuple::create(std::move(elems)));
}

void namedTupleConstruct(
    Stack& stack,
    at::TupleTypePtr type,
    size_t num_inputs) {
  std::vector<IValue> elems{
      std::make_move_iterator(stack.end() - num_inputs),
      std::make_move_iterator(stack.end())};
  drop(stack, num_inputs);
  push(
      stack,
      c10::ivalue::Tuple::createNamed(std::move(elems), std::move(type)));
}

void listConstruct(Stack& stack, const at::ListType& type, size_t num_inputs) {
  // Structuring the implementation this way allows NRVO to avoid
  // move-constructing vals on its way onto the stack. Moving a List
  // isn't free.
  auto makeList =
      [](Stack& stack, const at::ListType& type, size_t num_inputs) {
        c10::List<IValue> vals(type.getElementType());
        vals.reserve(num_inputs);
        for (size_t i = stack.size() - num_inputs; i < stack.size(); ++i) {
          vals.push_back(std::move(stack[i]));
        }
        drop(stack, num_inputs);
        return vals;
      };
  stack.push_back(makeList(stack, type, num_inputs));
}

void dictConstruct(
    Stack& stack,
    const at::DictTypePtr& type,
    size_t num_inputs) {
  at::TypePtr key_type = type->getKeyType();
  at::TypePtr value_type = type->getValueType();
  auto vals = c10::impl::GenericDict(key_type, value_type);
  vals.reserve(num_inputs / 2);
  // loop from the bottom of the stack to ensure the dictConstruct preserve
  // the inputs order.
  auto inputs = last(stack, num_inputs);
  for (size_t i = 0; i < num_inputs; i += 2) {
    auto key = inputs[i];
    auto val = inputs[i + 1];
    vals.insert_or_assign(std::move(key), std::move(val));
  }
  drop(stack, num_inputs);
  push(stack, std::move(vals));
}

void createObject(Stack& stack, const at::ClassTypePtr& type) {
  auto userObj = c10::ivalue::Object::create(
      c10::StrongTypePtr(type->compilation_unit(), type),
      type->numAttributes());
  push(stack, std::move(userObj));
}

void isinstance(Stack& stack, at::ArrayRef<at::TypePtr> types) {
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

void dequantize(Stack& stack) {
  auto iv = pop(stack);
  if (iv.isTuple()) {
    auto tuple = iv.toTuple();
    auto elems = tuple->elements();
    std::vector<IValue> output_elems;
    output_elems.reserve(elems.size());
    for (const auto& elem : elems) {
      if (elem.isTensor()) {
        output_elems.emplace_back(at::dequantize(elem.toTensor()));
      } else {
        output_elems.emplace_back(elem);
      }
    }
    push(stack, c10::ivalue::Tuple::create(std::move(output_elems)));
  } else if (iv.isTensorList()) {
    auto elems = iv.toTensorList();
    auto output_list = c10::impl::GenericList(elems.elementType());
    for (auto&& elem : elems) {
      output_list.emplace_back(at::dequantize(elem));
    }
    push(stack, std::move(output_list));
  } else {
    TORCH_CHECK(
        false,
        "Unsupported type in dequantize, only List[Tensor] and \
 Tuple[Tensor or other types] are supported, got type:",
        toString(iv.type()));
  }
}

} // namespace jit
} // namespace torch
