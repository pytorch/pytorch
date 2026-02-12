#include <torch/nativert/graph/GraphPasses.h>

#include <unordered_set>

#include <fmt/format.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>

#include <c10/util/StringUtil.h>

namespace torch::nativert {
namespace {
bool isScalar(const Constant& c) {
  return std::holds_alternative<int64_t>(c) ||
      std::holds_alternative<double>(c);
}

bool isScalar(const Value& v) {
  return v.type() == Type::Kind::SymInt || v.type() == Type::Kind::SymFloat;
}

bool schemaTypeMatch(const c10::FunctionSchema& schema, const Node& node) {
  std::unordered_set<std::string> inputNames;
  for (const auto& input : node.inputs()) {
    // The number of arguments is always O(10), so we can just do a linear scan.
    for (const auto& schemaArg : schema.arguments()) {
      if (schemaArg.name() == input.name) {
        if (schemaArg.type() == c10::TensorType::get() && input.value &&
            isScalar(*input.value)) {
          return false;
        }
        break;
      }
    }
    inputNames.insert(input.name);
  }
  for (const auto& constant : node.attributes()) {
    for (const auto& schemaArg : schema.arguments()) {
      if (schemaArg.name() == constant.name) {
        if (schemaArg.type() == c10::TensorType::get() &&
            isScalar(constant.value)) {
          return false;
        }
        break;
      }
    }
    inputNames.insert(constant.name);
  }

  // Make sure we have all the required arguments.
  for (const auto& schemaArg : schema.arguments()) {
    if (!schemaArg.default_value()) {
      if (inputNames.find(schemaArg.name()) == inputNames.end()) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

// PT2 intentionally broadcast things like aten.sub.Scalar
// to aten.sub.Tensor. https://github.com/pytorch/pytorch/issues/90923.
std::string selectScalarOverloadName(const Node& node) {
  // Copied from torch/csrc/utils/python_arg_parser.cpp
  // torch::should_allow_numbers_as_tensors() to workaround
  // some linking issues.
  static std::unordered_set<std::string> allowed = {
      "add",
      "add_",
      "add_out",
      "div",
      "div_",
      "div_out",
      "divide",
      "divide_",
      "divide_out", // alias of div
      "mul",
      "mul_",
      "mul_out",
      "multiply",
      "multiply_",
      "multiply_out", // alias of mul
      "sub",
      "sub_",
      "sub_out",
      "subtract",
      "subtract_",
      "subtract_out", // alias of sub
      "true_divide",
      "true_divide_",
      "true_divide_out",
      "to",
      "_to_copy",
      "copy_",
      "copy",
      "floor_divide",
      "floor_divide_",
      "floor_divide_out",
      "_conj"};
  std::vector<std::string_view> atoms = c10::split(node.target(), '.');

  if (atoms.size() < 3) {
    return "";
  }

  std::string ns = std::string{atoms[atoms.size() - 3]};
  std::string opName = std::string{atoms[atoms.size() - 2]};
  std::string overloadName = std::string{atoms[atoms.size() - 1]};
  if (overloadName != "Tensor" && overloadName != "Tensor_Tensor" &&
      overloadName != "Tensor_mode") {
    return overloadName;
  }
  if (allowed.find(opName) == allowed.end()) {
    return overloadName;
  }
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      fmt::format("{}::{}", ns, opName.c_str()).c_str(), overloadName.c_str());
  if (schemaTypeMatch(op.schema(), node)) {
    return overloadName;
  }
  for (const auto& variant :
       {"Scalar_mode", "Scalar", "Scalar_Tensor", "Tensor_Scalar"}) {
    if (auto schema = c10::Dispatcher::singleton().findSchema(
            {fmt::format("{}::{}", ns, opName.c_str()), variant})) {
      if (schemaTypeMatch(schema->schema(), node)) {
        return variant;
      }
    }
  }
  return overloadName;
}

void selectScalarOverload(Graph* graph) {
  for (auto& node : graph->nodes()) {
    for (auto& attr : node.attributes()) {
      if (std::holds_alternative<std::unique_ptr<Graph>>(attr.value)) {
        selectScalarOverload(
            std::get<std::unique_ptr<Graph>>(attr.value).get());
      }
    }

    auto target = node.target();
    std::vector<std::string_view> atoms = c10::split(target, '.');

    size_t numAtoms = atoms.size();
    if (numAtoms != 5) {
      continue;
    }

    const std::string_view ns = atoms[numAtoms - 3];
    const std::string_view opName = atoms[numAtoms - 2];
    if (atoms[0] != "torch" || atoms[1] != "ops" || ns != "aten") {
      continue;
    }

    auto overloadName = selectScalarOverloadName(node);
    if (overloadName != atoms[numAtoms - 1]) {
      node.setTarget(
          fmt::format("torch.ops.{}.{}.{}", ns, opName, overloadName));
    } else if (ns == "aten" && opName == "sub" && overloadName == "Tensor") {
      // Special case for aten.sub.Tensor.
      if (auto i = node.tryGetInput("self")) {
        if (isScalar(*i->value)) {
          node.updateInputName("self", "other");
          node.updateInputName("other", "self");
          node.setTarget("torch.ops.aten.rsub.Scalar");
        }
      }
      if (auto a = node.tryGetAttribute("self")) {
        if (isScalar(a->value)) {
          node.updateAttributeName("self", "other");
          node.updateInputName("other", "self");
          node.setTarget("torch.ops.aten.rsub.Scalar");
        }
      }
    }
  }
}

} // namespace torch::nativert
