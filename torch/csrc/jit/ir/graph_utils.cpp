#include <torch/csrc/jit/ir/graph_utils.h>

namespace torch {
namespace jit {

TypePtr getTensorType(const at::Tensor& t, bool complete) {
  auto r = TensorType::create(t);
  if (!complete) {
    r = r->dimensionedOnly();
  }
  return r;
}

TypePtr inferShapeAndTypeForInput(
    TypePtr input_type,
    Stack::const_iterator& s_iter,
    const Stack::const_iterator& s_iter_end,
    bool complete) {
  if (auto tuple_type = input_type->cast<TupleType>()) {
    std::vector<TypePtr> types;
    for (const auto& sub_type : tuple_type->containedTypes()) {
      TORCH_INTERNAL_ASSERT(s_iter != s_iter_end);
      types.emplace_back(
          inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete));
    }
    return TupleType::create(types);
  } else if (auto list_type = input_type->cast<ListType>()) {
    const TypePtr& sub_type = list_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return ListType::create(elem_type);
  } else if (auto tensor_type = input_type->cast<TensorType>()) {
    auto type = getTensorType(s_iter->toTensor(), complete);
    s_iter++;
    return type;
  } else if (auto optional_type = input_type->cast<OptionalType>()) {
    const TypePtr& sub_type = optional_type->getElementType();
    auto elem_type =
        inferShapeAndTypeForInput(sub_type, s_iter, s_iter_end, complete);
    return OptionalType::create(elem_type);
  } else {
    // Primitive type, keep as is.
    s_iter++;
    return input_type;
  }
}

void setInputTensorTypes(
    Graph& g,
    const Stack& stack,
    bool complete,
    const std::vector<int>& param_count_list) {
  at::ArrayRef<Value*> input_values = g.inputs();
  auto s_iter = stack.begin();
  size_t list_idx = 0;
  if (!param_count_list.empty()) {
    TORCH_INTERNAL_ASSERT(
        input_values.size() == param_count_list.size(),
        " input_values:",
        input_values.size(),
        " vs param_count_list:",
        param_count_list.size());
  }
  for (auto v : input_values) {
    // Leave packed param types alone. This is needed for downstream passes
    // (like alias analysis) to work properly. This will be unpacked later
    // in unpackQuantizedWeights.
    if (auto named_type = v->type()->cast<c10::NamedType>()) {
      if (auto qualname = named_type->name()) {
        if (getCustomClass(qualname->qualifiedName())) {
          if (param_count_list.empty()) {
            AT_ASSERT(s_iter != stack.end());
            s_iter++;
          } else {
            if (param_count_list[list_idx] > 0) {
              AT_ASSERT(s_iter != stack.end());
            }
            s_iter += param_count_list[list_idx];
          }
          list_idx++;
          continue;
        }
      }
    }
    auto type =
        inferShapeAndTypeForInput(v->type(), s_iter, stack.end(), complete);
    v->setType(type);
    list_idx++;
  }
}

} // namespace jit
} // namespace torch
