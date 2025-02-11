#include <c10/util/NestedStructure.h>

#include <functional>
#include <utility>

namespace c10::utils::nested {
template <typename T>
std::pair<List<IValue>, std::function<T(const List<IValue>&)>>
tree_flatten_one_level(const T& input) {
  TORCH_CHECK(false, "Unsupported IValue type");
}

template <>
std::pair<List<IValue>, std::function<Tuple<IValue>(const List<IValue>&)>>
tree_flatten_one_level(const Tuple<IValue>& input) {
  return {List(input.asArrayRef()), [](const List<IValue>& children) -> IValue {
            return IValue(Tuple(children.vec()));
          }};
}

template <>
std::pair<List<IValue>, std::function<List<IValue>(const List<IValue>&)>>
tree_flatten_one_level(const List<IValue>& input) {
  return {input.copy(), [](const List<IValue>& children) -> IValue {
            return IValue(children.copy());
          }};
}

template <>
std::pair<List<IValue>, std::function<Dict<IValue, IValue>(const List<IValue>&)>>
tree_flatten_one_level(const Dict<IValue, IValue>& input) {
  auto keys = List<IValue>();
  auto values = List<IValue>();
  for (const auto& it : input) {
    keys.emplace_back(it.key());
    values.emplace_back(it.value());
  }
  return {values, [const & keys](const List<IValue>& children) {
            TORCH_CHECK(
                children.size() == keys.size(),
                "Node arity mismatch for Dict node");
            auto result = Dict<IValue, IValue>();
            for (size_t i = 0; i < children.size(); ++i) {
              result.insert(keys, children[i]);
            }
            return IValue(result);
          }};
}

std::pair<List<IValue>, std::function<IValue(const List<IValue>&)>>
tree_flatten_one_level(const IValue& input) {
  if (input.isTuple()) {
    return tree_flatten_one_level<Tuple<IValue>>(input.toTupleRef());
  }
  if (input.isList()) {
    return tree_flatten_one_level<List<IValue>>(input.toList());
  }
  if (input.isGenericDict()) {
    return tree_flatten_one_level<Dict<IValue, IValue>>(input.toGenericDict());
  }

  // Leaf node
  return {List({input}), [](const List<IValue>& children) -> IValue {
            TORCH_CHECK(
                children.size() == 1, "Node arity mismatch for Leaf node");
            return children[0];
          }};
}

std::function<IValue(const List<IValue>&)> tree_flatten_helper(
    const IValue& input,
    List<IValue>& leaves);

std::pair<List<IValue>, std::function<IValue(const List<IValue>&)>> tree_flatten(
    const IValue& input) {
  List<IValue> leaves{};
  auto unflatten_func = tree_flatten_helper(input, leaves);
  return {leaves, unflatten_func};
}

} // namespace c10::utils::nested
