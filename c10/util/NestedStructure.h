#pragma once

#include <c10/macros/Export.h>
#include <ATen/core/ivalue.h>

#include <utility>
#include <functional>

namespace c10::utils {

C10_API std::pair<List<IValue>, std::function<IValue(const List<IValue>&)>>
tree_flatten(const IValue& input) {
  return nested::tree_flatten(input);
}

namespace nested {

std::pair<List<IValue>, std::function<IValue(const List<IValue>&)>> tree_flatten(
    const IValue& input);

} // namespace nested

} // namespace c10::utils
