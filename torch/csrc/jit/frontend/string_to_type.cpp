#include <torch/csrc/jit/frontend/string_to_type.h>

namespace torch {
namespace jit {
using namespace c10;
TORCH_API const std::unordered_map<std::string, TypePtr>& string_to_type_lut() {
  static std::unordered_map<std::string, TypePtr> map = {
#define MAP_ITEM(NAME, TYPE) {#NAME, TYPE##Type::get()},
      FORALL_JIT_BASE_TYPES(MAP_ITEM)
#undef MAP_ITEM
  };
  return map;
}
} // namespace jit
} // namespace torch
