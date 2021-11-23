#include <c10/util/irange.h>
#include <torch/csrc/jit/passes/value_refinement_utils.h>

namespace torch {
namespace jit {

ListRefinement intersectRefinements(
    const ListRefinement& ref1,
    const ListRefinement& ref2) {
  ListRefinement out;
  for (const auto& pair : ref1) {
    auto val2 = ref2.find(pair.first);
    if (val2 != ref2.end() && val2->second == pair.second) {
      out[pair.first] = pair.second;
    }
  }
  return out;
}

ListRefinement unionRefinements(
    const ListRefinement& ref1,
    const ListRefinement& ref2) {
  ListRefinement out = ref1;
  out.insert(ref2.begin(), ref2.end());
  return out;
}

} // namespace jit
} // namespace torch
