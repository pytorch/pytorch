#include <torch/csrc/jit/compilation_arena.h>

namespace torch {
namespace jit {
void CompilationArena::unionWith(CompilationArena* other) {
  // Merge the smaller blob into the bigger one
  CompilationArena* src;
  CompilationArena* dst;
  if (other->blob_->size() > blob_->size()) {
    src = this;
    dst = other;
  } else {
    src = other;
    dst = this;
  }
  auto& srcClasses = src->blob_->classes_;
  auto& dstClasses = dst->blob_->classes_;
  dstClasses.insert(
      std::make_move_iterator(srcClasses.begin()),
      std::make_move_iterator(srcClasses.end()));

  auto& srcFunctions = src->blob_->functions_;
  auto& dstFunctions = dst->blob_->functions_;
  dstFunctions.insert(
      std::make_move_iterator(srcFunctions.begin()),
      std::make_move_iterator(srcFunctions.end()));

  // The two arenas now point to the same blob
  src->blob_ = dst->blob_;
}
} // namespace jit
} // namespace torch