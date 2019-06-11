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
  auto& dstClasses = dst->blob_->classes_;
  for (auto& cls : src->blob_->classes_) {
    dstClasses.emplace_back(std::move(cls));
  }

  auto& dstFunctions = dst->blob_->functions_;
  for (auto& function : src->blob_->functions_) {
    dstFunctions.emplace_back(std::move(function));
  }

  // The two arenas now point to the same blob
  src->blob_ = dst->blob_;
}

const std::vector<c10::ClassTypePtr>& CompilationArena::getClasses()
    const {
  return blob_->classes_;
}

const std::vector<std::unique_ptr<Function>>& CompilationArena::
    getFunctions() const {
  return blob_->functions_;
}

std::vector<c10::ClassTypePtr>& CompilationArena::getClasses() {
  return blob_->classes_;
}

std::vector<std::unique_ptr<Function>>& CompilationArena::getFunctions() {
  return blob_->functions_;
}
} // namespace jit
} // namespace torch