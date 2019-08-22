#include <torch/csrc/jit/function.h>

#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {
struct RecursiveMethodCallError : public std::exception {};
void placeholderCreator(Function&) {
  throw RecursiveMethodCallError();
}

void Function::ensure_defined(bool multiple_output) {
  try {
    if (function_creator_) {
      auto creator = function_creator_;
      function_creator_ = placeholderCreator;
      creator(*this);
      function_creator_ = nullptr;
    }
  } catch (RecursiveMethodCallError&) {
    throw script::ErrorReport() // TODO: once lower_first_class methods is
                                // removed re-establish callsite info for
                                // debugging
        << " method '" << name() << "' is called recursively. "
        << "Recursive calls are not supported";
  }

  if (!multiple_output) {
    check_single_output();
  }
}

} // namespace jit
} // namespace torch
