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


  std::cout << "check_single_output222 = \n";
  if (this->graph()->outputs().size() != 1)
  {
    std::cout << "check_single_output = \n";
    this->graph()->dump();
  }
  
  std::cout << "multiple_output begin = " << multiple_output << std::endl;
  if (!multiple_output)
  {
    check_single_output();
  }
  std::cout << "multiple_output false = " << multiple_output << std::endl;
}

} // namespace jit
} // namespace torch
