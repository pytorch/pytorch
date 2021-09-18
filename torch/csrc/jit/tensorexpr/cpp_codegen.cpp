#include <torch/csrc/jit/tensorexpr/cpp_codegen.h>

namespace torch {
namespace jit {
namespace tensorexpr {

void CppPrinter::visit(AllocatePtr alloc) {
  constexpr size_t kAllocOnStackThresholdSize = 512;

  size_t size = 1;
  for (auto dim : alloc->dims()) {
    IntImmPtr v = to<IntImm>(dim);
    if (v) {
      size *= v->value();
    } else {
      throw std::runtime_error("Only IntImm dimensions are supported for now");
    }
  }

  emitIndent();
  if (size <= kAllocOnStackThresholdSize) {
    os() << alloc->dtype().ToCppString() << " " << (*alloc->buffer_var()) << "["
         << size << "];" << std::endl;
  } else {
    size *= alloc->dtype().byte_size();
    os() << alloc->dtype().ToCppString() << "* " << (*alloc->buffer_var())
         << " = static_cast<" << alloc->dtype().ToCppString() << "*>(malloc("
         << size << "));" << std::endl;
    allocated_on_heap_.insert(alloc->buffer_var());
  }
}

void CppPrinter::visit(FreePtr free) {
  VarPtr var = free->buffer_var();
  if (allocated_on_heap_.count(var)) {
    emitIndent();
    os() << "free(" << name_manager()->get_unique_name(var) << ");"
         << std::endl;
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
