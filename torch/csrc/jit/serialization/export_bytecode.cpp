#include <torch/csrc/jit/serialization/export_bytecode.h>

#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/export.h>

namespace torch {
namespace jit {

void BytecodeExportSet::add(
    const c10::QualifiedName& qn,
    ExportedFunction exported) {
  items_.emplace(qn, std::move(exported));
}

void BytecodeExportSet::update(const c10::QualifiedName& qn, bool toplevel) {
  items_.at(qn).toplevel = toplevel;
}

bool BytecodeExportSet::contains(const c10::QualifiedName& qn) const {
  return items_.find(qn) != items_.end();
}

} // namespace jit
} // namespace torch
