#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <c10/util/irange.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class Fusion;
namespace kir {
class Kernel;
class Scope;
} // namespace kir

void checkInlineable(const Expr* expr);
static constexpr char const* kTab = "  ";

// Indent the generated code
inline std::ostream& indent(std::ostream& os, int indent_size) {
  for (const auto _ : c10::irange(indent_size)) {
    (void)_; // Suppress unused variable warning
    os << "  ";
  }
  return os;
}

//! Define pretty printing functions for IR nodes
//!
//! This class is intended for debug printing, so it attempts
//! to handle invalid states as well.
//!
class TORCH_CUDA_CU_API IrPrinter {
 public:
  explicit IrPrinter(std::ostream& os, int indent_size = 0)
      : os_(os), indent_size_(indent_size) {}
  virtual ~IrPrinter() {}

  void resetIndent() {
    indent_size_ = 0;
  }

  bool printInline() const {
    return print_inline_;
  }

  virtual void handle(Fusion* f);

  // handle calls some non const fusion ops,
  // eventhough fusion should remain unchanged.
  // Need to look into this.
  virtual void handle(const Fusion* f) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    handle(const_cast<Fusion*>(f));
  }

  virtual void handle(Fusion& f) {
    handle(&f);
  }

  virtual void handle(const kir::Kernel* kernel);
  virtual void handle(kir::Kernel& kernel);

 protected:
  std::ostream& os() {
    return os_;
  }

 private:
  std::ostream& os_;
  bool print_inline_ = false;
  int indent_size_ = 0;
};

TORCH_CUDA_CU_API std::ostream& operator<<(
    std::ostream& os,
    const Statement* stmt);

TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream& os, Fusion* f);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream& os, Fusion& f);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
