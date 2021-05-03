#include <aten/src/Aten/core/jit_type_base.h>

namespace torch {
namespace jit {

struct Dummy;

using DummyPtr = std::shared_ptr<Dummy>;

struct TORCH_API Dummy : public c10::Type {
  static DummyPtr create() {
    return DummyPtr(new Dummy()); // NOLINT(modernize-make-shared)
  }
  bool operator==(const Type& rhs) const override {
    return rhs.kind() == kind();
  }
  std::string str() const override {
    return "Dummy"; // match what PythonArgParser says for clarity
  }
  static const at::TypeKind Kind = at::TypeKind::ListType;
  // global singleton
  //  static DummyPtr get();

  Dummy(at::TypeKind kind = at::TypeKind::ListType) : Type(kind) {}

 protected:
  std::string annotation_str_impl(
      at::TypePrinter printer = nullptr) const override {
    return "Dummy";
  }
};

} // namespace jit
} // namespace torch
