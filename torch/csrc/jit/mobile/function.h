#pragma once
#include <aten/src/ATen/core/ivalue.h>
//#include <aten/src/Aten/core/operator_name.h>
#include <vector>

namespace torch{
namespace jit{
using Stack = std::vector<c10::IValue>;
enum OpCode : uint8_t;

namespace mobile {
struct Code;

class Function{
 public:
  Function(c10::QualifiedName name);
  bool run(Stack& stack) const;
  const std::string& name() const;
  const c10::QualifiedName& qualname() const;
  void append_instruction(OpCode op, int N, int X);
  void append_operator(const std::string& name, const std::string& overload_name);
  void append_constant(const c10::IValue& constant);
  void set_register_size(size_t size);

 private:
  c10::QualifiedName name_;
  std::shared_ptr<Code> code_;
};

} // namespace mobile
} // namespace torch
} // namespace jit
