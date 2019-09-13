#pragma once
#include <ATen/core/ivalue.h>
#include <ATen/core/function_schema.h>
#include <torch/csrc/jit/instruction.h>

#include <vector>

namespace torch{
namespace jit{
namespace mobile {
using Stack = std::vector<c10::IValue>;

class Method{
 public:
  bool run(Stack& stack);
  const std::string& name() const;
  void set_name(const std::string& name);
  void append_instruction(OpCode op, int N, int X);
  void append_opname(const std::string& name, const std::string& overload_name);
  void append_constant(const c10::IValue& constant);
  void resize_registers(int size);
 private:
  c10::IValue& reg(size_t reg);
  std::string name_;
  std::vector<Instruction> instructions_;
  std::vector<c10::OperatorName> op_names_;
  std::vector<c10::IValue> constants_;
  std::vector<c10::IValue> registers_;
};

class TORCH_API Bytecode {
 public:
  c10::IValue run_method(const std::string& method_name, Stack& stack);
  void append_method(const Method& method);
  void set_object(const c10::intrusive_ptr<c10::ivalue::Object>& object);
 private:
  Method find_method(const std::string& name);
  c10::intrusive_ptr<c10::ivalue::Object> object_;
  std::vector<Method> methods_;
};
} // namespace mobile
} // namespace torch
} // namespace jit
