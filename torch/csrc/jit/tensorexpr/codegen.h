#pragma once

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
class PaddedBuffer;

class CodeGen {
 public:
  class BufferArg;
  class CallArg;

  template <typename... Ts>
  CodeGen(Stmt* stmt, Ts... ts)
      : stmt_(stmt), buffer_args_({BufferArg(ts)...}) {}

  CodeGen(Stmt* stmt, const std::vector<BufferArg>& buffer_args)
      : stmt_(stmt), buffer_args_(buffer_args) {}

  virtual ~CodeGen() {}

  Stmt* stmt() const {
    return stmt_;
  }

  std::vector<BufferArg>& buffer_args() {
    return buffer_args_;
  }

  const std::vector<BufferArg>& buffer_args() const {
    return buffer_args_;
  }

  TORCH_API virtual void call(const std::vector<CallArg>& args) {
    LOG(FATAL) << "unimplemented call";
  }

 private:
  Stmt* stmt_;
  std::vector<BufferArg> buffer_args_;
};

class CodeGen::BufferArg {
 public:
  BufferArg(const Buffer& buffer)
      : var_(buffer.data()), dtype_(buffer.dtype()) {}
  BufferArg(Tensor* tensor)
      : var_(tensor->function()->func_var()),
        dtype_(tensor->function()->body()->dtype()) {}
  BufferArg(const Function& func)
      : var_(func.func_var()), dtype_(func.body()->dtype()) {}
  BufferArg(const VarHandle& var) : var_(var.node()), dtype_(var.dtype()), isVar_(true) {}

  const Var* var() const {
    return var_;
  }
  Dtype dtype() const {
    return dtype_;
  }

  bool isVar() const {
    return isVar_;
  }

 private:
  const Var* var_;
  Dtype dtype_;
  bool isVar_{false};
};

class CodeGen::CallArg {
 public:
  template <typename T>
  CallArg(const PaddedBuffer<T>& buffer);

  template <typename T>
  CallArg(const std::vector<T>& buffer) : ptr_(const_cast<T*>(buffer.data())) {}

  CallArg(void* ptr) : ptr_(ptr) {}

  CallArg(int32_t i) : ival_(i) {}

  CallArg(float f) : fval_(f) {}

  void* data() const {
    return ptr_;
  }

  int32_t intData() const {
    return ival_;
  }

  float floatData() const {
    return fval_;
  }

  int* intPtr() const {
    return const_cast<int*>(&ival_);
  }

  float* floatPtr() const {
    return const_cast<float*>(&fval_);
  }

 private:
  union {
    void* ptr_;
    float fval_;
    int32_t ival_;
  };
};

class RegisterCodeGenList {
 public:
  TORCH_API static RegisterCodeGenList& GetInstance() {
    static RegisterCodeGenList codegen_list;
    return codegen_list;
  }

  using StmtFactoryMethod = std::function<std::unique_ptr<CodeGen>(
      Stmt* stmt,
      const std::vector<CodeGen::BufferArg>&)>;

  TORCH_API StmtFactoryMethod FindStmtFactoryMethod(const std::string& name);

 private:
  template <class CodeGenType>
  friend class RegisterCodeGen;
  RegisterCodeGenList() {}
  TORCH_API void AddStmtFactoryMethod(
      const std::string& name,
      const StmtFactoryMethod& stmt_factory_method);
  RegisterCodeGenList(const RegisterCodeGenList&) = delete;
  RegisterCodeGenList& operator=(const RegisterCodeGenList&) = delete;

  std::unordered_map<std::string, StmtFactoryMethod> stmt_factory_methods_;
};

template <class CodeGenType>
class RegisterCodeGen {
 public:
  explicit RegisterCodeGen(const std::string& name) {
    RegisterCodeGenList& codegen_list = RegisterCodeGenList::GetInstance();
    codegen_list.AddStmtFactoryMethod(
        name,
        [](Stmt* stmt, const std::vector<CodeGen::BufferArg>& params) {
          std::unique_ptr<CodeGen> method(new CodeGenType(stmt, params));
          return method;
        });
  }
};

TORCH_API std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    Stmt* stmt,
    const std::vector<CodeGen::BufferArg>& params);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
