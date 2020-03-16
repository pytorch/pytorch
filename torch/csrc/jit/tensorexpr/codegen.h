#pragma once

#include "torch/csrc/jit/tensorexpr/buffer.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
class PaddedBuffer;

class TORCH_API CodeGen {
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

  virtual void call(const std::vector<CallArg>& args) {
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
      : var_(tensor->function()->func_var(tensor->output_index())),
        dtype_(tensor->function()->body(tensor->output_index())->dtype()) {}
  BufferArg(const Function& func)
      : var_(func.func_var(0)), dtype_(func.body(0)->dtype()) {
    // TODO: Support multiple-output functions
    CHECK(func.func_vars().size() == 1);
  }
  BufferArg(const VarHandle& var)
      : var_(var.node()), dtype_(var.dtype()), isVar_(true) {}

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

#define ARG_TYPE_CTOR(Type, Name) \
  CallArg(Type v) : Name##val_(v) {}
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, ARG_TYPE_CTOR);
#undef ARG_TYPE_CTOR

  void* data() const {
    return ptr_;
  }

#define ARG_DATA_DEFINE(Type, Name) \
  Type Name##Data() const {         \
    return Name##val_;              \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, ARG_DATA_DEFINE);
#undef ARG_DATA_DEFINE

#define ARG_PTR_DEFINE(Type, Name)         \
  Type* Name##Ptr() const {                \
    return const_cast<Type*>(&Name##val_); \
  }
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, ARG_PTR_DEFINE);
#undef ARG_PTR_DEFINE

 private:
  union {
    void* ptr_;

#define ARG_BACKING(Type, Name) Type Name##val_;
    AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, ARG_BACKING);
#undef ARG_BACKING
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
        name, [](Stmt* stmt, const std::vector<CodeGen::BufferArg>& params) {
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
