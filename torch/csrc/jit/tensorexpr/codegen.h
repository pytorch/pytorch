#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

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

  CodeGen(
      Stmt* stmt,
      std::vector<BufferArg> buffer_args,
      at::Device device = at::kCPU,
      std::string kernel_func_name = "func")
      : stmt_(stmt),
        buffer_args_(std::move(buffer_args)),
        device_(device),
        kernel_func_name_(std::move(kernel_func_name)) {}

  virtual ~CodeGen() = default;

  Stmt* stmt() const {
    return stmt_;
  }

  void set_stmt(Stmt* s) {
    stmt_ = s;
  }

  void apply_mutator(IRMutator* mutator) {
    stmt_ = stmt_->accept_mutator(mutator);
  }

  std::vector<BufferArg>& buffer_args() {
    return buffer_args_;
  }

  const std::vector<BufferArg>& buffer_args() const {
    return buffer_args_;
  }

  at::Device device() {
    return device_;
  }

  // This function returns the generated code as
  // a string.
  virtual std::string getCodeText(const std::string& attr = "") {
    return ("");
  }

  // There are two ways to invoke the codegen:
  //  1) with a vector of CallArgs
  //  2) with a vector of raw 'void*' pointers
  //
  // The codegen knows types of all inputs from the buffer args, that's why
  // 'void*' pointers suffice.
  //
  // TODO: Eventually we might consider killing the CallArgs version, but
  // currently only LLVM codegen implements call_raw.
  virtual void call(const std::vector<CallArg>& args) = 0;
  virtual void call_raw(const std::vector<void*>& args) = 0;

  virtual at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      c10::optional<c10::ScalarType> dtype_opt,
      c10::optional<c10::Layout> layout_opt,
      c10::optional<c10::Device> device_opt,
      c10::optional<bool> pin_memory_opt) {
    return at::empty_strided(
        size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  }

  const std::string& kernel_func_name() const {
    return kernel_func_name_;
  }

 protected:
  static void* argToPtr(const BufferArg& bufferArg, const CallArg& callArg);

 private:
  Stmt* stmt_;
  std::vector<BufferArg> buffer_args_;
  at::Device device_ = at::kCPU;
  std::string kernel_func_name_ = "func";
};

class CodeGen::BufferArg {
 public:
  BufferArg(const Placeholder& buffer) : buf_(buffer.data()) {}
  BufferArg(Tensor* tensor) : buf_(tensor->buf()) {}
  BufferArg(const VarHandle& var) : var_(var.node()), isVar_(true) {}
  BufferArg(const BufHandle& buf) : buf_(buf.node()) {}

  const Var* var() const {
    return isVar_ ? var_ : buf_->base_handle();
  }

  const Buf* buf() const {
    return buf_;
  }

  bool isVar() const {
    return isVar_;
  }

  Dtype dtype() const {
    return isVar_ ? var_->dtype() : buf_->dtype();
  }

 private:
  const Var* var_ = nullptr;
  const Buf* buf_ = nullptr;
  bool isVar_ = false;
};

class CodeGen::CallArg {
 public:
  template <typename T>
  CallArg(const PaddedBuffer<T>& buffer);

  template <typename T>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-pro-type-const-cast)
  CallArg(const std::vector<T>& buffer)
      : data_(const_cast<T*>(buffer.data())) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CallArg(void* ptr) : data_(ptr) {}

#define ARG_TYPE_CTOR(Type, Name)     \
  CallArg(Type v) {                   \
    memcpy(&data_, &v, sizeof(Type)); \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, ARG_TYPE_CTOR);
#undef ARG_TYPE_CTOR

  void* data() const {
    return data_;
  }

#define ARG_PTR_DEFINE(Type, Name) \
  Type* Name##Ptr() const {        \
    return (Type*)&data_;          \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, ARG_PTR_DEFINE);
#undef ARG_PTR_DEFINE

 private:
  void* data_;
};

class RegisterCodeGenList {
 public:
  TORCH_API static RegisterCodeGenList& GetInstance() {
    static RegisterCodeGenList codegen_list;
    return codegen_list;
  }

  using StmtFactoryMethod = std::function<std::unique_ptr<CodeGen>(
      Stmt* stmt,
      const std::vector<CodeGen::BufferArg>&,
      at::Device device,
      const std::string& kernel_func_name)>;

  TORCH_API StmtFactoryMethod FindStmtFactoryMethod(const std::string& name);
  RegisterCodeGenList(const RegisterCodeGenList&) = delete;
  RegisterCodeGenList& operator=(const RegisterCodeGenList&) = delete;

 private:
  template <class CodeGenType>
  friend class RegisterCodeGen;
  RegisterCodeGenList() = default;
  TORCH_API void AddStmtFactoryMethod(
      const std::string& name,
      const StmtFactoryMethod& stmt_factory_method);

  std::unordered_map<std::string, StmtFactoryMethod> stmt_factory_methods_;
};

template <class CodeGenType>
class RegisterCodeGen {
 public:
  explicit RegisterCodeGen(const std::string& name) {
    RegisterCodeGenList& codegen_list = RegisterCodeGenList::GetInstance();
    codegen_list.AddStmtFactoryMethod(
        name,
        [](Stmt* stmt,
           const std::vector<CodeGen::BufferArg>& params,
           at::Device device,
           const std::string& kernel_func_name) {
          std::unique_ptr<CodeGen> method(
              new CodeGenType(stmt, params, device, kernel_func_name));
          return method;
        });
  }
};

TORCH_API std::unique_ptr<CodeGen> CreateCodeGen(
    const std::string& name,
    Stmt* stmt,
    const std::vector<CodeGen::BufferArg>& params,
    at::Device device = at::kCPU,
    const std::string& kernel_func_name = "func");

class TORCH_API GenericIntrinsicsExpander : public IRMutator {
 protected:
  const Expr* mutate(const Intrinsics* v) override;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
