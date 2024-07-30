#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <utility>

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
  CodeGen(StmtPtr stmt, Ts... ts)
      : stmt_(std::move(stmt)), buffer_args_({BufferArg(ts)...}) {}

  CodeGen(
      StmtPtr stmt,
      std::vector<BufferArg> buffer_args,
      at::Device device = at::kCPU,
      std::string kernel_func_name = "func");

  virtual ~CodeGen() = default;

  StmtPtr stmt() const {
    return stmt_;
  }

  void set_stmt(StmtPtr s) {
    stmt_ = std::move(s);
  }

  void apply_mutator(IRMutator* mutator) {
    stmt_ = stmt_->accept_mutator(mutator);
  }

  void apply_visitor(IRVisitor* visitor) {
    stmt_->accept(visitor);
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
  virtual std::string getCodeText(
      const std::string& attr [[maybe_unused]] = "") {
    return "";
  }

  // TODO: Figure out how to unify these call interfaces.

  /// Call a function with a vector of CallArgs, which are tagged
  /// unions that properly type the arguments.
  virtual void call(const std::vector<CallArg>& args) = 0;

  /// Call a function faster than a regular `call` by assuming that
  /// the generated kernel already knows the type of the arguments, so
  /// they can be type-punned with `void*`s.
  virtual void call_raw(const std::vector<void*>& args) = 0;

  /// Call a function even faster than a regular call, by assuming
  /// that the number of thread blocks can be derived from `numel` via
  /// a simple division, rather than evaluating an expression.
  virtual void call_with_numel(void** args, int64_t numel);

  virtual at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      std::optional<c10::ScalarType> dtype_opt,
      std::optional<c10::Layout> layout_opt,
      std::optional<c10::Device> device_opt,
      std::optional<bool> pin_memory_opt) {
    return at::empty_strided(
        size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
  }

  const std::string& kernel_func_name() const {
    return kernel_func_name_;
  }

  void allocIntermediateBufs();

 protected:
  static void* argToPtr(const BufferArg& bufferArg, const CallArg& callArg);

 private:
  StmtPtr stmt_;
  std::vector<BufferArg> buffer_args_;
  at::Device device_ = at::kCPU;
  std::string kernel_func_name_ = "func";
};

class TORCH_API ExtCallMemoryReuse : public IRMutator {
  static std::unordered_map<std::string, std::string> makeExtCallFuncNameMap();
  static const std::unordered_map<std::string, std::string> extCallFuncNameMap_;

 public:
  explicit ExtCallMemoryReuse(
      const std::vector<CodeGen::BufferArg>& bufferArgs);
  ~ExtCallMemoryReuse() override = default;
  StmtPtr mutate(ExternalCallPtr v) override;

 private:
  std::unordered_set<BufPtr> bufferArgs_;
};

class CodeGen::BufferArg {
 public:
  BufferArg(const Tensor& tensor) : buf_(tensor.buf()) {}
  BufferArg(const VarHandle& var) : var_(var.node()), isVar_(true) {}
  BufferArg(const BufHandle& buf) : buf_(buf.node()) {}
  BufferArg(BufPtr buf) : buf_(std::move(buf)) {}

  VarPtr var() const {
    return isVar_ ? var_ : buf_->base_handle();
  }

  BufPtr buf() const {
    return buf_;
  }

  bool isVar() const {
    return isVar_;
  }

  Dtype dtype() const {
    return isVar_ ? var_->dtype() : buf_->dtype();
  }

 private:
  VarPtr var_ = nullptr;
  BufPtr buf_ = nullptr;
  bool isVar_ = false;
};

class CodeGen::CallArg {
 public:
  template <typename T>
  CallArg(const PaddedBuffer<T>& buffer);

  template <typename T>
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,cppcoreguidelines-pro-type-const-cast)
  CallArg(const std::vector<T>& buffer)
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      : data_(const_cast<T*>(buffer.data())) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  CallArg(void* ptr) : data_(ptr) {}

#define ARG_TYPE_CTOR(Type, Name)      \
  CallArg(Type v) {                    \
    memcpy(buffer_, &v, sizeof(Type)); \
    data_ = (void*)buffer_;            \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ARG_TYPE_CTOR);
#undef ARG_TYPE_CTOR

  void* data() const {
    return data_;
  }

  CallArg(const CallArg& rhs) {
    if (rhs.data_ == rhs.buffer_) {
      memcpy(this->buffer_, rhs.buffer_, sizeof(rhs.buffer_));
      this->data_ = (void*)(this->buffer_);
    } else {
      this->data_ = rhs.data_;
    }
  }

  CallArg& operator=(const CallArg& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (rhs.data_ == rhs.buffer_) {
      memcpy(this->buffer_, rhs.buffer_, sizeof(rhs.buffer_));
      this->data_ = (void*)(this->buffer_);
    } else {
      this->data_ = rhs.data_;
    }
    return *this;
  }

#define ARG_PTR_DEFINE(Type, Name)                  \
  Type* Name##Ptr() const {                         \
    TORCH_INTERNAL_ASSERT(data_ == (void*)buffer_); \
    return (Type*)data_;                            \
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, ARG_PTR_DEFINE);
#undef ARG_PTR_DEFINE

 private:
  void* data_;
  // Regarding a scalar value, CallArg uses void**=&data_ to store it. But the
  // bit width of a pointer is 32bit on a 32bit platform. It cannot store the
  // scalar if the bit width of the scalar is larger than 32bit, such as double
  // and long. Hence, we add 8 bytes buffer dedicated to storing the scalar
  // value regardless its bit width is less or greater than 32bits.
  char buffer_[8] = {0}; // 64bits
};

class RegisterCodeGenList {
 public:
  TORCH_API static RegisterCodeGenList& GetInstance() {
    static RegisterCodeGenList codegen_list;
    return codegen_list;
  }

  using StmtFactoryMethod = std::function<std::unique_ptr<CodeGen>(
      StmtPtr stmt,
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
        [](StmtPtr stmt,
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
    StmtPtr stmt,
    const std::vector<CodeGen::BufferArg>& params,
    at::Device device = at::kCPU,
    const std::string& kernel_func_name = "func");

class TORCH_API GenericIntrinsicsExpander : public IRMutator {
 protected:
  ExprPtr mutate(IntrinsicsPtr v) override;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
