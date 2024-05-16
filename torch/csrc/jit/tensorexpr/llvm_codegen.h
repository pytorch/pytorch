#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <c10/util/Optional.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

class LLVMCodeGenImpl;
class LLVMCodeGenCallee;

class TORCH_API LLVMCodeGen : public CodeGen {
 public:
  explicit LLVMCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func",
      Dtype dtype = kInt,
      std::optional<std::string> triple = c10::nullopt,
      std::optional<std::string> cpu = c10::nullopt,
      std::optional<std::string> attrs = c10::nullopt);
  explicit LLVMCodeGen(StmtPtr stmt);

  LLVMCodeGen() = delete;
  ~LLVMCodeGen() override;

  // Cleans up all the memory used during LLVM code generation pass except
  // the generated kernel. After calling this method, users should not call
  // methods like `getCodeText` that require the LLVMCodeGenImpl data. However,
  // users can continue to call this kernel using `call` and `call_raw`.
  void cleanup_memory();

  TORCH_API void call(const std::vector<CallArg>& args) override;
  TORCH_API void call_raw(const std::vector<void*>& args) override;
  TORCH_API void call_with_numel(void** args, int64_t numel) override;

  at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      std::optional<c10::ScalarType> dtype_opt,
      std::optional<c10::Layout> layout_opt,
      std::optional<c10::Device> device_opt,
      std::optional<bool> pin_memory_opt) override;

  template <typename T>
  T value() {
    return value<T>(nullptr);
  }

  template <typename T>
  T value(std::vector<void*>& args) {
    return value<T>(args.data());
  }

  template <typename T>
  T value(void** args) {
    T (*fp)(void**) = (T(*)(void**))getKernelAddress(callee_.get());
    T rv = fp(args);
    return rv;
  }

  std::string getCodeText(const std::string& attr = "") override;

 private:
  void* getKernelAddress(LLVMCodeGenCallee* callee);

  std::unique_ptr<LLVMCodeGenCallee> callee_;
  std::unique_ptr<LLVMCodeGenImpl> impl_;
};

struct TORCH_API LLVMCodeGenBuilder {
  using BufferArg = CodeGen::BufferArg;

  LLVMCodeGenBuilder(StmtPtr stmt, std::vector<BufferArg> args)
      : stmt_(stmt), args_(std::move(args)) {}

  LLVMCodeGenBuilder& device(at::Device device) {
    device_ = device;
    return *this;
  }

  LLVMCodeGenBuilder& kernelFuncName(std::string name) {
    kernelFuncName_ = std::move(name);
    return *this;
  }

  LLVMCodeGenBuilder& dtype(Dtype d) {
    dtype_ = d;
    return *this;
  }

  LLVMCodeGenBuilder& triple(std::string triple) {
    triple_ = std::move(triple);
    return *this;
  }

  LLVMCodeGenBuilder& cpu(std::string cpu) {
    cpu_ = std::move(cpu);
    return *this;
  }

  LLVMCodeGenBuilder& attrs(std::string attrs) {
    attrs_ = std::move(attrs);
    return *this;
  }

  std::unique_ptr<LLVMCodeGen> build() {
    return std::make_unique<LLVMCodeGen>(
        stmt_, args_, device_, kernelFuncName_, dtype_, triple_, cpu_, attrs_);
  }

 private:
  StmtPtr stmt_;
  std::vector<BufferArg> args_;
  at::Device device_ = at::kCPU;
  std::string kernelFuncName_ = "func";
  Dtype dtype_ = kInt;
  std::optional<std::string> triple_ = c10::nullopt;
  std::optional<std::string> cpu_ = c10::nullopt;
  std::optional<std::string> attrs_ = c10::nullopt;
};

TORCH_API std::optional<std::string>& LLVMTargetTriple();
TORCH_API std::optional<std::string>& LLVMTargetCPU();
TORCH_API std::optional<std::string>& LLVMTargetAttrs();
TORCH_API bool& LLVMAOTWorkflow();

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM
