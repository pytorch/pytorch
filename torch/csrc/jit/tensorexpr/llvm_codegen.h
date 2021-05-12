#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/execution_counter.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

DECLARE_TRIGGER(llvm_codegen_parallel_dispatched);

class LLVMCodeGenImpl;

class TORCH_API LLVMCodeGen : public CodeGen {
 public:
  explicit LLVMCodeGen(
      Stmt* stmt,
      const std::vector<BufferArg>& args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func",
      Dtype dtype = kInt);
  explicit LLVMCodeGen(Stmt* stmt);

  LLVMCodeGen() = delete;
  ~LLVMCodeGen() override;

  TORCH_API void call(const std::vector<CallArg>& args) override;
  TORCH_API void call_raw(const std::vector<void*>& args) override;

  at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      c10::optional<c10::ScalarType> dtype_opt,
      c10::optional<c10::Layout> layout_opt,
      c10::optional<c10::Device> device_opt,
      c10::optional<bool> pin_memory_opt) override;

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
    T (*fp)(void**) = (T(*)(void**))getKernelAddress(impl_.get());
    T rv = fp(args);
    return rv;
  }

  std::string getCodeText(const std::string& attr = "") override;

 private:
  void* getKernelAddress(LLVMCodeGenImpl* impl);

  std::unique_ptr<LLVMCodeGenImpl> impl_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM
