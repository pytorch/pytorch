#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>

namespace torch {
namespace jit {
namespace tensorexpr {

void DispatchParallel(int8_t* func, int start, int stop, int8_t* packed_data);

inline std::string formatError(llvm::Error&& err, const char* msg) {
  static constexpr char* defaultErrorMsg = "Unexpected failure in LLVM JIT";
  std::string errorMsg(msg ? msg : defaultErrorMsg);
  llvm::raw_string_ostream ss(errorMsg);
  ss << ": " << err;
  return ss.str();
}

template <typename T>
T assertSuccess(llvm::Expected<T> valOrErr, const char* msg = nullptr) {
  TORCH_INTERNAL_ASSERT(valOrErr, formatError(valOrErr.takeError(), msg));
  return std::move(*valOrErr);
}

inline void assertSuccess(llvm::Error err, const char* msg = nullptr) {
  TORCH_INTERNAL_ASSERT(!err, formatError(std::move(err), msg));
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch

namespace llvm {
namespace orc {

class PytorchLLVMJITImpl;

class TORCH_API PytorchLLVMJIT {
 public:
  PytorchLLVMJIT();
  ~PytorchLLVMJIT();

  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C);

  JITSymbol findSymbol(const std::string Name);

  bool hasSymbol(const std::string& Name);

  TargetMachine& getTargetMachine();

  const DataLayout& getDataLayout();

 private:
  // Use the PImpl idiom here to hide the no-rtti parts of the JIT structure.
  std::unique_ptr<PytorchLLVMJITImpl> impl_;
};

} // end namespace orc
} // end namespace llvm

#endif // ENABLE LLVM
