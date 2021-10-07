#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Target/TargetMachine.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace tensorexpr {

extern "C" {
void DispatchParallel(
    int8_t* func,
    int64_t start,
    int64_t stop,
    int8_t* packed_data) noexcept;
}

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
  PytorchLLVMJIT(
      c10::optional<std::string> triple,
      c10::optional<std::string> cpu,
      c10::optional<std::string> attrs);
  ~PytorchLLVMJIT();

  // While creating any function in the module that is being added to this JIT,
  // get a unique name by calling `getUniqueFunctionName()` method. That
  // ensures that there is no duplicate function names in this JIT.
  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C);

  JITSymbol findSymbol(const std::string Name);

  bool hasSymbol(const std::string& Name);

  // Returns a function name that is unique in this JIT (among the function
  // names tracked by calling this method).
  //
  // When getUniqueFunctionName is called with a name that has never been used
  // before, it returns the input name as is. When it is called with the same
  // name subsequently, it appends "_<num>" to the name to uniquify it.
  //
  // For example:
  //  * First call to getUniqueFunctionName("func") => returns "func"
  //  * Second call to getUniqueFunctionName("func") => returns "func_1"
  //  * Third call to getUniqueFunctionName("func") => returns "func_2"
  //
  // NOTE: This method does not keep track of all the functions that are added
  // to this JIT. It only keeps track of the function names that are uniquified
  // by calling this method directly.
  //
  // Recommendation: Call this method before adding any function to this JIT.
  std::string getUniqueFunctionName(const std::string& name);

  TargetMachine& getTargetMachine();

  const DataLayout& getDataLayout();

 private:
  // Use the PImpl idiom here to hide the no-rtti parts of the JIT structure.
  std::unique_ptr<PytorchLLVMJITImpl> impl_;

  std::mutex mutex_;
  std::unordered_map<std::string, int> existing_functions_;
};

class TORCH_API PytorchLLVMJITCache {
 public:
  static PytorchLLVMJIT* getPytorchLLVMJITInstance(
      c10::optional<std::string> triple,
      c10::optional<std::string> cpu,
      c10::optional<std::string> attrs);

 private:
  static std::unordered_map<std::string, std::unique_ptr<PytorchLLVMJIT>>
      jit_cache_;
  static std::mutex mutex_;

  static std::string getCacheKey(
      c10::optional<std::string> triple,
      c10::optional<std::string> cpu,
      c10::optional<std::string> attrs);
};

} // end namespace orc
} // end namespace llvm

#endif // ENABLE LLVM
