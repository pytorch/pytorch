#ifdef ENABLE_LLVM

#include "torch/csrc/jit/tensorexpr/llvm_jit.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace llvm {
namespace orc {

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
class TORCH_API PytorchLLVMJITImpl {
 private:
  std::unique_ptr<LLJIT> LLJ;

 public:
  PytorchLLVMJITImpl() : LLJ(cantFail(LLJITBuilder().create())) {
    auto ProcSymbolsGenerator =
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
                LLJ->getDataLayout().getGlobalPrefix()));
    LLJ->getMainJITDylib().setGenerator(std::move(ProcSymbolsGenerator));

    // Handle platform-specific symbol mangling
    MangleAndInterner Mangle(LLJ->getExecutionSession(), LLJ->getDataLayout());

    // Register implementations of intrinsics
    cantFail(LLJ->defineAbsolute(
        *Mangle("log10f"), {llvm::pointerToJITTargetAddress(&log10f), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("logf"), {llvm::pointerToJITTargetAddress(&logf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("log2f"), {llvm::pointerToJITTargetAddress(&log2f), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("expf"), {llvm::pointerToJITTargetAddress(&expf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("erff"), {llvm::pointerToJITTargetAddress(&erff), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("cosf"), {llvm::pointerToJITTargetAddress(&cosf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("sinf"), {llvm::pointerToJITTargetAddress(&sinf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("tanf"), {llvm::pointerToJITTargetAddress(&tanf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("acosf"), {llvm::pointerToJITTargetAddress(&acosf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("asinf"), {llvm::pointerToJITTargetAddress(&asinf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("atanf"), {llvm::pointerToJITTargetAddress(&atanf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("coshf"), {llvm::pointerToJITTargetAddress(&coshf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("sinhf"), {llvm::pointerToJITTargetAddress(&sinhf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("tanhf"), {llvm::pointerToJITTargetAddress(&tanhf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("sqrtf"), {llvm::pointerToJITTargetAddress(&sqrtf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("fabsf"), {llvm::pointerToJITTargetAddress(&fabsf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("floorf"), {llvm::pointerToJITTargetAddress(&floorf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("ceilf"), {llvm::pointerToJITTargetAddress(&ceilf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("roundf"), {llvm::pointerToJITTargetAddress(&roundf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("truncf"), {llvm::pointerToJITTargetAddress(&truncf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("atan2f"), {llvm::pointerToJITTargetAddress(&atan2f), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("fmodf"), {llvm::pointerToJITTargetAddress(&fmodf), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("remainderf"),
        {llvm::pointerToJITTargetAddress(&remainderf), {}}));
  }

  Error addModule(ThreadSafeModule M) {
    if (auto Err = LLJ->addIRModule(std::move(M))) {
      return Err;
    }
    return Error::success();
  }

  JITSymbol findSymbol(const std::string Name) {
    return cantFail(LLJ->lookup(Name));
  }

  const DataLayout& getDataLayout() {
    return LLJ->getDataLayout();
  }
};

PytorchLLVMJIT::PytorchLLVMJIT()
    : impl_(std::make_unique<PytorchLLVMJITImpl>()) {}

PytorchLLVMJIT::~PytorchLLVMJIT() = default;

Error PytorchLLVMJIT::addModule(ThreadSafeModule M) {
  return impl_->addModule(std::move(M));
}

JITSymbol PytorchLLVMJIT::findSymbol(const std::string Name) {
  return impl_->findSymbol(std::move(Name));
}

const DataLayout& PytorchLLVMJIT::getDataLayout() {
  return impl_->getDataLayout();
}

} // end namespace orc
} // end namespace llvm

#endif // ENABLE_LLVM
