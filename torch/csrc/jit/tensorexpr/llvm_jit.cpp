#ifdef ENABLE_LLVM

#include "torch/csrc/jit/tensorexpr/llvm_jit.h"

#include <sleef.h>
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

    // FP32 Sleef functions -- SSE
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_acosf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_acosf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_asinf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_asinf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atanf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_atanf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atan2f4"),
        {llvm::pointerToJITTargetAddress(&Sleef_atan2f4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_cosf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_cosf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sinf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_sinf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tanf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_tanf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_coshf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_coshf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sinhf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_sinhf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tanhf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_tanhf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erff4"),
        {llvm::pointerToJITTargetAddress(&Sleef_erff4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erfcf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_erfcf4_u15), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_expf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expm1f4"),
        {llvm::pointerToJITTargetAddress(&Sleef_expm1f4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_logf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_log2f4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log2f4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf10f4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log10f4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf1pf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log1pf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_lgammaf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_lgammaf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_powf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_powf4_u10), {}}));

    // FP32 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_acosf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_acosf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_asinf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_asinf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atanf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_atanf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atan2f8"),
        {llvm::pointerToJITTargetAddress(&Sleef_atan2f8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_cosf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_cosf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sinf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_sinf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tanf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_tanf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_coshf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_coshf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sinhf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_sinhf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tanhf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_tanhf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erff8"),
        {llvm::pointerToJITTargetAddress(&Sleef_erff8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erfcf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_erfcf8_u15), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_expf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expm1f8"),
        {llvm::pointerToJITTargetAddress(&Sleef_expm1f8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_logf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_log2f8"),
        {llvm::pointerToJITTargetAddress(&Sleef_log2f8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf10f8"),
        {llvm::pointerToJITTargetAddress(&Sleef_log10f8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf1pf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_log1pf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_lgammaf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_lgammaf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_powf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_powf8_u10), {}}));
#endif
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
