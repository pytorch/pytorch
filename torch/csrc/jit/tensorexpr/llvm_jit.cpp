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
        *Mangle("Sleef_log10f4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log10f4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf1pf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log1pf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sqrtf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_sqrtf4_u05), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fabsf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_fabsf4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_floorf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_floorf4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_ceilf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_ceilf4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_truncf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_truncf4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_roundf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_roundf4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_lgammaf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_lgammaf4_u10), {}}));

    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atan2f4"),
        {llvm::pointerToJITTargetAddress(&Sleef_atan2f4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_powf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_powf4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fmodf4"),
        {llvm::pointerToJITTargetAddress(&Sleef_fmodf4), {}}));

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
        *Mangle("Sleef_log10f8"),
        {llvm::pointerToJITTargetAddress(&Sleef_log10f8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf1pf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_log1pf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sqrtf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_sqrtf8_u05), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fabsf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_fabsf8), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_floorf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_floorf8), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_ceilf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_ceilf8), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_truncf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_truncf8), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_roundf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_roundf8), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_lgammaf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_lgammaf8_u10), {}}));

    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atan2f8"),
        {llvm::pointerToJITTargetAddress(&Sleef_atan2f8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_powf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_powf8_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fmodf8"),
        {llvm::pointerToJITTargetAddress(&Sleef_fmodf8), {}}));
#endif

    // FP64 Sleef functions -- SSE
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_acosd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_acosd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_asind2"),
        {llvm::pointerToJITTargetAddress(&Sleef_asind2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atand2"),
        {llvm::pointerToJITTargetAddress(&Sleef_atand2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_cosd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_cosd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sind2"),
        {llvm::pointerToJITTargetAddress(&Sleef_sind2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tand2"),
        {llvm::pointerToJITTargetAddress(&Sleef_tand2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_coshd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_coshd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sinhd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_sinhd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tanhd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_tanhd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erfd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_erfd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erfcd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_erfcd2_u15), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_expd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expm1d2"),
        {llvm::pointerToJITTargetAddress(&Sleef_expm1d2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_logd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_log2d2"),
        {llvm::pointerToJITTargetAddress(&Sleef_log2d2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_log10d2"),
        {llvm::pointerToJITTargetAddress(&Sleef_log10d2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf1pd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_log1pd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sqrtd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_sqrtd2_u05), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fabsd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_fabsd2), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_floord2"),
        {llvm::pointerToJITTargetAddress(&Sleef_floord2), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_ceild2"),
        {llvm::pointerToJITTargetAddress(&Sleef_ceild2), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_truncd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_truncd2), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_roundd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_roundd2), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_lgammad2"),
        {llvm::pointerToJITTargetAddress(&Sleef_lgammad2_u10), {}}));

    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atan2d2"),
        {llvm::pointerToJITTargetAddress(&Sleef_atan2d2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_powd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_powd2_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fmodd2"),
        {llvm::pointerToJITTargetAddress(&Sleef_fmodd2), {}}));

    // FP64 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_acosd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_acosd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_asind4"),
        {llvm::pointerToJITTargetAddress(&Sleef_asind4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atand4"),
        {llvm::pointerToJITTargetAddress(&Sleef_atand4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_cosd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_cosd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sind4"),
        {llvm::pointerToJITTargetAddress(&Sleef_sind4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tand4"),
        {llvm::pointerToJITTargetAddress(&Sleef_tand4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_coshd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_coshd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sinhd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_sinhd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_tanhd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_tanhd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erfd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_erfd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_erfcd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_erfcd4_u15), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_expd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_expm1d4"),
        {llvm::pointerToJITTargetAddress(&Sleef_expm1d4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_logd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_log2d4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log2d4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_log10d4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log10d4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_logf1pd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_log1pd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_sqrtd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_sqrtd4_u05), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fabsd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_fabsd4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_floord4"),
        {llvm::pointerToJITTargetAddress(&Sleef_floord4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_ceild4"),
        {llvm::pointerToJITTargetAddress(&Sleef_ceild4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_truncd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_truncd4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_roundd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_roundd4), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_lgammad4"),
        {llvm::pointerToJITTargetAddress(&Sleef_lgammad4_u10), {}}));

    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_atan2d4"),
        {llvm::pointerToJITTargetAddress(&Sleef_atan2d4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_powd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_powd4_u10), {}}));
    cantFail(LLJ->defineAbsolute(
        *Mangle("Sleef_fmodd4"),
        {llvm::pointerToJITTargetAddress(&Sleef_fmodd4), {}}));
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
