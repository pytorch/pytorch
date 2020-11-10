#ifdef TORCH_ENABLE_LLVM

#include <torch/csrc/jit/tensorexpr/llvm_jit.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/SymbolStringPool.h>
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Mangler.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <sleef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
namespace orc {

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
#if LLVM_VERSION_MAJOR >= 9 && LLVM_VERSION_MAJOR <= 12
class TORCH_API PytorchLLVMJITImpl {
 private:
  std::unique_ptr<LLJIT> LLJ;

 public:
  PytorchLLVMJITImpl() : LLJ(cantFail(LLJITBuilder().create())) {
    auto ProcSymbolsGenerator =
        cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
            LLJ->getDataLayout().getGlobalPrefix()));
    auto& JD = LLJ->getMainJITDylib();
#if LLVM_VERSION_MAJOR == 9
    JD.setGenerator(std::move(ProcSymbolsGenerator));
#else
    JD.addGenerator(std::move(ProcSymbolsGenerator));
#endif

    // Handle platform-specific symbol mangling
    MangleAndInterner Mangle(LLJ->getExecutionSession(), LLJ->getDataLayout());

    // Register implementations of intrinsics
    cantFail(JD.define(absoluteSymbols({
      {Mangle("log10f"),
       {llvm::pointerToJITTargetAddress(&log10f), JITSymbolFlags::None}},
          {Mangle("log1pf"),
           {llvm::pointerToJITTargetAddress(&log1pf), JITSymbolFlags::None}},
          {Mangle("logf"),
           {llvm::pointerToJITTargetAddress(&logf), JITSymbolFlags::None}},
          {Mangle("log2f"),
           {llvm::pointerToJITTargetAddress(&log2f), JITSymbolFlags::None}},
          {Mangle("expf"),
           {llvm::pointerToJITTargetAddress(&expf), JITSymbolFlags::None}},
          {Mangle("erff"),
           {llvm::pointerToJITTargetAddress(&erff), JITSymbolFlags::None}},
          {Mangle("cosf"),
           {llvm::pointerToJITTargetAddress(&cosf), JITSymbolFlags::None}},
          {Mangle("sinf"),
           {llvm::pointerToJITTargetAddress(&sinf), JITSymbolFlags::None}},
          {Mangle("tanf"),
           {llvm::pointerToJITTargetAddress(&tanf), JITSymbolFlags::None}},
          {Mangle("acosf"),
           {llvm::pointerToJITTargetAddress(&acosf), JITSymbolFlags::None}},
          {Mangle("asinf"),
           {llvm::pointerToJITTargetAddress(&asinf), JITSymbolFlags::None}},
          {Mangle("atanf"),
           {llvm::pointerToJITTargetAddress(&atanf), JITSymbolFlags::None}},
          {Mangle("coshf"),
           {llvm::pointerToJITTargetAddress(&coshf), JITSymbolFlags::None}},
          {Mangle("sinhf"),
           {llvm::pointerToJITTargetAddress(&sinhf), JITSymbolFlags::None}},
          {Mangle("tanhf"),
           {llvm::pointerToJITTargetAddress(&tanhf), JITSymbolFlags::None}},
          {Mangle("sqrtf"),
           {llvm::pointerToJITTargetAddress(&sqrtf), JITSymbolFlags::None}},
          {Mangle("fabsf"),
           {llvm::pointerToJITTargetAddress(&fabsf), JITSymbolFlags::None}},
          {Mangle("floorf"),
           {llvm::pointerToJITTargetAddress(&floorf), JITSymbolFlags::None}},
          {Mangle("ceilf"),
           {llvm::pointerToJITTargetAddress(&ceilf), JITSymbolFlags::None}},
          {Mangle("roundf"),
           {llvm::pointerToJITTargetAddress(&roundf), JITSymbolFlags::None}},
          {Mangle("truncf"),
           {llvm::pointerToJITTargetAddress(&truncf), JITSymbolFlags::None}},
          {Mangle("atan2f"),
           {llvm::pointerToJITTargetAddress(&atan2f), JITSymbolFlags::None}},
          {Mangle("fmodf"),
           {llvm::pointerToJITTargetAddress(&fmodf), JITSymbolFlags::None}},
          {Mangle("remainderf"),
           {llvm::pointerToJITTargetAddress(&remainderf),
            JITSymbolFlags::None}},

          // FP32 Sleef functions -- SSE
          {Mangle("Sleef_acosf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_acosf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_asinf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_asinf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atanf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_atanf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_cosf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_cosf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sinf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_sinf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tanf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_tanf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_coshf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_coshf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sinhf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_sinhf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tanhf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_tanhf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erff4"),
           {llvm::pointerToJITTargetAddress(&Sleef_erff4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erfcf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_erfcf4_u15),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_expf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expm1f4"),
           {llvm::pointerToJITTargetAddress(&Sleef_expm1f4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_logf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_logf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log2f4"),
           {llvm::pointerToJITTargetAddress(&Sleef_log2f4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log10f4"),
           {llvm::pointerToJITTargetAddress(&Sleef_log10f4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log1pf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_log1pf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sqrtf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_sqrtf4_u05),
            JITSymbolFlags::None}},
          {Mangle("Sleef_fabsf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_fabsf4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_floorf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_floorf4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_ceilf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_ceilf4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_truncf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_truncf4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_roundf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_roundf4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_lgammaf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_lgammaf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atan2f4"),
           {llvm::pointerToJITTargetAddress(&Sleef_atan2f4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_powf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_powf4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_fmodf4"),
           {llvm::pointerToJITTargetAddress(&Sleef_fmodf4),
            JITSymbolFlags::None}},

      // FP32 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
          {Mangle("Sleef_acosf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_acosf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_asinf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_asinf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atanf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_atanf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_cosf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_cosf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sinf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_sinf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tanf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_tanf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_coshf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_coshf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sinhf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_sinhf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tanhf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_tanhf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erff8"),
           {llvm::pointerToJITTargetAddress(&Sleef_erff8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erfcf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_erfcf8_u15),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_expf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expm1f8"),
           {llvm::pointerToJITTargetAddress(&Sleef_expm1f8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_logf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_logf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log2f8"),
           {llvm::pointerToJITTargetAddress(&Sleef_log2f8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log10f8"),
           {llvm::pointerToJITTargetAddress(&Sleef_log10f8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log1pf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_log1pf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sqrtf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_sqrtf8_u05),
            JITSymbolFlags::None}},
          {Mangle("Sleef_fabsf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_fabsf8),
            JITSymbolFlags::None}},
          {Mangle("Sleef_floorf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_floorf8),
            JITSymbolFlags::None}},
          {Mangle("Sleef_ceilf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_ceilf8),
            JITSymbolFlags::None}},
          {Mangle("Sleef_truncf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_truncf8),
            JITSymbolFlags::None}},
          {Mangle("Sleef_roundf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_roundf8),
            JITSymbolFlags::None}},
          {Mangle("Sleef_lgammaf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_lgammaf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atan2f8"),
           {llvm::pointerToJITTargetAddress(&Sleef_atan2f8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_powf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_powf8_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_fmodf8"),
           {llvm::pointerToJITTargetAddress(&Sleef_fmodf8),
            JITSymbolFlags::None}},
#endif

          // FP64 Sleef functions -- SSE
          {Mangle("Sleef_acosd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_acosd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_asind2"),
           {llvm::pointerToJITTargetAddress(&Sleef_asind2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atand2"),
           {llvm::pointerToJITTargetAddress(&Sleef_atand2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_cosd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_cosd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sind2"),
           {llvm::pointerToJITTargetAddress(&Sleef_sind2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tand2"),
           {llvm::pointerToJITTargetAddress(&Sleef_tand2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_coshd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_coshd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sinhd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_sinhd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tanhd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_tanhd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erfd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_erfd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erfcd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_erfcd2_u15),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_expd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expm1d2"),
           {llvm::pointerToJITTargetAddress(&Sleef_expm1d2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_logd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_logd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log2d2"),
           {llvm::pointerToJITTargetAddress(&Sleef_log2d2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log10d2"),
           {llvm::pointerToJITTargetAddress(&Sleef_log10d2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log1pd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_log1pd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sqrtd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_sqrtd2_u05),
            JITSymbolFlags::None}},
          {Mangle("Sleef_fabsd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_fabsd2),
            JITSymbolFlags::None}},
          {Mangle("Sleef_floord2"),
           {llvm::pointerToJITTargetAddress(&Sleef_floord2),
            JITSymbolFlags::None}},
          {Mangle("Sleef_ceild2"),
           {llvm::pointerToJITTargetAddress(&Sleef_ceild2),
            JITSymbolFlags::None}},
          {Mangle("Sleef_truncd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_truncd2),
            JITSymbolFlags::None}},
          {Mangle("Sleef_roundd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_roundd2),
            JITSymbolFlags::None}},
          {Mangle("Sleef_lgammad2"),
           {llvm::pointerToJITTargetAddress(&Sleef_lgammad2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atan2d2"),
           {llvm::pointerToJITTargetAddress(&Sleef_atan2d2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_powd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_powd2_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_fmodd2"),
           {llvm::pointerToJITTargetAddress(&Sleef_fmodd2),
            JITSymbolFlags::None}},

      // FP64 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
          {Mangle("Sleef_acosd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_acosd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_asind4"),
           {llvm::pointerToJITTargetAddress(&Sleef_asind4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atand4"),
           {llvm::pointerToJITTargetAddress(&Sleef_atand4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_cosd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_cosd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sind4"),
           {llvm::pointerToJITTargetAddress(&Sleef_sind4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tand4"),
           {llvm::pointerToJITTargetAddress(&Sleef_tand4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_coshd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_coshd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sinhd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_sinhd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_tanhd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_tanhd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erfd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_erfd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_erfcd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_erfcd4_u15),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_expd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_expm1d4"),
           {llvm::pointerToJITTargetAddress(&Sleef_expm1d4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_logd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_logd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log2d4"),
           {llvm::pointerToJITTargetAddress(&Sleef_log2d4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log10d4"),
           {llvm::pointerToJITTargetAddress(&Sleef_log10d4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_log1pd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_log1pd4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_sqrtd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_sqrtd4_u05),
            JITSymbolFlags::None}},
          {Mangle("Sleef_fabsd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_fabsd4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_floord4"),
           {llvm::pointerToJITTargetAddress(&Sleef_floord4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_ceild4"),
           {llvm::pointerToJITTargetAddress(&Sleef_ceild4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_truncd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_truncd4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_roundd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_roundd4),
            JITSymbolFlags::None}},
          {Mangle("Sleef_lgammad4"),
           {llvm::pointerToJITTargetAddress(&Sleef_lgammad4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_atan2d4"),
           {llvm::pointerToJITTargetAddress(&Sleef_atan2d4_u10),
            JITSymbolFlags::None}},
          {Mangle("Sleef_powd4"),
           {llvm::pointerToJITTargetAddress(&Sleef_powd4_u10),
            JITSymbolFlags::None}},
      {
        Mangle("Sleef_fmodd4"), {
          llvm::pointerToJITTargetAddress(&Sleef_fmodd4), JITSymbolFlags::None
        }
      }
#endif
    })));
  }

  Error addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C) {
    if (auto Err =
            LLJ->addIRModule(ThreadSafeModule(std::move(M), std::move(C)))) {
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

Error PytorchLLVMJIT::addModule(
    std::unique_ptr<Module> M,
    std::unique_ptr<LLVMContext> C) {
  return impl_->addModule(std::move(M), std::move(C));
}

JITSymbol PytorchLLVMJIT::findSymbol(const std::string Name) {
  return impl_->findSymbol(std::move(Name));
}

const DataLayout& PytorchLLVMJIT::getDataLayout() {
  return impl_->getDataLayout();
}

#elif LLVM_VERSION_MAJOR == 8 && LLVM_VERSION_PATCH == 20181009

class TORCH_API PytorchLLVMJITImpl {
 private:
  ExecutionSession ES;
  std::shared_ptr<SymbolResolver> Resolver;
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  RTDyldObjectLinkingLayer ObjectLayer;
  IRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;

 public:
  PytorchLLVMJITImpl()
      : Resolver(createLegacyLookupResolver(
            ES,
            [this](const std::string& Name) -> JITSymbol {
              if (auto Sym = CompileLayer.findSymbol(Name, false))
                return Sym;
              else if (auto Err = Sym.takeError())
                return std::move(Err);
              if (auto SymAddr =
                      RTDyldMemoryManager::getSymbolAddressInProcess(Name))
                return JITSymbol(SymAddr, JITSymbolFlags::Exported);
              return nullptr;
            },
            [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
        TM(EngineBuilder().selectTarget()),
        DL(TM->createDataLayout()),
        ObjectLayer(
            ES,
            [this](VModuleKey) {
              return RTDyldObjectLinkingLayer::Resources{
                  std::make_shared<SectionMemoryManager>(), Resolver};
            }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  TargetMachine& getTargetMachine() {
    return *TM;
  }

  VModuleKey addModule(std::unique_ptr<Module> M) {
    // Add the module to the JIT with a new VModuleKey.
    auto K = ES.allocateVModule();
    cantFail(CompileLayer.addModule(K, std::move(M)));
    return K;
  }

  JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompileLayer.findSymbol(MangledNameStream.str(), true);
  }

  JITTargetAddress getSymbolAddress(const std::string Name) {
    return cantFail(findSymbol(Name).getAddress());
  }

  void removeModule(VModuleKey K) {
    cantFail(CompileLayer.removeModule(K));
  }

  const DataLayout& getDataLayout() {
    return DL;
  }
};

PytorchLLVMJIT::PytorchLLVMJIT()
    : impl_(std::make_unique<PytorchLLVMJITImpl>()) {}

PytorchLLVMJIT::~PytorchLLVMJIT() = default;

Error PytorchLLVMJIT::addModule(
    std::unique_ptr<Module> M,
    std::unique_ptr<LLVMContext> C) {
  impl_->addModule(std::move(M));
  return Error::success();
}

JITSymbol PytorchLLVMJIT::findSymbol(const std::string Name) {
  return impl_->findSymbol(std::move(Name));
}

const DataLayout& PytorchLLVMJIT::getDataLayout() {
  return impl_->getDataLayout();
}

#else // LLVM_VERSION_MAJOR
#error Only LLVM versions 8 through 12 are supported.
#endif

} // end namespace orc
} // end namespace llvm

#endif // TORCH_ENABLE_LLVM
