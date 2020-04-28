#ifdef TORCH_ENABLE_LLVM

#include <torch/csrc/jit/tensorexpr/llvm_jit.h>

#include <sleef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"

namespace llvm {
namespace orc {

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
class TORCH_API PytorchLLVMJITImpl {
 private:
  ExecutionEngine* EE_;
  TargetMachine* TM_;

 public:
  PytorchLLVMJITImpl(TargetMachine* TM) : TM_(TM) {}

  void addModule(std::unique_ptr<Module> M) {
    EE_ = EngineBuilder(std::move(M)).create(TM_);

    // Register implementations of intrinsics
    EE_->addGlobalMapping("log10f", (uint64_t)&log10f);
    EE_->addGlobalMapping("logf", (uint64_t)&logf);
    EE_->addGlobalMapping("log2f", (uint64_t)&log2f);
    EE_->addGlobalMapping("expf", (uint64_t)&expf);
    EE_->addGlobalMapping("erff", (uint64_t)&erff);
    EE_->addGlobalMapping("cosf", (uint64_t)&cosf);
    EE_->addGlobalMapping("sinf", (uint64_t)&sinf);
    EE_->addGlobalMapping("tanf", (uint64_t)&tanf);
    EE_->addGlobalMapping("acosf", (uint64_t)&acosf);
    EE_->addGlobalMapping("asinf", (uint64_t)&asinf);
    EE_->addGlobalMapping("atanf", (uint64_t)&atanf);
    EE_->addGlobalMapping("coshf", (uint64_t)&coshf);
    EE_->addGlobalMapping("sinhf", (uint64_t)&sinhf);
    EE_->addGlobalMapping("tanhf", (uint64_t)&tanhf);
    EE_->addGlobalMapping("sqrtf", (uint64_t)&sqrtf);
    EE_->addGlobalMapping("fabsf", (uint64_t)&fabsf);
    EE_->addGlobalMapping("floorf", (uint64_t)&floorf);
    EE_->addGlobalMapping("ceilf", (uint64_t)&ceilf);
    EE_->addGlobalMapping("roundf", (uint64_t)&roundf);
    EE_->addGlobalMapping("truncf", (uint64_t)&truncf);
    EE_->addGlobalMapping("atan2f", (uint64_t)&atan2f);
    EE_->addGlobalMapping("fmodf", (uint64_t)&fmodf);
    EE_->addGlobalMapping("remainderf", (uint64_t)&remainderf);

    // FP32 Sleef functions -- SSE
    EE_->addGlobalMapping("Sleef_acosf4", (uint64_t)&Sleef_acosf4_u10);
    EE_->addGlobalMapping("Sleef_asinf4", (uint64_t)&Sleef_asinf4_u10);
    EE_->addGlobalMapping("Sleef_atanf4", (uint64_t)&Sleef_atanf4_u10);
    EE_->addGlobalMapping("Sleef_cosf4", (uint64_t)&Sleef_cosf4_u10);
    EE_->addGlobalMapping("Sleef_sinf4", (uint64_t)&Sleef_sinf4_u10);
    EE_->addGlobalMapping("Sleef_tanf4", (uint64_t)&Sleef_tanf4_u10);
    EE_->addGlobalMapping("Sleef_coshf4", (uint64_t)&Sleef_coshf4_u10);
    EE_->addGlobalMapping("Sleef_sinhf4", (uint64_t)&Sleef_sinhf4_u10);
    EE_->addGlobalMapping("Sleef_tanhf4", (uint64_t)&Sleef_tanhf4_u10);
    EE_->addGlobalMapping("Sleef_erff4", (uint64_t)&Sleef_erff4_u10);
    EE_->addGlobalMapping("Sleef_erfcf4", (uint64_t)&Sleef_erfcf4_u15);
    EE_->addGlobalMapping("Sleef_expf4", (uint64_t)&Sleef_expf4_u10);
    EE_->addGlobalMapping("Sleef_expm1f4", (uint64_t)&Sleef_expm1f4_u10);
    EE_->addGlobalMapping("Sleef_logf4", (uint64_t)&Sleef_logf4_u10);
    EE_->addGlobalMapping("Sleef_log2f4", (uint64_t)&Sleef_log2f4_u10);
    EE_->addGlobalMapping("Sleef_log10f4", (uint64_t)&Sleef_log10f4_u10);
    EE_->addGlobalMapping("Sleef_logf1pf4", (uint64_t)&Sleef_log1pf4_u10);
    EE_->addGlobalMapping("Sleef_sqrtf4", (uint64_t)&Sleef_sqrtf4_u05);
    EE_->addGlobalMapping("Sleef_fabsf4", (uint64_t)&Sleef_fabsf4);
    EE_->addGlobalMapping("Sleef_floorf4", (uint64_t)&Sleef_floorf4);
    EE_->addGlobalMapping("Sleef_ceilf4", (uint64_t)&Sleef_ceilf4);
    EE_->addGlobalMapping("Sleef_truncf4", (uint64_t)&Sleef_truncf4);
    EE_->addGlobalMapping("Sleef_roundf4", (uint64_t)&Sleef_roundf4);
    EE_->addGlobalMapping("Sleef_lgammaf4", (uint64_t)&Sleef_lgammaf4_u10);

    EE_->addGlobalMapping("Sleef_atan2f4", (uint64_t)&Sleef_atan2f4_u10);
    EE_->addGlobalMapping("Sleef_powf4", (uint64_t)&Sleef_powf4_u10);
    EE_->addGlobalMapping("Sleef_fmodf4", (uint64_t)&Sleef_fmodf4);

    // FP32 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
    EE_->addGlobalMapping("Sleef_acosf8", (uint64_t)&Sleef_acosf8_u10);
    EE_->addGlobalMapping("Sleef_asinf8", (uint64_t)&Sleef_asinf8_u10);
    EE_->addGlobalMapping("Sleef_atanf8", (uint64_t)&Sleef_atanf8_u10);
    EE_->addGlobalMapping("Sleef_cosf8", (uint64_t)&Sleef_cosf8_u10);
    EE_->addGlobalMapping("Sleef_sinf8", (uint64_t)&Sleef_sinf8_u10);
    EE_->addGlobalMapping("Sleef_tanf8", (uint64_t)&Sleef_tanf8_u10);
    EE_->addGlobalMapping("Sleef_coshf8", (uint64_t)&Sleef_coshf8_u10);
    EE_->addGlobalMapping("Sleef_sinhf8", (uint64_t)&Sleef_sinhf8_u10);
    EE_->addGlobalMapping("Sleef_tanhf8", (uint64_t)&Sleef_tanhf8_u10);
    EE_->addGlobalMapping("Sleef_erff8", (uint64_t)&Sleef_erff8_u10);
    EE_->addGlobalMapping("Sleef_erfcf8", (uint64_t)&Sleef_erfcf8_u15);
    EE_->addGlobalMapping("Sleef_expf8", (uint64_t)&Sleef_expf8_u10);
    EE_->addGlobalMapping("Sleef_expm1f8", (uint64_t)&Sleef_expm1f8_u10);
    EE_->addGlobalMapping("Sleef_logf8", (uint64_t)&Sleef_logf8_u10);
    EE_->addGlobalMapping("Sleef_log2f8", (uint64_t)&Sleef_log2f8_u10);
    EE_->addGlobalMapping("Sleef_log10f8", (uint64_t)&Sleef_log10f8_u10);
    EE_->addGlobalMapping("Sleef_logf1pf8", (uint64_t)&Sleef_log1pf8_u10);
    EE_->addGlobalMapping("Sleef_sqrtf8", (uint64_t)&Sleef_sqrtf8_u05);
    EE_->addGlobalMapping("Sleef_fabsf8", (uint64_t)&Sleef_fabsf8);
    EE_->addGlobalMapping("Sleef_floorf8", (uint64_t)&Sleef_floorf8);
    EE_->addGlobalMapping("Sleef_ceilf8", (uint64_t)&Sleef_ceilf8);
    EE_->addGlobalMapping("Sleef_truncf8", (uint64_t)&Sleef_truncf8);
    EE_->addGlobalMapping("Sleef_roundf8", (uint64_t)&Sleef_roundf8);
    EE_->addGlobalMapping("Sleef_lgammaf8", (uint64_t)&Sleef_lgammaf8_u10);

    EE_->addGlobalMapping("Sleef_atan2f8", (uint64_t)&Sleef_atan2f8_u10);
    EE_->addGlobalMapping("Sleef_powf8", (uint64_t)&Sleef_powf8_u10);
    EE_->addGlobalMapping("Sleef_fmodf8", (uint64_t)&Sleef_fmodf8);
#endif

    // FP64 Sleef functions -- SSE
    EE_->addGlobalMapping("Sleef_acosd2", (uint64_t)&Sleef_acosd2_u10);
    EE_->addGlobalMapping("Sleef_asind2", (uint64_t)&Sleef_asind2_u10);
    EE_->addGlobalMapping("Sleef_atand2", (uint64_t)&Sleef_atand2_u10);
    EE_->addGlobalMapping("Sleef_cosd2", (uint64_t)&Sleef_cosd2_u10);
    EE_->addGlobalMapping("Sleef_sind2", (uint64_t)&Sleef_sind2_u10);
    EE_->addGlobalMapping("Sleef_tand2", (uint64_t)&Sleef_tand2_u10);
    EE_->addGlobalMapping("Sleef_coshd2", (uint64_t)&Sleef_coshd2_u10);
    EE_->addGlobalMapping("Sleef_sinhd2", (uint64_t)&Sleef_sinhd2_u10);
    EE_->addGlobalMapping("Sleef_tanhd2", (uint64_t)&Sleef_tanhd2_u10);
    EE_->addGlobalMapping("Sleef_erfd2", (uint64_t)&Sleef_erfd2_u10);
    EE_->addGlobalMapping("Sleef_erfcd2", (uint64_t)&Sleef_erfcd2_u15);
    EE_->addGlobalMapping("Sleef_expd2", (uint64_t)&Sleef_expd2_u10);
    EE_->addGlobalMapping("Sleef_expm1d2", (uint64_t)&Sleef_expm1d2_u10);
    EE_->addGlobalMapping("Sleef_logd2", (uint64_t)&Sleef_logd2_u10);
    EE_->addGlobalMapping("Sleef_log2d2", (uint64_t)&Sleef_log2d2_u10);
    EE_->addGlobalMapping("Sleef_log10d2", (uint64_t)&Sleef_log10d2_u10);
    EE_->addGlobalMapping("Sleef_logf1pd2", (uint64_t)&Sleef_log1pd2_u10);
    EE_->addGlobalMapping("Sleef_sqrtd2", (uint64_t)&Sleef_sqrtd2_u05);
    EE_->addGlobalMapping("Sleef_fabsd2", (uint64_t)&Sleef_fabsd2);
    EE_->addGlobalMapping("Sleef_floord2", (uint64_t)&Sleef_floord2);
    EE_->addGlobalMapping("Sleef_ceild2", (uint64_t)&Sleef_ceild2);
    EE_->addGlobalMapping("Sleef_truncd2", (uint64_t)&Sleef_truncd2);
    EE_->addGlobalMapping("Sleef_roundd2", (uint64_t)&Sleef_roundd2);
    EE_->addGlobalMapping("Sleef_lgammad2", (uint64_t)&Sleef_lgammad2_u10);

    EE_->addGlobalMapping("Sleef_atan2d2", (uint64_t)&Sleef_atan2d2_u10);
    EE_->addGlobalMapping("Sleef_powd2", (uint64_t)&Sleef_powd2_u10);
    EE_->addGlobalMapping("Sleef_fmodd2", (uint64_t)&Sleef_fmodd2);

    // FP64 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
    EE_->addGlobalMapping("Sleef_acosd4", (uint64_t)&Sleef_acosd4_u10);
    EE_->addGlobalMapping("Sleef_asind4", (uint64_t)&Sleef_asind4_u10);
    EE_->addGlobalMapping("Sleef_atand4", (uint64_t)&Sleef_atand4_u10);
    EE_->addGlobalMapping("Sleef_cosd4", (uint64_t)&Sleef_cosd4_u10);
    EE_->addGlobalMapping("Sleef_sind4", (uint64_t)&Sleef_sind4_u10);
    EE_->addGlobalMapping("Sleef_tand4", (uint64_t)&Sleef_tand4_u10);
    EE_->addGlobalMapping("Sleef_coshd4", (uint64_t)&Sleef_coshd4_u10);
    EE_->addGlobalMapping("Sleef_sinhd4", (uint64_t)&Sleef_sinhd4_u10);
    EE_->addGlobalMapping("Sleef_tanhd4", (uint64_t)&Sleef_tanhd4_u10);
    EE_->addGlobalMapping("Sleef_erfd4", (uint64_t)&Sleef_erfd4_u10);
    EE_->addGlobalMapping("Sleef_erfcd4", (uint64_t)&Sleef_erfcd4_u15);
    EE_->addGlobalMapping("Sleef_expd4", (uint64_t)&Sleef_expd4_u10);
    EE_->addGlobalMapping("Sleef_expm1d4", (uint64_t)&Sleef_expm1d4_u10);
    EE_->addGlobalMapping("Sleef_logd4", (uint64_t)&Sleef_logd4_u10);
    EE_->addGlobalMapping("Sleef_log2d4", (uint64_t)&Sleef_log2d4_u10);
    EE_->addGlobalMapping("Sleef_log10d4", (uint64_t)&Sleef_log10d4_u10);
    EE_->addGlobalMapping("Sleef_logf1pd4", (uint64_t)&Sleef_log1pd4_u10);
    EE_->addGlobalMapping("Sleef_sqrtd4", (uint64_t)&Sleef_sqrtd4_u05);
    EE_->addGlobalMapping("Sleef_fabsd4", (uint64_t)&Sleef_fabsd4);
    EE_->addGlobalMapping("Sleef_floord4", (uint64_t)&Sleef_floord4);
    EE_->addGlobalMapping("Sleef_ceild4", (uint64_t)&Sleef_ceild4);
    EE_->addGlobalMapping("Sleef_truncd4", (uint64_t)&Sleef_truncd4);
    EE_->addGlobalMapping("Sleef_roundd4", (uint64_t)&Sleef_roundd4);
    EE_->addGlobalMapping("Sleef_lgammad4", (uint64_t)&Sleef_lgammad4_u10);

    EE_->addGlobalMapping("Sleef_atan2d4", (uint64_t)&Sleef_atan2d4_u10);
    EE_->addGlobalMapping("Sleef_powd4", (uint64_t)&Sleef_powd4_u10);
    EE_->addGlobalMapping("Sleef_fmodd4", (uint64_t)&Sleef_fmodd4);
#endif

    EE_->finalizeObject();
  }

  void* findSymbol(const std::string& Name) {
    if (Function* F = EE_->FindFunctionNamed(Name.c_str())) {
      return EE_->getPointerToFunction(F);
    }

    return nullptr;
  }
};

PytorchLLVMJIT::PytorchLLVMJIT(TargetMachine* TM)
    : impl_(std::make_unique<PytorchLLVMJITImpl>(TM)) {}

PytorchLLVMJIT::~PytorchLLVMJIT() = default;

void PytorchLLVMJIT::addModule(std::unique_ptr<Module> M) {
  impl_->addModule(std::move(M));
}

void* PytorchLLVMJIT::findSymbol(const std::string& Name) {
  return impl_->findSymbol(std::move(Name));
}

} // end namespace orc
} // end namespace llvm

#endif // TORCH_ENABLE_LLVM
