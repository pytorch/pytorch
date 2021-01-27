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
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <c10/util/Half.h>

#include <sleef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace torch::jit::tensorexpr;

template <typename T>
static llvm::JITTargetAddress toAddress(T* Ptr) {
  return static_cast<llvm::JITTargetAddress>(reinterpret_cast<uintptr_t>(Ptr));
}

static llvm::orc::JITTargetMachineBuilder makeTargetMachineBuilder() {
#if 0
  // FIXME: Switch to using detectHost() rather than setting up the JTMB manually
  // once LLVM 10 is available.
  return assertSuccess(llvm::orc::JITTargetMachineBuilder::detectHost());
#else
  llvm::orc::JITTargetMachineBuilder JTMB(
      (llvm::Triple(llvm::sys::getProcessTriple())));

  // Retrieve host CPU name and sub-target features and add them to builder.
  // Relocation model, code model and codegen opt level are kept to default
  // values.
  llvm::SubtargetFeatures SubtargetFeatures;
  llvm::StringMap<bool> FeatureMap;
  llvm::sys::getHostCPUFeatures(FeatureMap);
  for (auto& Feature : FeatureMap) {
    SubtargetFeatures.AddFeature(Feature.first(), Feature.second);
  }

  JTMB.setCodeGenOptLevel(llvm::CodeGenOpt::Default);
  JTMB.setCPU(llvm::sys::getHostCPUName().str());
  JTMB.addFeatures(SubtargetFeatures.getFeatures());
  JTMB.getOptions().AllowFPOpFusion = llvm::FPOpFusion::Fast;

  return JTMB;
#endif
}

static void registerIntrinsics(
    llvm::orc::JITDylib& JD,
    llvm::orc::MangleAndInterner& Mangle) {
  using namespace llvm;
  using namespace llvm::orc;

  auto entry = [&](const char* name, auto ptr) -> SymbolMap::value_type {
    return {Mangle(name), {toAddress(ptr), JITSymbolFlags::None}};
  };

  assertSuccess(JD.define(absoluteSymbols({
    entry("log10f", &log10f), entry("log1pf", &log1pf), entry("logf", &logf),
        entry("log2f", &log2f), entry("expf", &expf), entry("erff", &erff),
        entry("cosf", &cosf), entry("sinf", &sinf), entry("tanf", &tanf),
        entry("acosf", &acosf), entry("asinf", &asinf), entry("atanf", &atanf),
        entry("coshf", &coshf), entry("sinhf", &sinhf), entry("tanhf", &tanhf),
        entry("sqrtf", &sqrtf), entry("fabsf", &fabsf),
        entry("floorf", &floorf), entry("ceilf", &ceilf),
        entry("roundf", &roundf), entry("truncf", &truncf),
        entry("atan2f", &atan2f), entry("fmodf", &fmodf),
        entry("remainderf", &remainderf),

        // float -> half & half -> float conversions
        entry("__gnu_h2f_ieee", &c10::detail::fp16_ieee_to_fp32_value),
        entry("__gnu_f2h_ieee", &c10::detail::fp16_ieee_from_fp32_value),

        // FP32 Sleef functions -- SSE
        entry("Sleef_acosf4", &Sleef_acosf4_u10),
        entry("Sleef_asinf4", &Sleef_asinf4_u10),
        entry("Sleef_atanf4", &Sleef_atanf4_u10),
        entry("Sleef_cosf4", &Sleef_cosf4_u10),
        entry("Sleef_sinf4", &Sleef_sinf4_u10),
        entry("Sleef_tanf4", &Sleef_tanf4_u10),
        entry("Sleef_coshf4", &Sleef_coshf4_u10),
        entry("Sleef_sinhf4", &Sleef_sinhf4_u10),
        entry("Sleef_tanhf4", &Sleef_tanhf4_u10),
        entry("Sleef_erff4", &Sleef_erff4_u10),
        entry("Sleef_erfcf4", &Sleef_erfcf4_u15),
        entry("Sleef_expf4", &Sleef_expf4_u10),
        entry("Sleef_expm1f4", &Sleef_expm1f4_u10),
        entry("Sleef_logf4", &Sleef_logf4_u10),
        entry("Sleef_log2f4", &Sleef_log2f4_u10),
        entry("Sleef_log10f4", &Sleef_log10f4_u10),
        entry("Sleef_log1pf4", &Sleef_log1pf4_u10),
        entry("Sleef_sqrtf4", &Sleef_sqrtf4_u05),
        entry("Sleef_fabsf4", &Sleef_fabsf4),
        entry("Sleef_floorf4", &Sleef_floorf4),
        entry("Sleef_ceilf4", &Sleef_ceilf4),
        entry("Sleef_truncf4", &Sleef_truncf4),
        entry("Sleef_roundf4", &Sleef_roundf4),
        entry("Sleef_lgammaf4", &Sleef_lgammaf4_u10),
        entry("Sleef_atan2f4", &Sleef_atan2f4_u10),
        entry("Sleef_powf4", &Sleef_powf4_u10),
        entry("Sleef_fmodf4", &Sleef_fmodf4),

    // FP32 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
        entry("Sleef_acosf8", &Sleef_acosf8_u10),
        entry("Sleef_asinf8", &Sleef_asinf8_u10),
        entry("Sleef_atanf8", &Sleef_atanf8_u10),
        entry("Sleef_cosf8", &Sleef_cosf8_u10),
        entry("Sleef_sinf8", &Sleef_sinf8_u10),
        entry("Sleef_tanf8", &Sleef_tanf8_u10),
        entry("Sleef_coshf8", &Sleef_coshf8_u10),
        entry("Sleef_sinhf8", &Sleef_sinhf8_u10),
        entry("Sleef_tanhf8", &Sleef_tanhf8_u10),
        entry("Sleef_erff8", &Sleef_erff8_u10),
        entry("Sleef_erfcf8", &Sleef_erfcf8_u15),
        entry("Sleef_expf8", &Sleef_expf8_u10),
        entry("Sleef_expm1f8", &Sleef_expm1f8_u10),
        entry("Sleef_logf8", &Sleef_logf8_u10),
        entry("Sleef_log2f8", &Sleef_log2f8_u10),
        entry("Sleef_log10f8", &Sleef_log10f8_u10),
        entry("Sleef_log1pf8", &Sleef_log1pf8_u10),
        entry("Sleef_sqrtf8", &Sleef_sqrtf8_u05),
        entry("Sleef_fabsf8", &Sleef_fabsf8),
        entry("Sleef_floorf8", &Sleef_floorf8),
        entry("Sleef_ceilf8", &Sleef_ceilf8),
        entry("Sleef_truncf8", &Sleef_truncf8),
        entry("Sleef_roundf8", &Sleef_roundf8),
        entry("Sleef_lgammaf8", &Sleef_lgammaf8_u10),
        entry("Sleef_atan2f8", &Sleef_atan2f8_u10),
        entry("Sleef_powf8", &Sleef_powf8_u10),
        entry("Sleef_fmodf8", &Sleef_fmodf8),
#endif

        // FP64 Sleef functions -- SSE
        entry("Sleef_acosd2", &Sleef_acosd2_u10),
        entry("Sleef_asind2", &Sleef_asind2_u10),
        entry("Sleef_atand2", &Sleef_atand2_u10),
        entry("Sleef_cosd2", &Sleef_cosd2_u10),
        entry("Sleef_sind2", &Sleef_sind2_u10),
        entry("Sleef_tand2", &Sleef_tand2_u10),
        entry("Sleef_coshd2", &Sleef_coshd2_u10),
        entry("Sleef_sinhd2", &Sleef_sinhd2_u10),
        entry("Sleef_tanhd2", &Sleef_tanhd2_u10),
        entry("Sleef_erfd2", &Sleef_erfd2_u10),
        entry("Sleef_erfcd2", &Sleef_erfcd2_u15),
        entry("Sleef_expd2", &Sleef_expd2_u10),
        entry("Sleef_expm1d2", &Sleef_expm1d2_u10),
        entry("Sleef_logd2", &Sleef_logd2_u10),
        entry("Sleef_log2d2", &Sleef_log2d2_u10),
        entry("Sleef_log10d2", &Sleef_log10d2_u10),
        entry("Sleef_log1pd2", &Sleef_log1pd2_u10),
        entry("Sleef_sqrtd2", &Sleef_sqrtd2_u05),
        entry("Sleef_fabsd2", &Sleef_fabsd2),
        entry("Sleef_floord2", &Sleef_floord2),
        entry("Sleef_ceild2", &Sleef_ceild2),
        entry("Sleef_truncd2", &Sleef_truncd2),
        entry("Sleef_roundd2", &Sleef_roundd2),
        entry("Sleef_lgammad2", &Sleef_lgammad2_u10),
        entry("Sleef_atan2d2", &Sleef_atan2d2_u10),
        entry("Sleef_powd2", &Sleef_powd2_u10),
        entry("Sleef_fmodd2", &Sleef_fmodd2),

    // FP64 Sleef functions -- AVX2
#if defined(__AVX__) && !defined(_MSC_VER)
        entry("Sleef_acosd4", &Sleef_acosd4_u10),
        entry("Sleef_asind4", &Sleef_asind4_u10),
        entry("Sleef_atand4", &Sleef_atand4_u10),
        entry("Sleef_cosd4", &Sleef_cosd4_u10),
        entry("Sleef_sind4", &Sleef_sind4_u10),
        entry("Sleef_tand4", &Sleef_tand4_u10),
        entry("Sleef_coshd4", &Sleef_coshd4_u10),
        entry("Sleef_sinhd4", &Sleef_sinhd4_u10),
        entry("Sleef_tanhd4", &Sleef_tanhd4_u10),
        entry("Sleef_erfd4", &Sleef_erfd4_u10),
        entry("Sleef_erfcd4", &Sleef_erfcd4_u15),
        entry("Sleef_expd4", &Sleef_expd4_u10),
        entry("Sleef_expm1d4", &Sleef_expm1d4_u10),
        entry("Sleef_logd4", &Sleef_logd4_u10),
        entry("Sleef_log2d4", &Sleef_log2d4_u10),
        entry("Sleef_log10d4", &Sleef_log10d4_u10),
        entry("Sleef_log1pd4", &Sleef_log1pd4_u10),
        entry("Sleef_sqrtd4", &Sleef_sqrtd4_u05),
        entry("Sleef_fabsd4", &Sleef_fabsd4),
        entry("Sleef_floord4", &Sleef_floord4),
        entry("Sleef_ceild4", &Sleef_ceild4),
        entry("Sleef_truncd4", &Sleef_truncd4),
        entry("Sleef_roundd4", &Sleef_roundd4),
        entry("Sleef_lgammad4", &Sleef_lgammad4_u10),
        entry("Sleef_atan2d4", &Sleef_atan2d4_u10),
        entry("Sleef_powd4", &Sleef_powd4_u10),
        entry("Sleef_fmodd4", &Sleef_fmodd4),
#endif
  })));
}

namespace llvm {
namespace orc {

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
#if LLVM_VERSION_MAJOR >= 9 && LLVM_VERSION_MAJOR <= 12
class TORCH_API PytorchLLVMJITImpl {
 private:
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<LLJIT> LLJ;

 public:
  PytorchLLVMJITImpl()
      : TM(assertSuccess(makeTargetMachineBuilder().createTargetMachine())),
        LLJ(assertSuccess(
            LLJITBuilder()
                .setJITTargetMachineBuilder(makeTargetMachineBuilder())
                .create())) {
    auto ProcSymbolsGenerator =
        assertSuccess(DynamicLibrarySearchGenerator::GetForCurrentProcess(
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
    registerIntrinsics(JD, Mangle);
  }

  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C) {
    assertSuccess(
        LLJ->addIRModule(ThreadSafeModule(std::move(M), std::move(C))),
        "Failed to add module to compile layer");
  }

  JITSymbol findSymbol(const std::string Name) {
    return assertSuccess(LLJ->lookup(Name));
  }

  bool hasSymbol(const std::string& Name) {
    return (bool)LLJ->lookup(Name);
  }

  TargetMachine& getTargetMachine() {
    return *TM;
  }

  const DataLayout& getDataLayout() {
    return LLJ->getDataLayout();
  }
};

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
              if (auto Sym = CompileLayer.findSymbol(Name, false)) {
                return Sym;
              } else if (auto Err = Sym.takeError()) {
                return std::move(Err);
              }
              if (auto SymAddr =
                      RTDyldMemoryManager::getSymbolAddressInProcess(Name)) {
                return JITSymbol(SymAddr, JITSymbolFlags::Exported);
              }
              MangleAndInterner Mangle(ES, DL);
              return assertSuccess(
                  lookup({&ES.getMainJITDylib()}, Mangle(Name)));
            },
            [](Error Err) {
              assertSuccess(std::move(Err), "lookupFlags failed");
            })),
        TM(assertSuccess(makeTargetMachineBuilder().createTargetMachine())),
        DL(TM->createDataLayout()),
        ObjectLayer(
            ES,
            [this](VModuleKey) {
              return RTDyldObjectLinkingLayer::Resources{
                  std::make_shared<SectionMemoryManager>(), Resolver};
            }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)) {
    auto& JD = ES.getMainJITDylib();
    MangleAndInterner Mangle(ES, DL);
    registerIntrinsics(JD, Mangle);
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  TargetMachine& getTargetMachine() {
    return *TM;
  }

  void addModule(std::unique_ptr<Module> M, std::unique_ptr<LLVMContext> C) {
    // Add the module to the JIT with a new VModuleKey.
    auto K = ES.allocateVModule();
    assertSuccess(
        CompileLayer.addModule(K, std::move(M)),
        "Failed to add module to compile layer");
  }

  JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompileLayer.findSymbol(MangledNameStream.str(), true);
  }

  bool hasSymbol(const std::string& Name) {
    MangleAndInterner mangle(ES, DL);
    return (bool)ES.lookup({mangle(Name)});
  }

  JITTargetAddress getSymbolAddress(const std::string Name) {
    return assertSuccess(findSymbol(Name).getAddress());
  }

  void removeModule(VModuleKey K) {
    assertSuccess(CompileLayer.removeModule(K));
  }

  const DataLayout& getDataLayout() {
    return DL;
  }
};

#else // LLVM_VERSION_MAJOR
#error Only LLVM versions 8 through 12 are supported.
#endif

PytorchLLVMJIT::PytorchLLVMJIT()
    : impl_(std::make_unique<PytorchLLVMJITImpl>()) {}

PytorchLLVMJIT::~PytorchLLVMJIT() = default;

void PytorchLLVMJIT::addModule(
    std::unique_ptr<Module> M,
    std::unique_ptr<LLVMContext> C) {
  return impl_->addModule(std::move(M), std::move(C));
}

JITSymbol PytorchLLVMJIT::findSymbol(const std::string Name) {
  return impl_->findSymbol(std::move(Name));
}

bool PytorchLLVMJIT::hasSymbol(const std::string& Name) {
  return impl_->hasSymbol(Name);
}

TargetMachine& PytorchLLVMJIT::getTargetMachine() {
  return impl_->getTargetMachine();
}

const DataLayout& PytorchLLVMJIT::getDataLayout() {
  return impl_->getDataLayout();
}

} // end namespace orc
} // end namespace llvm

#endif // TORCH_ENABLE_LLVM
