#ifdef TORCH_ENABLE_LLVM

#include <torch/csrc/jit/tensorexpr/intrinsic_symbols.h>
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
#include <llvm/Support/CFGUpdate.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

#include <c10/util/Half.h>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
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
    llvm::orc::MangleAndInterner& Mangle,
    std::unordered_set<std::string>& intrinsics) {
  using namespace llvm;
  using namespace llvm::orc;

  auto entry = [&](const char* name, auto ptr) -> SymbolMap::value_type {
    return {Mangle(name), {toAddress(ptr), JITSymbolFlags::None}};
  };

  SymbolMap symbols;
  for (auto const& sym : getIntrinsicSymbols()) {
    symbols.insert(entry(sym.symbol, sym.address));
    intrinsics.insert(sym.symbol);
  }
  assertSuccess(JD.define(absoluteSymbols(symbols)));

  for (const auto& kv : getNNCFunctionRegistry()) {
    assertSuccess(
        JD.define(absoluteSymbols({entry(kv.first.c_str(), kv.second)})));
  }
  assertSuccess(JD.define(
      absoluteSymbols({entry("DispatchParallel", DispatchParallel)})));
}

namespace llvm {
namespace orc {

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
#if LLVM_VERSION_MAJOR >= 9
class TORCH_API PytorchLLVMJITImpl {
 private:
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<LLJIT> LLJ;
  std::unordered_set<std::string> intrinsics;

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
    registerIntrinsics(JD, Mangle, intrinsics);
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
    return intrinsics.find(Name) != intrinsics.end();
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
  std::unordered_set<std::string> intrinsics;

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
    registerIntrinsics(JD, Mangle, intrinsics);
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
    return intrinsics.find(Name) != intrinsics.end();
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
#error Only LLVM versions 8 and above are supported.
#endif

PytorchLLVMJIT::PytorchLLVMJIT()
    : impl_(std::make_unique<PytorchLLVMJITImpl>()) {}

PytorchLLVMJIT::~PytorchLLVMJIT() = default;

void PytorchLLVMJIT::addModule(
    std::unique_ptr<Module> M,
    std::unique_ptr<LLVMContext> C) {
  impl_->addModule(std::move(M), std::move(C));
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

#if !defined(NDEBUG)
void dumpCFG(const llvm::cfg::Update<llvm::BasicBlock*>& update) {
  // XXX: This method call is only here to placate gcov builds.  The `dump`
  // method is conditionally defined when NDEBUG is unset, so if you try to
  // link a debug-mode pytorch with an opt-mode llvm, the symbol is undefined.
  update.dump();
}
#endif

} // end namespace orc
} // end namespace llvm

#endif // TORCH_ENABLE_LLVM
