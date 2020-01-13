#include "torch/csrc/jit/compiler/include/llvm_jit.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

static llvm::SmallVector<std::string, 0> getAttrs() {
  llvm::SmallVector<std::string, 0> res;
  llvm::StringMap<bool> features;
  if (llvm::sys::getHostCPUFeatures(features)) {
    for (auto const& feature : features) {
      if (feature.second) {
        res.push_back(feature.first());
      }
    }
  }
  return res;
}

namespace llvm {
namespace orc {

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
class PytorchLLVMJITImpl {
 private:
#if LLVM_VERSION_MAJOR == 8 || LLVM_VERSION_MAJOR == 9
  using JITLinkingLayer = LegacyRTDyldObjectLinkingLayer;
  template <typename B, typename C>
  using JITCompileLayer = LegacyIRCompileLayer<B, C>;
#elif LLVM_VERSION_MAJOR == 7
  using JITLinkingLayer = RTDyldObjectLinkingLayer;
  template <typename B, typename C>
  using JITCompileLayer = IRCompileLayer<B, C>;
#else
#error "Supported LLVM versions: 7, 8"
#endif

  ExecutionSession ES;
  std::shared_ptr<SymbolResolver> Resolver;
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  JITLinkingLayer ObjectLayer;
  JITCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;

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
        TM(EngineBuilder().setCodeModel(CodeModel::Medium).selectTarget(
            llvm::Triple(),
            "",
            llvm::sys::getHostCPUName(),
            getAttrs())),
        DL(TM->createDataLayout()),
        ObjectLayer(
            ES,
            [this](VModuleKey) {
              return JITLinkingLayer::Resources{
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
};

PytorchLLVMJIT::PytorchLLVMJIT()
    : impl_(std::make_unique<PytorchLLVMJITImpl>()) {}

PytorchLLVMJIT::~PytorchLLVMJIT() = default;

TargetMachine& PytorchLLVMJIT::getTargetMachine() {
  return impl_->getTargetMachine();
}

VModuleKey PytorchLLVMJIT::addModule(std::unique_ptr<Module> M) {
  return impl_->addModule(std::move(M));
}

JITSymbol PytorchLLVMJIT::findSymbol(const std::string Name) {
  return impl_->findSymbol(Name);
}

JITTargetAddress PytorchLLVMJIT::getSymbolAddress(const std::string Name) {
  return impl_->getSymbolAddress(Name);
}

void PytorchLLVMJIT::removeModule(VModuleKey K) {
  impl_->removeModule(K);
}

} // end namespace orc
} // end namespace llvm
