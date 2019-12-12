#ifndef NNC_LIB_LLVM_JIT_H_
#define NNC_LIB_LLVM_JIT_H_

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

// Lightly modified implementation from LLVM's Kaleidoscope JIT tutorial:
// https://llvm.org/docs/tutorial/BuildingAJIT1.html
class PytorchLlvmJit {
 private:
  llvm::orc::ExecutionSession ES;
  std::shared_ptr<llvm::orc::SymbolResolver> Resolver;
  std::unique_ptr<llvm::TargetMachine> TM;
  const llvm::DataLayout DL;
  llvm::orc::LegacyRTDyldObjectLinkingLayer ObjectLayer;
  llvm::orc::LegacyIRCompileLayer<decltype(ObjectLayer), llvm::orc::SimpleCompiler> CompileLayer;

 public:
  PytorchLlvmJit()
    : Resolver(createLegacyLookupResolver(
                 ES,
                 [this](const std::string &Name) -> llvm::JITSymbol {
                   if (auto Sym = CompileLayer.findSymbol(Name, false))
                     return Sym;
                   else if (auto Err = Sym.takeError())
                     return std::move(Err);
                   if (auto SymAddr =
                       llvm::RTDyldMemoryManager::getSymbolAddressInProcess(Name))
                     return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
                   return nullptr;
                 },
                 [](llvm::Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
      TM(llvm::EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
      ObjectLayer(ES,
                  [this](llvm::orc::VModuleKey) {
                    return llvm::orc::LegacyRTDyldObjectLinkingLayer::Resources{
                      std::make_shared<llvm::SectionMemoryManager>(), Resolver};
                  }),
      CompileLayer(ObjectLayer, llvm::orc::SimpleCompiler(*TM)) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  llvm::TargetMachine &getTargetMachine() { return *TM; }

  llvm::orc::VModuleKey addModule(std::unique_ptr<llvm::Module> M) {
    // Add the module to the JIT with a new VModuleKey.
    auto K = ES.allocateVModule();
    cantFail(CompileLayer.addModule(K, std::move(M)));
    return K;
  }

  llvm::JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    llvm::raw_string_ostream MangledNameStream(MangledName);
    llvm::Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CompileLayer.findSymbol(MangledNameStream.str(), true);
  }

  llvm::JITTargetAddress getSymbolAddress(const std::string Name) {
    return cantFail(findSymbol(Name).getAddress());
  }

  void removeModule(llvm::orc::VModuleKey K) {
    cantFail(CompileLayer.removeModule(K));
  }
};

#endif // NNC_LIB_LLVM_JIT_H_
