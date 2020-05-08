#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

namespace {

cl::opt<std::string> InputFilename(
    cl::Positional,
    cl::desc("<input bitcode file>"),
    cl::init("-"),
    cl::value_desc("filename"));

} // namespace

int main(int argc, char **argv) {
  LLVMContext Context;
  cl::ParseCommandLineOptions(argc, argv);
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);

  auto opDependencyPass = PassRegistry::getPassRegistry()
      ->getPassInfo(StringRef("op_dependency"))
      ->createPass();
  static_cast<ModulePass*>(opDependencyPass)->runOnModule(*M);
  return 0;
}
