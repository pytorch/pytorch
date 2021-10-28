#include <torch/csrc/deploy/deploy.h>
#include <torch/csrc/deploy/unity/xar_environment.h>
#include <memory>

namespace torch {
namespace deploy {

// the way we lookup main module follows how an xar file is setup
std::string lookupMainModule(InterpreterManager& m) {
  auto I = m.acquireOne();
  auto mainModule =
      I.global("__manifest__", "fbmake").attr("get")({"main_module"});
  std::ostringstream ss;
  ss << mainModule.toIValue();
  LOG(INFO) << "main module is " << ss.str();
  return ss.str();
}

int doMain(int argc, char** argv) {
  std::unique_ptr<Environment> env = std::make_unique<XarEnvironment>();
  InterpreterManager m(2, std::move(env));

  auto mainModule = lookupMainModule(m);
  auto I = m.acquireOne();
  I.global("runpy", "run_module")({mainModule});
  return 0;
}

} // namespace deploy
} // namespace torch

int main(int argc, char** argv) {
  return torch::deploy::doMain(argc, argv);
}
