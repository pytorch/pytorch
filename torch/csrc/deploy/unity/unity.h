#pragma once
#include <torch/csrc/deploy/deploy.h>
#include <string>

#define DEFAULT_PYTHON_APP_DIR "/tmp/torch_deploy_python_app"

namespace torch {
namespace deploy {
class Unity {
 public:
  explicit Unity(
      int nInterp,
      std::string pythonAppDir = DEFAULT_PYTHON_APP_DIR);

  const std::string& getPythonAppRoot() const {
    return pythonAppRoot_;
  }

  InterpreterManager& getInterpreterManager() {
    return *interpreterManager_;
  }

  // This method simply runs the main module for the python application. But
  // sometimew, people may want to call an python method instead. In
  // that case, they can call getInterpreterManager() to get the
  // InterpreterManager, acquire an InterpreterSession and do whatever they
  // need.
  void runMainModule();

 private:
  void setupPythonApp();
  void preloadSharedLibraries();
  std::string lookupMainModule();

  std::string pythonAppDir_;
  std::string pythonAppRoot_;
  bool alreadySetupPythonApp_ = false;
  std::unique_ptr<InterpreterManager> interpreterManager_;
  std::string mainModule_; // the fully qualified name of the main module
};
} // namespace deploy
} // namespace torch
