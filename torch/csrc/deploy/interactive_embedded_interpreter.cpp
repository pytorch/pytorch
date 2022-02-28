/*
 * The tool provides a shell to the embedded interpreter. Useful to inspect the
 * state of the embedding interpreter interactively.
 */
#include <c10/util/Flags.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/csrc/deploy/path_environment.h>
#include <torch/csrc/deploy/ArrayRef.h>

C10_DEFINE_string(
    python_path,
    "",
    "The root of the installed python libraries in the system");
C10_DEFINE_string(pyscript, "", "The path of the python script to execute");

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
  c10::ParseCommandLineFlags(&argc, &argv);

  if (FLAGS_python_path.size() > 0) {
    LOG(INFO) << "Will add " << FLAGS_python_path << " to python sys.path";
  }
  std::shared_ptr<torch::deploy::Environment> env =
      std::make_shared<torch::deploy::PathEnvironment>(FLAGS_python_path);
  // create multiple interpreter instances so the tool does not just cover the
  // simplest case with a single interpreter instance.
  torch::deploy::InterpreterManager m(2, env);
  auto I = m.acquireOne();

  if (FLAGS_pyscript.size() > 0) {
    auto realpath = I.global("os", "path").attr("expanduser")({FLAGS_pyscript});
    I.global("runpy", "run_path")({realpath});
  } else {
    c10::ArrayRef<torch::deploy::Obj> noArgs;
    I.global("pdb", "set_trace")(noArgs);
  }
  return 0;
}
