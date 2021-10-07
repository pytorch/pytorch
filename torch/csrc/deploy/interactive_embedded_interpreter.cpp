/*
 * The tool provides a shell to the embedded interpreter. Useful to inspect the
 * state of the embedding interpreter interactively.
 */
#include <torch/csrc/deploy/deploy.h>

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char** argv) {
  // create multiple interpreter instances so the tool does not just cover the
  // simplest case with a single interpreter instance.
  torch::deploy::InterpreterManager m(2);
  auto I = m.acquireOne();
  c10::ArrayRef<torch::deploy::Obj> noArgs;
  I.global("pdb", "set_trace")(noArgs);
  return 0;
}
