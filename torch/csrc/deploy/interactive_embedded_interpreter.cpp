/*
 * The tool provides a shell to the embedded interpreter. Useful to inspect the
 * state of the embedding interpreter interactively.
 */
#include <torch/csrc/deploy/deploy.h>

int main(int argc, char** argv) {
  // create multiple interpreter instances so the tool does not just cover the
  // simplest case with a single interpreter instance.
  torch::deploy::InterpreterManager m(2);
  auto I = m.acquire_one();
  c10::ArrayRef<torch::deploy::Obj> no_args;
  I.global("pdb", "set_trace")(no_args);
  return 0;
}
