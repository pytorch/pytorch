#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/export.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-of-script-module> <path-of-bytecode-file>";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module = torch::jit::load(argv[1]);

  // Save in new format with code and pkl files.
  std::string nfile(argv[1]);
  nfile += "1.pt";
  module.save(nfile);

  // Save in bytecode format.
  module.save(argv[2], torch::jit::script::ExtraFilesMap(), true /*bytecode_format*/);
}
