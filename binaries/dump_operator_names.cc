/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/instruction.h>
#include <c10/util/Flags.h>

#include <fstream>

namespace torch {
namespace jit {
void dump_opnames(const script::Module& m, std::unordered_set<std::string>& opnames) {
  auto methods = m.get_methods();
  for (const auto& method : methods) {
    const auto& func = method.function();
    std::cout << "function name: " << func.name() << std::endl;
    torch::jit::Code code(func.graph());
    for (size_t i = 0; i < code.instructions().size(); ++i) {
      auto ins = code.instructions()[i];
      auto node = code.instructions_source()[i];
      if (ins.op == OpCode::OP) {
        auto opname = node->schema().operator_name();
        std::string namestr = opname.name;
        if (!opname.overload_name.empty())
          namestr += "." + opname.overload_name;
        std::cout << "    " << namestr << std::endl;
        opnames.emplace(namestr);
      }
    }
  }
  for (const auto& sub_m : m.children()) {
    std::cout << "sub module name: " << sub_m.name().qualifiedName() << std::endl;
    dump_opnames(sub_m, opnames);
  }
}
}
}

C10_DEFINE_string(model, "", "The given torch script model.");
C10_DEFINE_string(output, "", "The output yaml file of operator list.");

int main(int argc, char** argv) {
  c10::SetUsageMessage(
    "Dump operators in a script module and its sub modules.\n"
    "Example usage:\n"
    "./dump_operator_names"
    " --model=<model_file>"
    " --output=<output.yaml>");

  if (!c10::ParseCommandLineFlags(&argc, &argv)) {
    std::cerr << "Failed to parse command line flags!" << std::endl;
    return 1;
  }

  CAFFE_ENFORCE_GE(FLAGS_model.size(), 0, "Model file must be specified.");
  CAFFE_ENFORCE_GE(FLAGS_output.size(), 0, "Output yaml file must be specified.");

  auto m = torch::jit::load(FLAGS_model);
  std::unordered_set<std::string> opnames;
  torch::jit::dump_opnames(m, opnames);
  std::ofstream ofile(FLAGS_output);
  std::cout << "-- Final List --" << std::endl;
  for (const auto& name : opnames) {
    std::cout << name << std::endl;
    ofile << "- " << name << std::endl;
  }
  ofile.close();
}
