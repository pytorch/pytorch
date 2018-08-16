#ifndef _WIN32

#include "torch/csrc/jit/fusers/cpu/fusion_compiler.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/assertions.h"

#include "ATen/ATen.h"

#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <iostream>
#include <cstdlib>

namespace torch { namespace jit { namespace cpufuser {

// // Note: there is only one CPUFusionCompiler
CPUFusionCompiler& getCompiler() { 
  static CPUFusionCompiler compiler;
  return compiler; 
}

static const std::string check_exists_string = "which '${program}' > /dev/null";

static bool programExists(const std::string& program) {
  TemplateEnv env;
  env.s("program", program);
  std::string cmd = format(check_exists_string, env);
  return 0 == system(cmd.c_str());
}

CPUFusionCompiler::CPUFusionCompiler() {
  const char* cxx_env = getenv("CXX");
  if (cxx_env != nullptr) config_.cxx = cxx_env;
  if (!programExists(config_.cxx)) config_.cxx = "";
  const char* debug_env = getenv("PYTORCH_FUSION_DEBUG");
  config_.debug = debug_env && atoi(debug_env) != 0;
}

std::shared_ptr<CPUFusionFunction> CPUFusionCompiler::getOrCompile(
  AnnotatedGraph& agraph) {
  std::stringstream key;
  key << *agraph.graph << "\n";
  key << "device " << agraph.device << "\n";
  for (auto& i : agraph.input_desc) key << i << "\n";
  for (auto& i : agraph.output_desc) key << i << "\n";
  std::string key_ = key.str();

  auto it = cache.find(key_);
  if (it == cache.end()) {
    JIT_ASSERT(agraph.device == kCPUDevice);
    JIT_ASSERT(canCompileOnCPU());
    std::string name = "kernel_" + std::to_string(cache.size());
    CPUFusionFunction* raw_func = new CPUFusionFunction(name, agraph, config_);
    it = cache.emplace(key_, std::shared_ptr<CPUFusionFunction>(raw_func)).first;
  }
  return it->second;
}

std::shared_ptr<CPUFusionFunction> CPUFusionCompiler::getOrCompile(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs
, at::ArrayRef<at::Tensor> outputs) {
  AnnotatedGraph agraph(graph, device);
  for(auto& i : inputs) agraph.input_desc.emplace_back(i);
  for(auto& i : outputs) agraph.output_desc.emplace_back(i);
  return getOrCompile(agraph);
}

void CPUFusionCompiler::debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs
, at::ArrayRef<at::Tensor> outputs) {
  auto func = getOrCompile(graph, device, inputs, outputs);
  func->launch_with_tensors(inputs, outputs);
}

} // namespace cpufuser
} // namespace jit
} // namespace torch

#endif // _WIN32