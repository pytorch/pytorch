#include "torch/csrc/jit/fuser/cpu/fused_kernel.h"

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/fuser/cpu/config.h"
#include "torch/csrc/jit/fuser/cpu/temp_file.h"
#include "torch/csrc/jit/fuser/cpu/dynamic_library.h"
#include "torch/csrc/jit/fuser/common/annotated_graph.h"

#include <sstream>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>


namespace torch { namespace jit { namespace fuser { namespace cpu {

static const std::string so_template = "/tmp/pytorch_fuserXXXXXX.so";
static const std::string cpp_template = "/tmp/pytorch_fuserXXXXXX.cpp";

// NB: -march=native not supported on PPC64 g++.  It's a bit annoying
// to do a configure-style test to decide whether or not the g++
// actually supports it or not, so we heuristically use the host
// compiler to predict if the runtime compiler supports the option we
// want.  This probably won't work if you're cross-compiling.
// NB: -march=native is disabled because it has caused problems where
// compiler and assembler do not agree on what native instruction they
// understand for AVX512. When we need better CPU performance this
// optimization can be re-enabled by tracking down the platforms where
// this error occurs and only selectively disabling it.
static const std::string compile_string =
  "\"${cxx}\" -O3 -g "
#ifndef __PPC64__
//  "-march=native "
#endif
  "-std=c++11 -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\" -lm";

static void runCompiler(
  CompilerConfig& config
, const std::string& cpp_file
, const std::string& so_file) {
  TemplateEnv env;
  env.s("cxx", config.cxx);
  env.s("fopenmp", config.openmp ? "-fopenmp" : "");
  env.s("cpp_file", cpp_file);
  env.s("so_file", so_file);
  std::string result = format(compile_string, env);
  int r = system(result.c_str());
  if (config.openmp && r != 0) {
    std::cerr << "warning: pytorch jit fuser failed to compile with openmp, trying without it...\n";
    config.openmp = false; // disable for future compiles
    return runCompiler(config, cpp_file, so_file);
  }
  throw std::runtime_error("Failed to compile a fused CPU kernel.");
}

static const std::string disas_string =
  "objdump -M  intel -d \"${so_file}\"";
static void disas(const std::string& so_file) {
  TemplateEnv env;
  env.s("so_file", so_file);
  std::string cmd = format(disas_string, env);
  int r = system(cmd.c_str());
  JIT_ASSERT(r == 0);
}

FusedKernelCPU::FusedKernelCPU(
  CompilerConfig& config
, const std::string& _name
, const std::string& _code
, const std::vector<TensorDesc> _input_desc
, const std::vector<TensorDesc> _output_desc
, const std::vector<PartitionDesc> _chunk_desc
, const std::vector<PartitionDesc> _concat_desc
, const bool _has_random)
: FusedKernel{_name, _code, _input_desc, _output_desc, _chunk_desc, _concat_desc, _has_random} {
  TempFile so_file(so_template, 3);
  TempFile cpp_file(cpp_template, 4);
  cpp_file.write(code_);
  cpp_file.sync();
  runCompiler(config, cpp_file.name(), so_file.name());
  if (config.debug) disas(so_file.name());
  so_lib.reset(new DynamicLibrary(so_file.name().c_str()));
  #pragma GCC diagnostic ignored "-Wpedantic"
    kernel = reinterpret_cast<void(*)(uint32_t, void**)>(so_lib->sym(name_.c_str()));
  #pragma GCC diagnostic pop
}

} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
