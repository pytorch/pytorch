#include "torch/csrc/jit/fusers/cpu/fused_kernel.h"

#include "torch/csrc/jit/fusers/cpu/fusion_compiler.h"
#include "torch/csrc/jit/fusers/cpu/temp_file.h"
#include "torch/csrc/jit/fusers/cpu/dynamic_library.h"
#include "torch/csrc/jit/fusers/common/annotated_graph.h"

#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/code_template.h"

#include <sstream>
#include <tuple>
#include <cstdlib>
#include <iostream>
#include <string>


namespace torch { namespace jit { namespace cpufuser {

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
  CPUFusionCompilerConfig& config
, const std::string& cpp_file
, const std::string& so_file) {
  TemplateEnv env;
  env.s("cxx", config.cxx);
  env.s("fopenmp", config.openmp ? "-fopenmp" : "");
  env.s("cpp_file",cpp_file);
  env.s("so_file",so_file);
  std::string result = format(compile_string, env);
  int r = system(result.c_str());
  if (config.openmp && r != 0) {
    std::cerr << "warning: pytorch jit fuser failed to compile with openmp, trying without it...\n";
    config.openmp = false; // disable for future compiles
    return runCompiler(config, cpp_file, so_file);
  }
  JIT_ASSERTM(r == 0, "Failed to compile a fused CPU kernel");
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

CPUFusedKernel::CPUFusedKernel(
  const std::string& name
, AnnotatedGraph& agraph
, CPUFusionCompilerConfig& config)
: FusedKernel(name, agraph) {
  TempFile so_file(so_template, 3);
  TempFile cpp_file(cpp_template, 4);

  std::stringstream cu;
  std::tie(chunk_desc, concat_desc, has_random) = emitCompilationUnit(cu, name, agraph, false);
  JIT_ASSERT(!has_random);
  compilation_unit = cu.str();
  cpp_file.write(compilation_unit);
  cpp_file.sync();
  runCompiler(config, cpp_file.name(), so_file.name());
  if (config.debug) {
    disas(so_file.name());
  }
  so_lib.reset(new DynamicLibrary(so_file.name().c_str()));
  #pragma GCC diagnostic ignored "-Wpedantic"
    kernel = reinterpret_cast<void(*)(uint32_t, void**)>(so_lib->sym(name.c_str()));
  #pragma GCC diagnostic pop
}

} // namespace cpufuser
} // namespace jit
} // namespace torch
