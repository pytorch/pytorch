#include <torch/csrc/jit/fuser/cpu/fused_kernel.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/code_template.h>
#include <torch/csrc/jit/fuser/compiler.h>
#include <torch/csrc/jit/fuser/cpu/temp_file.h>
#include <torch/csrc/utils/memory.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

static const std::string so_template = "/tmp/pytorch_fuserXXXXXX.so";
static const std::string cpp_template = "/tmp/pytorch_fuserXXXXXX.cpp";
static const std::string check_exists_string = "which '${program}' > /dev/null";

static bool programExists(const std::string& program) {
  TemplateEnv env;
  env.s("program", program);
  std::string cmd = format(check_exists_string, env);
  return (system(cmd.c_str()) == 0);
}

// A single compiler config is accessed through getConfig() (below)
// Controls compilation options and may be updated based on the result
// of compilation attempts.
struct CompilerConfig {
  CompilerConfig() {
    const char* cxx_env = getenv("CXX");
    if (cxx_env != nullptr) {
      cxx = cxx_env;
    }

    if (!programExists(cxx)) {
      cxx = "";
    }
  }

  ~CompilerConfig() = default;

  std::string cxx = "g++"; // compiler location
  bool openmp = true;
};

static CompilerConfig& getConfig() {
  static CompilerConfig config;
  return config;
}

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
    const std::string& cpp_file,
    const std::string& so_file) {
  auto& config = getConfig();
  TemplateEnv env;
  env.s("cxx", config.cxx);
  env.s("fopenmp", config.openmp ? "-fopenmp" : "");
  env.s("cpp_file", cpp_file);
  env.s("so_file", so_file);
  std::string result = format(compile_string, env);
  int r = system(result.c_str());
  if (config.openmp && r != 0) {
    std::cerr
        << "warning: pytorch jit fuser failed to compile with openmp, trying without it...\n";
    config.openmp = false; // disable for future compiles
    return runCompiler(cpp_file, so_file);
  }
  TORCH_CHECK(r == 0, "Failed to compile a fused CPU kernel");
}

static const std::string disas_string = "objdump -M  intel -d \"${so_file}\"";
static void disas(const std::string& so_file) {
  TemplateEnv env;
  env.s("so_file", so_file);
  std::string cmd = format(disas_string, env);
  int r = system(cmd.c_str());
  AT_ASSERT(r == 0);
}

FusedKernelCPU::FusedKernelCPU(
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)
    : FusedKernel(
          std::move(name),
          std::move(code),
          std::move(input_desc),
          std::move(output_desc),
          std::move(chunk_desc),
          std::move(concat_desc),
          has_random) {
  TempFile so_file(so_template, 3);
  TempFile cpp_file(cpp_template, 4);
  cpp_file.write(code_);
  cpp_file.sync();
  runCompiler(cpp_file.name(), so_file.name());
  if (debugFuser() >= 2)
    disas(so_file.name());
  so_lib = make_unique<at::DynamicLibrary>(so_file.name().c_str());
#pragma GCC diagnostic ignored "-Wpedantic"
  kernel =
      reinterpret_cast<void (*)(uint32_t, void**)>(so_lib->sym(name_.c_str()));
#pragma GCC diagnostic pop
}

static std::shared_ptr<FusedKernel> createFusionKernel(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random) {
  return std::make_shared<FusedKernelCPU>(
      std::move(name),
      std::move(code),
      std::move(input_desc),
      std::move(output_desc),
      std::move(chunk_desc),
      std::move(concat_desc),
      has_random);
}

RegisterFusionBackend reg(at::DeviceType::CPU, createFusionKernel);
} // namespace cpu
} // namespace fuser
} // namespace jit
} // namespace torch
