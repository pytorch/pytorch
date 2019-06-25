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

#ifdef _MSC_VER
static const std::string getTempPath() {
  char lpTempPathBuffer[MAX_PATH];

  DWORD dwRetVal = GetTempPath(
      MAX_PATH, // length of the buffer
      lpTempPathBuffer); // buffer for path

  TORCH_CHECK(dwRetVal < MAX_PATH && dwRetVal != 0, "GetTempPath failed.");

  return std::string(lpTempPathBuffer);
}
static const std::string temp_dir = getTempPath();
static const std::string so_template = temp_dir + "pytorch_fuserXXXXXX.dll";
static const std::string cpp_template = temp_dir + "pytorch_fuserXXXXXX.cpp";
static const std::string check_exists_string = "where \"${program}\" > nul 2> nul";
constexpr int so_suffix_len = 4;
constexpr int cpp_suffix_len = 4;
#else
static const std::string so_template = "/tmp/pytorch_fuserXXXXXX.so";
static const std::string cpp_template = "/tmp/pytorch_fuserXXXXXX.cpp";
static const std::string check_exists_string = "which '${program}' > /dev/null";
constexpr int so_suffix_len = 3;
constexpr int cpp_suffix_len = 4;
#endif

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
      #ifdef _MSC_VER
        // First check whether vswhere exists.
        int rc = system("if not exist \"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\" exit /b 1");
        if (rc == 0) {
          // Second check whether a valid MSVC installation exists.
          rc = system("@echo off && for /f \"usebackq tokens=*\" %i in (`\"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\" -version [15^,17^) -products * -latest -prerelease -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call \"%i\\VC\\Auxiliary\\Build\\vcvarsall.bat\" x64 > NUL && where cl > NUL 2> NUL");
        }
        if (rc == 0) {
          // Append enviornment activation scripts to the compiler.
          cxx = "@echo off && for /f \"usebackq tokens=*\" %i in (`\"%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe\" -version [15^,17^) -products * -latest -prerelease -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call \"%i\\VC\\Auxiliary\\Build\\vcvarsall.bat\" x64 > NUL && cl";
        } else {
          cxx = "";
        }
      #else
        cxx = "";
      #endif
    }
  }

  ~CompilerConfig() = default;

  #ifdef _MSC_VER
    std::string cxx = "cl";
    const std::string openmp_flags = "/openmp";
  #else
    std::string cxx = "g++";
    const std::string openmp_flags = "-fopenmp";
  #endif
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
#ifdef _MSC_VER
static const std::string compile_string =
    "cd /D \"" + temp_dir + "\" && "
    "${cxx} /MD /Ox /LD /EHsc "
    "${fopenmp} \"${cpp_file}\" /link /out:\"${so_file}\"";
#else
static const std::string compile_string =
    "\"${cxx}\" -O3 -g "
#ifndef __PPC64__
//  "-march=native "
#endif
    "-std=c++11 -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\" -lm";
#endif
static void runCompiler(
    const std::string& cpp_file,
    const std::string& so_file) {
  auto& config = getConfig();
  TemplateEnv env;
  env.s("cxx", config.cxx);
  env.s("fopenmp", config.openmp ? config.openmp_flags : "");
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

#ifdef _MSC_VER
static const std::string disas_string = "dumpbin /DISASM:NOBYTES \"${so_file}\"";
#else
static const std::string disas_string = "objdump -M  intel -d \"${so_file}\"";
#endif
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
  TempFile so_file(so_template, so_suffix_len);
  TempFile cpp_file(cpp_template, cpp_suffix_len);
  cpp_file.write(code_);
  cpp_file.sync();
#ifdef _MSC_VER
  so_file.close();
  cpp_file.close();
#endif
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
