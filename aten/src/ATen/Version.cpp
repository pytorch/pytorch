#include <ATen/Version.h>
#include <ATen/Config.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#if AT_ONEDNN_ENABLED()
#include <dnnl.hpp>
#include <ideep.hpp>
#endif

#include <caffe2/core/common.h>

#include <ATen/native/DispatchStub.h>

#include <sstream>

namespace at {

std::string get_mkl_version() {
  std::string version;
  #if AT_MKL_ENABLED()
    {
      // Magic buffer number is from MKL documentation
      // https://software.intel.com/en-us/mkl-developer-reference-c-mkl-get-version-string
      char buf[198];
      mkl_get_version_string(buf, 198);
      version = buf;
    }
  #else
    version = "MKL not found";
  #endif
  return version;
}

std::string get_onednn_version() {
  std::ostringstream ss;
  #if AT_ONEDNN_ENABLED()
    // Cribbed from mkl-dnn/src/common/verbose.cpp
    // Too bad: can't get ISA info conveniently :(
    // Apparently no way to get ideep version?
    // https://github.com/intel/ideep/issues/29
    {
      const dnnl_version_t* ver = dnnl_version();
      ss << "Intel(R) MKL-DNN v" << ver->major << "." << ver->minor << "." << ver->patch
         << " (Git Hash " << ver->hash << ")";
    }
  #else
    ss << "ONEDNN not found";
  #endif
  return ss.str();
}

std::string get_openmp_version() {
  std::ostringstream ss;
  #ifdef _OPENMP
    {
      ss << "OpenMP " << _OPENMP;
      // Reference:
      // https://stackoverflow.com/questions/1304363/how-to-check-the-version-of-openmp-on-linux
      const char* ver_str = nullptr;
      switch (_OPENMP) {
        case 200505:
          ver_str = "2.5";
          break;
        case 200805:
          ver_str = "3.0";
          break;
        case 201107:
          ver_str = "3.1";
          break;
        case 201307:
          ver_str = "4.0";
          break;
        case 201511:
          ver_str = "4.5";
          break;
        default:
          ver_str = nullptr;
          break;
      }
      if (ver_str) {
        ss << " (a.k.a. OpenMP " << ver_str << ")";
      }
    }
  #else
    ss << "OpenMP not found";
  #endif
  return ss.str();
}

std::string get_cpu_capability() {
  // It is possible that we override the cpu_capability with
  // environment variable
  auto capability = native::get_cpu_capability();
  switch (capability) {
#if defined(HAVE_VSX_CPU_DEFINITION)
    case native::CPUCapability::DEFAULT:
      return "DEFAULT";
    case native::CPUCapability::VSX:
      return "VSX";
#elif defined(HAVE_ZVECTOR_CPU_DEFINITION)
    case native::CPUCapability::DEFAULT:
      return "DEFAULT";
    case native::CPUCapability::ZVECTOR:
      return "Z VECTOR";
#else
    case native::CPUCapability::DEFAULT:
      return "NO AVX";
    case native::CPUCapability::AVX2:
      return "AVX2";
    case native::CPUCapability::AVX512:
      return "AVX512";
#endif
    default:
      break;
  }
  return "";
}

static std::string used_cpu_capability() {
  // It is possible that we override the cpu_capability with
  // environment variable
  std::ostringstream ss;
  ss << "CPU capability usage: " << get_cpu_capability();
  return ss.str();
}

std::string show_config() {
  std::ostringstream ss;
  ss << "PyTorch built with:\n";

  // Reference:
  // https://blog.kowalczyk.info/article/j/guide-to-predefined-macros-in-c-compilers-gcc-clang-msvc-etc..html

#if defined(__GNUC__)
  {
    ss << "  - GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
  }
#endif

#if defined(__cplusplus)
  {
    ss << "  - C++ Version: " << __cplusplus << "\n";
  }
#endif

#if defined(__clang_major__)
  {
    ss << "  - clang " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
  }
#endif

#if defined(_MSC_VER)
  {
    ss << "  - MSVC " << _MSC_FULL_VER << "\n";
  }
#endif

#if AT_MKL_ENABLED()
  ss << "  - " << get_mkl_version() << "\n";
#endif

#if AT_ONEDNN_ENABLED()
  ss << "  - " << get_onednn_version() << "\n";
#endif

#ifdef _OPENMP
  ss << "  - " << get_openmp_version() << "\n";
#endif

#if AT_BUILD_WITH_LAPACK()
  // TODO: Actually record which one we actually picked
  ss << "  - LAPACK is enabled (usually provided by MKL)\n";
#endif

#if AT_NNPACK_ENABLED()
  // TODO: No version; c.f. https://github.com/Maratyszcza/NNPACK/issues/165
  ss << "  - NNPACK is enabled\n";
#endif

#ifdef CROSS_COMPILING_MACOSX
  ss << "  - Cross compiling on MacOSX\n";
#endif

  ss << "  - "<< used_cpu_capability() << "\n";

  if (hasCUDA()) {
    ss << detail::getCUDAHooks().showConfig();
  }

  if (hasMAIA()) {
    ss << detail::getMAIAHooks().showConfig();
  }

  if (hasXPU()) {
    ss << detail::getXPUHooks().showConfig();
  }

  ss << "  - Build settings: ";
  for (const auto& pair : caffe2::GetBuildOptions()) {
    if (!pair.second.empty()) {
      ss << pair.first << "=" << pair.second << ", ";
    }
  }
  ss << "\n";

  // TODO: do HIP
  // TODO: do XLA
  // TODO: do MPS

  return ss.str();
}

std::string get_cxx_flags() {
  #if defined(FBCODE_CAFFE2)
  TORCH_CHECK(
    false,
    "Buck does not populate the `CXX_FLAGS` field of Caffe2 build options. "
    "As a result, `get_cxx_flags` is OSS only."
  );
  #endif
  return caffe2::GetBuildOptions().at("CXX_FLAGS");
}

}
