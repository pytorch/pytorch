#include <torch/csrc/jit/codegen/cuda/interface.h>

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

// NOLINTNEXTLINE
C10_DEFINE_bool(
    torch_jit_nvfuser_singleton_fusion,
    false,
    "enable single node fusion for nvfuser");

// NOLINTNEXTLINE
C10_DEFINE_bool(
    torch_jit_nvfuser_horizontal_fusion,
    true,
    "enable horizontal fusion for nvfuser");

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class LoadingNvfuserLibrary {
 public:
#ifdef USE_CUDA
  LoadingNvfuserLibrary() {
    std::string library_name;
    if (const char* path = std::getenv("TORCH_NVFUSER_LIBRARY_PATH")) {
      library_name = path;
    }
#if defined(_WIN32)
    library_name += "nvfuser_codegen.dll";
#elif defined(__APPLE__)
    library_name += "libnvfuser_codegen.dylib";
#else
    library_name += "libnvfuser_codegen.so";
#endif
    try {
      // NOTE: we need to refactor this to a lazy load instead. We could end up
      // with double de-allocation with our python API loading the library.
      // Leaking the handle should solve the problem for now
      nvfuserLib_ = std::make_shared<at::DynamicLibrary>(
          library_name.c_str(), nullptr, true);
    } catch (const c10::DynamicLibraryError& e) {
#if defined(BUILD_NVFUSER) || !defined(NDEBUG)
      TORCH_WARN("Loading nvfuser library failed with: ", e.msg());
#endif
    }
  }

#endif // USE_CUDA
  std::shared_ptr<at::DynamicLibrary> nvfuserLib_;
};

static LoadingNvfuserLibrary loading_nvfuser_library_;

static std::atomic<bool> cuda_fusion_guard_mode{true};

// There are 3 sources of information on whether to enable nvfuser:
// 1. assigned value from setEnabled() - takes precendence if it has been set
// 2. value from environment variable - only used if setEnabled() is unset
// 3. default value - used if both 1 and 2 are unset.
//
// If 1 or 2 tries to enable nvfuser when it cannot be enabled (e.g. cuda not
// available), then an error will be thrown. The default will not error.
class NVFuserEnabler {
 private:
  c10::optional<bool> runtime_assigned_fuser_enabled_ = c10::nullopt;
  c10::once_flag enabled_check_flag_;
  std::mutex mutex_;

 public:
  bool nvfuserCanBeEnabled() {
#if defined(USE_ROCM) || defined(FBCODE_CAFFE2)
    return false;
#endif
    return at::globalContext().hasCUDA() && getExecutorMode() &&
        loading_nvfuser_library_.nvfuserLib_ != nullptr;
  }

 private:
  void assertFuserCanBeEnabled(bool is_enabled) {
    if (!is_enabled) {
      return;
    }
    TORCH_CHECK(
        nvfuserCanBeEnabled(),
        "Running CUDA fuser is only supported on CUDA builds.");
  }

  static c10::optional<bool> getFuserEnabledEnvVar() {
    static const char* enable_c_str = std::getenv("PYTORCH_JIT_ENABLE_NVFUSER");
    if (!enable_c_str) {
      return c10::nullopt;
    }
    std::string enable(enable_c_str);
    if (enable == "0" || enable == "OFF") {
      return false;
    }
    return true;
  }

  static c10::optional<bool> getCachedFuserEnabledEnvVar() {
    static c10::optional<bool> default_enabled = getFuserEnabledEnvVar();
    return default_enabled;
  }

  static bool getNNCNotNVFuser() {
    static const char* env_c_str =
        std::getenv("PYTORCH_JIT_USE_NNC_NOT_NVFUSER");
    if (!env_c_str) {
      return false;
    }
    std::string env(env_c_str);
    if (env == "1" || env == "ON") {
      return true;
    }
    return false;
  }

  static bool getCachedNNCNotNVFuser() {
    static bool force_disable = getNNCNotNVFuser();
    return force_disable;
  }

  bool isEnabledImpl() {
    // 0. opportunity to force disable NVFuser
    if (getCachedNNCNotNVFuser()) {
      return false;
    }
    c10::call_once(enabled_check_flag_, [&]() {
      // if environment variable is setting the value, we must
      if (!runtime_assigned_fuser_enabled_.has_value() &&
          getCachedFuserEnabledEnvVar().has_value()) {
        assertFuserCanBeEnabled(*getCachedFuserEnabledEnvVar());
      }
    });
    // 1. if user has explicitly assigned fuser value, that value takes
    // precedence.
    if (runtime_assigned_fuser_enabled_.has_value()) {
      return *runtime_assigned_fuser_enabled_;
    }
    // 2. next precedence is any value assigned by
    if (getCachedFuserEnabledEnvVar().has_value()) {
      return *getCachedFuserEnabledEnvVar();
    }
    // 3. default value
#if defined(USE_ROCM) || defined(FBCODE_CAFFE2)
    return false;
#else
    return nvfuserCanBeEnabled();
#endif
  }

 public:
  bool setEnabled(bool is_enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    assertFuserCanBeEnabled(is_enabled);
    bool old_value = isEnabledImpl();
    runtime_assigned_fuser_enabled_ = is_enabled;
    return old_value;
  }

  bool isEnabled() {
    std::lock_guard<std::mutex> lock(mutex_);
    return isEnabledImpl();
  }
};

static NVFuserEnabler nvfuser_enabler;

bool isEnabled() {
  return nvfuser_enabler.isEnabled();
}

bool setEnabled(bool is_enabled) {
  return nvfuser_enabler.setEnabled(is_enabled);
}

bool canBeEnabled() {
  return nvfuser_enabler.nvfuserCanBeEnabled();
}

bool getSingletonFusion() {
  return FLAGS_torch_jit_nvfuser_singleton_fusion;
}

bool setSingletonFusion(bool value) {
  bool old_value = FLAGS_torch_jit_nvfuser_singleton_fusion;
  FLAGS_torch_jit_nvfuser_singleton_fusion = value;
  return old_value;
}

bool getHorizontalFusion() {
  return FLAGS_torch_jit_nvfuser_horizontal_fusion;
}

bool setHorizontalFusion(bool value) {
  bool old_value = FLAGS_torch_jit_nvfuser_horizontal_fusion;
  FLAGS_torch_jit_nvfuser_horizontal_fusion = value;
  return old_value;
}

std::atomic<bool>& getCudaFusionGuardMode() {
  return cuda_fusion_guard_mode;
}

CudaFuserInterface* getFuserInterface() {
  static CudaFuserInterface fuser_interface_;
  return &fuser_interface_;
}

void compileFusionGroup(Node* fusion_node) {
  TORCH_CHECK(
      getFuserInterface()->fn_compile_n != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_compile_n(fusion_node);
}

void runFusionGroup(const Node* fusion_node, Stack& stack) {
  TORCH_CHECK(
      getFuserInterface()->fn_run_n_s != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_run_n_s(fusion_node, stack);
}

void fuseGraph(std::shared_ptr<Graph>& graph) {
  if (!isEnabled()) {
    return;
  }

  TORCH_CHECK(
      getFuserInterface()->fn_fuse_graph != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_fuse_graph(graph);
}

bool canFuseNode(const Node* node) {
  return getFuserInterface()->fn_can_fuse_n != nullptr &&
      getFuserInterface()->fn_can_fuse_n(node);
}

void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr) {
  if (getFuserInterface()->fn_insert_profile_inodes) {
    getFuserInterface()->fn_insert_profile_inodes(pr);
  }
}

bool profileNode(const Node* node) {
  return getFuserInterface()->fn_profile_n != nullptr &&
      getFuserInterface()->fn_profile_n(node);
}

bool skipNode(const std::string& symbol_str, bool flip) {
  return getFuserInterface()->fn_skip_n != nullptr &&
      getFuserInterface()->fn_skip_n(symbol_str, flip);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
