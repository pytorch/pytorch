
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <c10/util/string_view.h>

#include <cstdlib>
#include <iostream>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

auto parseDebugDumpOptions() {
  std::unordered_map<DebugDumpOption, bool> options_map = {
      {DebugDumpOption::FusionIr, false},
      {DebugDumpOption::FusionIrMath, false},
      {DebugDumpOption::FusionIrPresched, false},
      {DebugDumpOption::KernelIr, false},
      {DebugDumpOption::ComputeAtMap, false},
      {DebugDumpOption::CudaKernel, false},
      {DebugDumpOption::CudaFull, false},
      {DebugDumpOption::CudaToFile, false},
      {DebugDumpOption::DebugInfo, false},
      {DebugDumpOption::LaunchParam, false},
      {DebugDumpOption::FusionSegments, false},
      {DebugDumpOption::FusionSegmenterLog, false},
      {DebugDumpOption::FusionArgs, false},
      {DebugDumpOption::KernelArgs, false},
      {DebugDumpOption::EffectiveBandwidth, false},
      {DebugDumpOption::FusionSegmentsDrawing, false},
      {DebugDumpOption::PrintPtxasLog, false},
      {DebugDumpOption::BufferReuseInfo, false},
      {DebugDumpOption::SchedulerDebug, false},
      {DebugDumpOption::ParallelDimensions, false},
      {DebugDumpOption::Halo, false},
      {DebugDumpOption::PerfDebugVerbose, false},
      {DebugDumpOption::PythonDefinition, false},
      {DebugDumpOption::PythonFrontendDebug, false},
      {DebugDumpOption::TransformPropagator, false},
      {DebugDumpOption::InlinePropagator, false},
      {DebugDumpOption::Cubin, false},
      {DebugDumpOption::Ptx, false}};

  if (const char* dump_options = std::getenv("PYTORCH_NVFUSER_DUMP")) {
    c10::string_view options_view(dump_options);
    while (!options_view.empty()) {
      const auto end_pos = options_view.find_first_of(',');
      const auto token = options_view.substr(0, end_pos);
      if (token == "fusion_ir") {
        options_map[DebugDumpOption::FusionIr] = true;
      } else if (token == "fusion_ir_math") {
        options_map[DebugDumpOption::FusionIrMath] = true;
      } else if (token == "fusion_ir_presched") {
        options_map[DebugDumpOption::FusionIrPresched] = true;
      } else if (token == "kernel_ir") {
        options_map[DebugDumpOption::KernelIr] = true;
      } else if (token == "ca_map") {
        options_map[DebugDumpOption::ComputeAtMap] = true;
      } else if (token == "cuda_kernel") {
        options_map[DebugDumpOption::CudaKernel] = true;
      } else if (token == "cuda_full") {
        options_map[DebugDumpOption::CudaFull] = true;
      } else if (token == "cuda_to_file") {
        options_map[DebugDumpOption::CudaToFile] = true;
      } else if (token == "debug_info") {
        options_map[DebugDumpOption::DebugInfo] = true;
      } else if (token == "launch_param") {
        options_map[DebugDumpOption::LaunchParam] = true;
      } else if (token == "segmented_fusion") {
        options_map[DebugDumpOption::FusionSegments] = true;
      } else if (token == "segmenter_logging") {
        options_map[DebugDumpOption::FusionSegmenterLog] = true;
      } else if (token == "fusion_args") {
        options_map[DebugDumpOption::FusionArgs] = true;
      } else if (token == "kernel_args") {
        options_map[DebugDumpOption::KernelArgs] = true;
      } else if (token == "dump_eff_bandwidth") {
        options_map[DebugDumpOption::EffectiveBandwidth] = true;
      } else if (token == "draw_segmented_fusion") {
        options_map[DebugDumpOption::FusionSegmentsDrawing] = true;
      } else if (token == "ptxas_verbose") {
        options_map[DebugDumpOption::PrintPtxasLog] = true;
      } else if (token == "buffer_reuse_verbose") {
        options_map[DebugDumpOption::BufferReuseInfo] = true;
      } else if (token == "scheduler_params") {
        options_map[DebugDumpOption::SchedulerDebug] = true;
      } else if (token == "parallel_dimensions") {
        options_map[DebugDumpOption::ParallelDimensions] = true;
      } else if (token == "halo") {
        options_map[DebugDumpOption::Halo] = true;
      } else if (token == "perf_debug_verbose") {
        options_map[DebugDumpOption::PerfDebugVerbose] = true;
      } else if (token == "python_definition") {
        options_map[DebugDumpOption::PythonDefinition] = true;
      } else if (token == "python_frontend_debug") {
        options_map[DebugDumpOption::PythonFrontendDebug] = true;
      } else if (token == "transform_propagator") {
        options_map[DebugDumpOption::TransformPropagator] = true;
      } else if (token == "inline_propagator") {
        options_map[DebugDumpOption::InlinePropagator] = true;
      } else if (token == "cubin") {
        options_map[DebugDumpOption::Cubin] = true;
      } else if (token == "ptx") {
        options_map[DebugDumpOption::Ptx] = true;
      } else {
        TORCH_CHECK(
            false,
            "Invalid debug dump option: '",
            token,
            "'\nAvailable options:\n",
            "\tfusion_ir, fusion_ir_math, fusion_ir_presched, kernel_ir, ca_map,\n",
            "\tcuda_kernel, cuda_full, cuda_to_file, debug_info, launch_param,\n",
            "\tsegmented_fusion, fusion_args, kernel_args, dump_eff_bandwidth,\n",
            "\tdraw_segmented_fusion, scheduler_params, parallel_dimensions,\n",
            "\tbuffer_reuse_verbose, ptxas_verbose, halo, segmenter_logging,\n",
            "\tperf_debug_verbose, python_definition, python_frontend_debug,\n",
            "\ttransform_propagator, inline_propagator, cubin, ptx\n");
      }
      options_view = (end_pos != c10::string_view::npos)
          ? options_view.substr(end_pos + 1)
          : "";
    }
  }

  return options_map;
}

auto parseDisableOptions() {
  std::unordered_map<DisableOption, bool> options_map = {
      {DisableOption::ArchCheck, false},
      {DisableOption::Fallback, false},
      {DisableOption::Fma, false},
      {DisableOption::IndexHoist, false},
      {DisableOption::Nvtx, false},
      {DisableOption::PredicateElimination, false}};

  if (const char* dump_options = std::getenv("PYTORCH_NVFUSER_DISABLE")) {
    c10::string_view options_view(dump_options);
    while (!options_view.empty()) {
      const auto end_pos = options_view.find_first_of(',');
      const auto token = options_view.substr(0, end_pos);
      if (token == "arch_check") {
        options_map[DisableOption::ArchCheck] = true;
      } else if (token == "fallback") {
        options_map[DisableOption::Fallback] = true;
      } else if (token == "fma") {
        TORCH_WARN(
            "fmad is disabled for nvrtc, which could negatively affect performance. Try removing `fma` from env variable PYTORCH_NVFUSER_DISABLE for optimal performance.");
        options_map[DisableOption::Fma] = true;
      } else if (token == "index_hoist") {
        options_map[DisableOption::IndexHoist] = true;
      } else if (token == "nvtx") {
        options_map[DisableOption::Nvtx] = true;
      } else if (token == "predicate_elimination") {
        options_map[DisableOption::PredicateElimination] = true;
      } else {
        TORCH_CHECK(
            false,
            "Invalid disable option: '",
            token,
            "'\nAvailable options:\n",
            "\tarch_check, fallback, fma, index_hoist, nvtx, predicate_elimination\n");
      }
      options_view = (end_pos != c10::string_view::npos)
          ? options_view.substr(end_pos + 1)
          : "";
    }
  }

  return options_map;
}

auto parseEnableOptions() {
  std::unordered_map<EnableOption, bool> options_map = {
      {EnableOption::Complex, false},
      {EnableOption::KernelProfile, false},
      {EnableOption::LinearDecomposition, false},
      {EnableOption::ConvDecomposition, false},
      {EnableOption::TransposeScheduler, false}};

  if (const char* dump_options = std::getenv("PYTORCH_NVFUSER_ENABLE")) {
    c10::string_view options_view(dump_options);
    while (!options_view.empty()) {
      const auto end_pos = options_view.find_first_of(',');
      const auto token = options_view.substr(0, end_pos);
      if (token == "complex") {
        options_map[EnableOption::Complex] = true;
      } else if (token == "kernel_profile") {
        options_map[EnableOption::KernelProfile] = true;
      } else if (token == "linear_decomposition") {
        options_map[EnableOption::LinearDecomposition] = true;
      } else if (token == "conv_decomposition") {
        options_map[EnableOption::ConvDecomposition] = true;
      } else if (token == "transpose_scheduler") {
        options_map[EnableOption::TransposeScheduler] = true;
      } else {
        TORCH_CHECK(
            false,
            "Invalid enable option: '",
            token,
            "'\nAvailable options:\n",
            "\tcomplex, kernel_profile, linear_decomposition,",
            "conv_decomposition, transpose_scheduler");
      }
      options_view = (end_pos != c10::string_view::npos)
          ? options_view.substr(end_pos + 1)
          : "";
    }
  }

  return options_map;
}

} // namespace

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
void debugPrint(const c10::TensorTypePtr& type) {
  std::stringstream sizes_s;
  if (auto sizes = type->symbolic_sizes().sizes()) {
    for (const auto& shape_symbol : *sizes) {
      if (shape_symbol.is_static()) {
        sizes_s << shape_symbol.static_size() << ", ";
      } else {
        sizes_s << "s(" << *reinterpret_cast<const int64_t*>(&shape_symbol)
                << "), ";
      }
    }
  } else {
    sizes_s << "no size available";
  }
  std::cout << "sizes:" << sizes_s.str() << std::endl;
  if (const auto& stride_properties = type->stride_properties().sizes()) {
    std::stringstream stride_s;
    std::stringstream index_s;
    std::stringstream contig_s;

    for (const auto& stride_property : *stride_properties) {
      if (stride_property.has_value() && stride_property->stride_.has_value()) {
        stride_s << *stride_property->stride_ << ", ";
      } else {
        stride_s << "?, ";
      }
      if (stride_property.has_value() &&
          stride_property->stride_index_.has_value()) {
        index_s << *stride_property->stride_index_ << ", ";
      } else {
        index_s << "?, ";
      }
      if (stride_property.has_value() &&
          stride_property->contiguous_.has_value()) {
        contig_s << *stride_property->contiguous_ << ", ";
      } else {
        contig_s << "?, ";
      }
    }
    std::cout << "stride: " << stride_s.str() << std::endl;
    std::cout << "stride index: " << index_s.str() << std::endl;
    std::cout << "contiguous: " << contig_s.str() << std::endl;
  } else {
    std::cout << "no stride properties available" << std::endl;
  }
}
#pragma clang diagnostic pop

bool is_zero_dim_tensor(const std::shared_ptr<c10::TensorType>& tensor_type) {
  return tensor_type && tensor_type->dim().has_value() &&
      tensor_type->dim().value() == 0;
}

bool is_zero_sized_tensor(const std::shared_ptr<c10::TensorType>& tensor_type) {
  auto opt_sizes = tensor_type->sizes().concrete_sizes();
  if (opt_sizes.has_value()) {
    auto sizes = opt_sizes.value();
    for (const auto& size : sizes) {
      if (size == 0) {
        return true;
      }
    }
  }
  return false;
}

bool is_cpu_scalar(const at::Tensor& tensor) {
  return tensor.device().is_cpu() && tensor.numel() == 1 && tensor.dim() == 0;
}

bool is_cpu_scalar(const c10::TensorType& tensor_type) {
  auto opt_device = tensor_type.device();
  auto opt_dim = tensor_type.dim();
  auto opt_numel = tensor_type.numel();
  return opt_device.has_value() && opt_device->is_cpu() &&
      opt_dim.has_value() && opt_numel.has_value() && opt_dim.value() == 0 &&
      opt_numel.value() == 1;
}

// Check device of TensorType in all inputs ensure all tensors are on cuda
// devices.
// return common device index (or -1 if device differs).
int getCommonDeviceCUDA(const at::ArrayRef<IValue>& inputs) {
  int index = -1;
  for (const auto& input : inputs) {
    if (!input.isTensor()) {
      continue;
    }
    const auto& device = input.toTensor().device();
    // skip cpu scalar tensor as they'll be promoted to scalar later
    if (device.is_cpu() && is_cpu_scalar(input.toTensor())) {
      continue;
    }
    TORCH_CHECK(device.is_cuda(), "nvfuser only supports cuda device");
    auto cur_index = device.index();
    if (index != -1 && index != cur_index) {
      return -1;
    }
    index = (int)cur_index; // NOLINT
  }
  return index;
}

KernelIndexMode collectIndexMode(const at::ArrayRef<at::IValue>& inputs) {
  // Save 1 more bit besides the sign bit to be conservative
  constexpr int64_t most_positive_int32_index =
      std::numeric_limits<int>::max() / 2;
  constexpr int64_t most_negative_int32_index =
      std::numeric_limits<int>::min() / 2;

  // Check all runtime inputs, and if any one of
  //  the input's index exceeds max_int32 will
  //  fall back to int64 indexing
  for (auto ivalue_input : inputs) {
    if (ivalue_input.isTensor()) {
      auto tensor_input = ivalue_input.toTensor();
      int64_t tensor_most_positive_index = 0;
      int64_t tensor_most_negative_index = 0;
      for (auto dim_i = 0; dim_i < tensor_input.ndimension(); dim_i++) {
        // Ignore broadcast dimensions
        if (tensor_input.size(dim_i) > 1) {
          // accumulate based on the sign of stride
          if (tensor_input.stride(dim_i) > 0) {
            // Acuumulate positive stride
            tensor_most_positive_index +=
                (tensor_input.size(dim_i) - 1) * tensor_input.stride(dim_i);
          } else {
            // Acuumulate negative stride
            tensor_most_negative_index +=
                (tensor_input.size(dim_i) - 1) * tensor_input.stride(dim_i);
          }
        }
      }

      // Fall back to int64 if it can be either too positive
      //  or too negative.
      if (tensor_most_positive_index > most_positive_int32_index ||
          tensor_most_negative_index < most_negative_int32_index) {
        return KernelIndexMode::INT64;
      }
    }
  }
  // return index mode as int32
  return KernelIndexMode::INT32;
}

bool isDebugDumpEnabled(DebugDumpOption option) {
  const static auto dump_options = parseDebugDumpOptions();
  return dump_options.at(option);
}

bool isOptionDisabled(DisableOption option) {
  const static auto options = parseDisableOptions();
  return options.at(option);
}

bool isOptionEnabled(EnableOption option) {
  const static auto options = parseEnableOptions();
  return options.at(option);
}

bool useFallback() {
  // Keep this env var for compatibility
  const char* disable_fb_env = getenv("PYTORCH_NVFUSER_DISABLE_FALLBACK");
  bool fallback_disabled = disable_fb_env ? atoi(disable_fb_env) : false;
  fallback_disabled =
      fallback_disabled || isOptionDisabled(DisableOption::Fallback);

  return !fallback_disabled;
}

std::vector<int64_t> getTensorSizes(TensorTypePtr const& tensor_type) {
  TORCH_INTERNAL_ASSERT(tensor_type != nullptr, "Input must be a Tensor.");
  auto optional_sizes = tensor_type->sizes().concrete_sizes();
  TORCH_INTERNAL_ASSERT(
      optional_sizes.has_value(), "Missing size information for the tensor.");
  return optional_sizes.value();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
