
#include <torch/csrc/jit/codegen/cuda/executor.h>

#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/llvm_jit_strings.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>

#include <fstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int FusionExecutor::fusion_id_counter_ = 0; // NOLINT

namespace {

static const char* defineIndexMode(KernelIndexMode index_mode) {
  switch (index_mode) {
    case KernelIndexMode::INT32:
      return "typedef int nvfuser_index_t;\n";
    case KernelIndexMode::INT64:
      return "typedef int64_t nvfuser_index_t;\n";
    default:
      break;
  }

  TORCH_INTERNAL_ASSERT(false, "unknow indexing mode");
  return "";
}

static const char* defineIntegerTypes() {
  return R"(
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int int16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
)";
}

static const std::string& defineComplexTypes() {
  static std::string result = std::string(R"ESCAPE(
#define POS_INFINITY __int_as_float(0x7f800000)
#define INFINITY POS_INFINITY
#define NEG_INFINITY __int_as_float(0xff800000)
#define NAN __int_as_float(0x7fffffff)
)ESCAPE") +
      at::cuda::get_traits_string() + at::cuda::get_complex_body_string() +
      at::cuda::get_cmath_string() + at::cuda::get_complex_math_string();
  return result;
}

} // namespace

std::string FusionExecutor::getStructuredCode(const std::string& kernel) {
  // generating cuda code;
  std::string code = "";
#ifdef __HIP_PLATFORM_HCC__
#if ROCM_VERSION < 40200
  code += std::string("#include <hip/hip_runtime.h>\n") +
      std::string("#include <hip/hip_bf16.h>\n") +
      std::string("#include <hip/hip_fp16.h>\n");
#endif
#endif
  code += std::string("namespace ") + FusionExecutor::kernelNamespace() +
      " {\n" + defineIntegerTypes() + defineIndexMode(options_.index_mode) +
      defineComplexTypes() + executor_utils::kernelPreamble() + kernel + "}\n";

  if (isDebugDumpEnabled(DebugDumpOption::CudaKernel)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << kernel << "\n======================================\n\n";
  } else if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n======= Codegen output for kernel: " << kernelName()
              << " =======\n\n"
              << code << "\n======================================\n\n";
  } else if (isDebugDumpEnabled(DebugDumpOption::CudaToFile)) {
    std::stringstream file_name;
    file_name << "__tmp_kernel" << fusion_id_ << ".cu";
    std::cout << "PRINTING: " << file_name.str() << std::endl;
    std::ofstream out(file_name.str());
    out << code << std::endl;
    out.close();
  }

  return code;
}

// TODO: come up with a more user friendly interface
void FusionExecutor::debugCompileFusionFromStr(
    Fusion* fusion,
    const std::string& code,
    const std::string& name,
    int id,
    CompileOptions options) {
  options_ = options;

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  if (isDebugDumpEnabled(DebugDumpOption::CudaFull)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "======================================\n"
              << std::endl;
  }

  lowered_ = std::make_unique<GpuLower>(fusion);
  const auto kernel = lowered_->kernel();
  fusion_ = lowered_->kernel();

  fusion_id_ = id;
  setUsedTVs();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  const auto& kernel_summary = kernel->summary();

  if (!kernel_summary.static_smem_allocations.empty()) {
    kir::ExpressionEvaluator static_evaluator;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_device_smem,
        "The static shared memory allocation is larger than available memory.");
  }

  compiled_kernel_ = executor_utils::nvrtcCompile(code, name, fusion_id_);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "assign a fusion_id_ <= 0 is not accepted.");
}

void FusionExecutor::compileFusion(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    const LaunchParams& launch_constraints,
    CompileOptions options) {
  FUSER_PERF_SCOPE("compileFusion");

  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs()) {
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");
  }

  if (isDebugDumpEnabled(DebugDumpOption::FusionIr)) {
    fusion->print();
  } else if (isDebugDumpEnabled(DebugDumpOption::FusionIrMath)) {
    fusion->printMath();
  }

  options_ = options;
  c10::DeviceGuard dg(options_.device);

  TORCH_INTERNAL_ASSERT(
      options_.device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  auto properties = at::cuda::getDeviceProperties(options_.device.index());
  max_device_smem = properties->sharedMemPerBlock;
  warp_size_ = properties->warpSize;

  lowered_ = std::make_unique<GpuLower>(
      fusion,
      options_.index_mode == KernelIndexMode::INT64 ? DataType::Int
                                                    : DataType::Int32);
  const auto kernel = lowered_->kernel();
  fusion_ = lowered_->kernel()->as<Fusion>();

  fusion_id_ = ++fusion_id_counter_;
  setUsedTVs();

  if (isDebugDumpEnabled(DebugDumpOption::KernelIr)) {
    kernel->print();
  }

  const auto kernel_code = codegen::generateCudaKernel(kernel, kernelName());
  const auto structured_code = getStructuredCode(kernel_code);

  const auto& kernel_summary = kernel->summary();

  if (!kernel_summary.static_smem_allocations.empty()) {
    kir::ExpressionEvaluator static_evaluator;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const auto static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_device_smem,
        "The static shared memory allocation is larger than available memory.");
  }

  if (kernel_summary.has_dynamic_local_memory_allocations) {
    std::stringstream ss;
    ss << "Allocations must be based on constant integers for local memory. However, found: ";
    for (auto alloc : kernel_summary.dynamic_lmem_allocations) {
      ss << alloc->buffer()->toString() << ", ";
    }
    ss << " have dynamic allocations but are placed in local memory.";
    TORCH_INTERNAL_ASSERT(false, ss.str());
  }

  // TODO: pass block_size here;
  c10::optional<int> block_size = c10::nullopt;
  if (!inputs.empty()) {
    auto expr_eval = executor_utils::bindKernelInputs(inputs, kernel);
    auto launch_params =
        computeLaunchParams(launch_constraints, expr_eval, warp_size_);
    block_size = launch_params.nThreads();
    TORCH_INTERNAL_ASSERT(
        block_size > 0, "launch param inferred block size < 0");
  }

  block_size_high_water_mark =
      block_size.has_value() ? block_size.value() : block_size_high_water_mark;
  compiled_kernel_ = executor_utils::nvrtcCompile(
      structured_code,
      (kernelNamespace() + "::" + kernelName()).c_str(),
      fusion_id_,
      block_size);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "failed to assign a fusion_id_ after compilation.");
}

namespace {

at::Tensor inferAndAlloc(
    const TensorView* tv,
    const std::vector<Val*>& sizes,
    kir::ExpressionEvaluator& expr_eval,
    const CompileOptions& options,
    bool zero_init = false) {
  FUSER_PERF_SCOPE("inferAndAlloc");

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<int64_t> inferred_sizes;

  for (const auto size : sizes) {
    const auto inferred_val = expr_eval.evaluate(size);
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Could not launch kernel as program could not infer ",
        size->toString(),
        "(",
        size->name(),
        ") for the buffer ",
        tv->toString());
    inferred_sizes.push_back(inferred_val.value());
  }

  const auto at_type = data_type_to_aten(tv->dtype());

  if (zero_init) {
    const auto tensor_options =
        at::TensorOptions().dtype(at_type).device(options.device);
    c10::IntArrayRef isizes(inferred_sizes);
    return at::zeros(isizes, tensor_options);
  } else {
    c10::IntArrayRef isizes(inferred_sizes);
    // Non Variable type guard for empty_cuda call
    at::AutoDispatchBelowADInplaceOrView non_variable_type_mode;
    return at::native::empty_cuda(
        isizes, at_type, c10::nullopt, options.device, c10::nullopt);
  }
}

at::Tensor inferAndAllocOutput(
    const TensorView* tv,
    kir::ExpressionEvaluator& expr_eval,
    const CompileOptions& options,
    bool zero_init = false) {
  const auto domain = tv->domain();
  const auto maybe_rfactor_domain = domain->hasRFactor()
      ? domain->getRFactorDomain()
      : domain->getRootDomain();

  std::vector<Val*> sizes;

  for (const auto id : maybe_rfactor_domain) {
    if (id->isReduction() || id->isStride() ||
        id->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    }
    sizes.push_back(id->extent());
  }

  return inferAndAlloc(tv, sizes, expr_eval, options, zero_init);
}

} // namespace

uint64_t FusionExecutor::computeSharedMemory(
    kir::ExpressionEvaluator& expr_eval,
    const std::vector<const kir::Allocate*>& buffers,
    bool align_padding,
    uint64_t total) {
  FUSER_PERF_SCOPE("computeSharedMemory");
  for (auto smem_alloc : buffers) {
    // If this buffer aliases another buffer,
    // then do not allocate memory for this buffer.
    if (smem_alloc->alias() == nullptr) {
      const auto inferred_val = expr_eval.evaluate(smem_alloc->size());
      if (inferred_val.has_value()) {
        const uint64_t data_size = dataTypeSize(smem_alloc->buffer()->dtype());
        // Add padding to align dynamic shared memory
        if (align_padding) {
          total = ceilDiv(total, data_size) * data_size;
        }
        total += inferred_val.value() * data_size;
      } else {
        TORCH_INTERNAL_ASSERT(
            false,
            "Failed to evaluate the size ",
            smem_alloc->size(),
            " of shared memory buffer - T",
            smem_alloc->buffer()->name());
      }
    }
  }
  return total;
}

LaunchParams FusionExecutor::computeLaunchParams(
    const LaunchParams& launch_constraints,
    kir::ExpressionEvaluator& expr_eval,
    const int warp_size) {
  FUSER_PERF_SCOPE("FusionExecutor::ComputeLaunchParams");
  TORCH_INTERNAL_ASSERT(warp_size > 0, "WARP_SIZE should be larger than 0");

  LaunchParams launch_params;

  auto data_cache = compileTimeDataCache();

  auto lower = lowered_.get();
  auto& used_tvs = getUsedTVs();
  auto parallel_binding_ids_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::ParallelBindingIterDomains>(
          data_cache, [&used_tvs, &lower]() {
            return std::make_unique<std::vector<IterDomain*>>(
                executor_utils::getParallelBindingsIterDomains(
                    lower, used_tvs));
          });
  auto& parallel_binding_ids = parallel_binding_ids_entry.get();

  auto parallel_iter_extent_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::ParallelIterExtentMap>(
          data_cache, [&parallel_binding_ids]() {
            return executor_utils::getParallelIterExtents(parallel_binding_ids);
          });
  auto& parallel_iter_extents = parallel_iter_extent_entry.get();

  auto simplified_parallel_iter_extent_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::SimplifiedParallelIterExtentMap>(
          data_cache, [&parallel_binding_ids, &lower]() {
            return executor_utils::getSimplifiedParallelIterExtents(
                lower, parallel_binding_ids);
          });
  auto& simplified_parallel_iter_extents =
      simplified_parallel_iter_extent_entry.get();

  auto warp_padded_parallel_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::WarpPaddedParallelExtents>(
          data_cache, [&parallel_binding_ids, &lower]() {
            return executor_utils::getWarpPaddedExtentsInfo(
                lower->kernel(), parallel_binding_ids);
          });
  auto& warp_padded_extent_set =
      warp_padded_parallel_entry.get().warp_padded_extent_set;
  auto& warp_padded_constant =
      warp_padded_parallel_entry.get().warp_padded_constant;

  // TODO: Need to redesign this part a bit to
  //   find the right place to trigger evaluate
  if (expr_eval.precomputedIntegers()) {
    expr_eval.precomputedIntegers()->bindParallelExtents(
        parallel_iter_extents, launch_constraints);
    expr_eval.precomputedIntegers()->evaluate();
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be inferred the inference matches what was set.
  for (auto& entry : parallel_iter_extents) {
    auto p_type = entry.first;
    if (launch_constraints.hasDim(p_type)) {
      auto parallel_extents = entry.second;
      for (auto extent : parallel_extents) {
        auto inferred_val = expr_eval.evaluate(extent);
        if (inferred_val.has_value()) {
          // This value could have been inferred, make sure it was set right.
          bool valid =
              inferred_val.value() == launch_constraints.getDim(p_type) ||
              launch_constraints.getRawVal(p_type) == -1;
          if (!useFallback() && !valid) {
            TORCH_WARN_ONCE(
                "Cannot validate parallelization scheme, "
                "this may be due to mixed broadcast axes that are parallelized.");
          }
        } else if (!expr_eval.precomputedIntegers()) {
          expr_eval.bind(extent, launch_constraints.getDim(p_type));
        }
        if (!launch_params.hasDim(p_type)) {
          // Bind the launch constraint into our evaluation context
          launch_params.bind(launch_constraints.getDim(p_type), p_type);
          // Makes sure the p-types bound to evaluators are the
          //  final values that will become the actual launch
          //  param size to ensure accurate smem buffer size
          //  computation.
          expr_eval.bind(p_type, launch_constraints.getDim(p_type));
        }
      }
    }
  }

  // Run through the rest of the parallel IterDomains and infer their size
  for (auto& entry : simplified_parallel_iter_extents) {
    FUSER_PERF_SCOPE("FusionExecutor::ParallelBindingResolution");
    auto p_type = entry.first;
    auto parallel_extents = entry.second;
    // Select the maxmimum value out of all the parallel extents
    int64_t maximum_value = std::numeric_limits<int64_t>::min();
    for (auto extent : parallel_extents) {
      auto val = expr_eval.evaluate(extent);
      TORCH_INTERNAL_ASSERT(
          val.has_value(),
          "Tried to evaluate the extent, ",
          extent->toInlineString(),
          " for the ptype: ",
          p_type,
          " to set launch bounds but could not.");

      // apply padding to the extent if needed
      if (warp_padded_extent_set.count(extent)) {
        // Check if the extent has const value
        auto padded_constant_it = warp_padded_constant.find(extent);

        if (padded_constant_it != warp_padded_constant.end()) {
          // If already specified padded to constant, need to check
          //  runtime value not over the constant bound
          TORCH_INTERNAL_ASSERT(*val <= padded_constant_it->second);
          *val = padded_constant_it->second;
        } else {
          // If no specified constant, pad to the smallest multiple of warp
          //  above the value.
          auto padded_number_of_warps = (*val + warp_size - 1) / warp_size;
          *val = warp_size * padded_number_of_warps;
        }
        TORCH_INTERNAL_ASSERT(
            *val <= 1024, "padded dimension larger than max block size");
      }
      maximum_value = std::max(maximum_value, *val);
    }
    // Protect for size-0 tensors, they still have a value so would prefer to
    // bind nothing than 0
    if (maximum_value > 0) {
      expr_eval.bind(p_type, maximum_value);
      launch_params.bind(maximum_value, p_type);
    }
  }

  // Re-run the integer machine with all
  //  the thread sizes now determined.
  if (expr_eval.precomputedIntegers()) {
    expr_eval.precomputedIntegers()->evaluate();
  }

  const auto kernel = lowered_->kernel();
  const auto& kernel_summary = kernel->summary();

  // Calculate Dynamic Shared Memory Size
  // Add workspace for reduction and broadcast
  uint64_t reduction_broadcast_workspace = 0;
  const bool has_workspace = kernel_summary.has_block_reductions ||
      kernel_summary.has_grid_reductions ||
      kernel_summary.has_block_broadcasts || kernel_summary.has_grid_broadcasts;
  if (has_workspace &&
      kernel_summary.largest_smem_data_type != DataType::Null) {
    // Not using nThreads here since it does not handle uninitialized value

    // TODO: here is an optimization opportunity since welford uses int64_t for
    // N while the data type is not neccessarily double. But it may need more
    // work on the alignment
    const int welford_factor =
        kernel_summary.has_block_welford || kernel_summary.has_grid_welford ? 3
                                                                            : 1;
    reduction_broadcast_workspace =
        dataTypeSize(kernel_summary.largest_smem_data_type) * welford_factor *
        launch_params.bdimx() * launch_params.bdimy() * launch_params.bdimz();
  }

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const uint64_t dynamic_smem_size = computeSharedMemory(
      expr_eval,
      kernel_summary.dynamic_smem_allocations,
      true,
      reduction_broadcast_workspace);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const uint64_t static_smem_size =
      computeSharedMemory(expr_eval, kernel_summary.static_smem_allocations);

  TORCH_INTERNAL_ASSERT(
      (dynamic_smem_size + static_smem_size) < max_device_smem,
      "The total shared memory allocation is larger than available memory.",
      " Dynamic size: ",
      dynamic_smem_size,
      ". Static size: ",
      static_smem_size,
      ". Available size: ",
      max_device_smem);
  launch_params.setSmem(dynamic_smem_size);

  return launch_params;
}

FusionExecutor::GlobalBuffers FusionExecutor::allocGlobalVals(
    kir::ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("FusionExecutor::AllocGlobalVals");
  GlobalBuffers global_buffers;
  const auto kernel = lowered_->kernel();
  const auto& kernel_summary = kernel->summary();
  for (auto alloc : kernel_summary.global_allocations) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->isA<TensorView>(),
        "Cannot allocate global buffers that are not tensors.");
    auto tv = alloc->buffer()->as<TensorView>();
    if (tv->isFusionOutput()) {
      continue;
    }
    if (alloc->zeroInit()) {
      global_buffers.buffers.push_back(
          inferAndAlloc(tv, alloc->shape(), expr_eval, options_, true));
      global_buffers.zero_init.push_back(true);
    } else {
      global_buffers.buffers.push_back(
          inferAndAlloc(tv, alloc->shape(), expr_eval, options_, false));
      global_buffers.zero_init.push_back(false);
    }
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::allocOutputs(
    const at::ArrayRef<IValue>& inputs,
    kir::ExpressionEvaluator& expr_eval,
    const std::unordered_set<int>& alias_indices) {
  FUSER_PERF_SCOPE("FusionExecutor::AllocOutputs");
  const auto kernel = lowered_->kernel();
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<at::Tensor> outputs;
  for (const auto out_i : c10::irange(kernel->outputs().size())) {
    // Dummy output.
    if (kernel->outputs()[out_i]->isFusionInput()) {
      for (auto inp_i : c10::irange(kernel->inputs().size())) {
        if (kernel->inputs()[inp_i] == kernel->outputs()[out_i]) {
          TORCH_INTERNAL_ASSERT(
              inp_i < inputs.size(),
              "Issue with an input showing up as output, couldn't find input.");
          TORCH_INTERNAL_ASSERT(
              inputs[inp_i].isTensor(),
              "Cannot register a scalar as an output in a fusion.");
          outputs.push_back(inputs[inp_i].toTensor());
          break;
        }
      }
    } else {
      TORCH_INTERNAL_ASSERT(
          kernel->outputs()[out_i]->isA<TensorView>(),
          "Cannot allocate outputs that are not tensors.");
      auto output = kernel->outputs()[out_i]->as<TensorView>();
      if (alias_indices.count(out_i) == 0) {
        outputs.push_back(
            inferAndAllocOutput(output, expr_eval, options_, false));
      } else {
        // aliasing to inputs, no need to allocate real output
        outputs.push_back(
            inferAndAlloc(output, {}, expr_eval, options_, false));
      }
    }
  }
  return outputs;
}

void FusionExecutor::setUsedTVs() {
  auto used_vals = fusion_->usedMathVals();
  auto used_tvs = ir_utils::filterByType<TensorView>(used_vals);
  used_tvs_.clear();

  for (auto tv : used_tvs)
    used_tvs_.push_back(tv);
}

std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    const LaunchParams& launch_constraints,
    const c10::optional<size_t>& opt_code) {
  FUSER_PERF_SCOPE("FusionExecutor::RunFusion");
  TORCH_INTERNAL_ASSERT(compiled());
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "Cannot run fusion, it was not compiled.");
  TORCH_INTERNAL_ASSERT(
      !opt_code.has_value() || outputs.empty(),
      "short cut input cache is not compatible with pre-allocated output");

  ExecutorEntry* executor_entry = nullptr;
  if (opt_code.has_value()) {
    executor_entry = &executor_entry_lookup_[*opt_code];
  }

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();
  executor_utils::initializeCudaContext();
  TORCH_INTERNAL_ASSERT(lowered_);
  LaunchParams launch_params;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<at::Tensor> allocated_outputs = outputs;
  GlobalBuffers global_buffers;
  uint64_t rand_offset = 0;

  if (executor_entry && executor_entry->init && !disable_parameter_cache_) {
    {
      // context manager to disable auto grad for `empty_cuda` calls later
      at::AutoDispatchBelowADInplaceOrView non_variable_type_mode;
      // take the short-cut for launch if we see a recorded input set again
      launch_params = executor_entry->launch_params;
      // only allocate outputs when not given
      if (outputs.empty()) {
        FUSER_PERF_SCOPE("ExecutorRunFusion::OutputAlloc");
        for (const auto i : c10::irange(executor_entry->output_sizes.size())) {
          allocated_outputs.push_back(at::native::empty_cuda(
              executor_entry->output_sizes[i],
              executor_entry->output_types[i],
              c10::nullopt,
              options_.device,
              c10::nullopt));
        }
        for (const auto& entry : executor_entry->io_alias_indices) {
          TORCH_INTERNAL_ASSERT(
              inputs[entry.second].isTensor(), "alias io only supports tensor");
          allocated_outputs[entry.first] = inputs[entry.second].toTensor();
        }
      } else {
        TORCH_INTERNAL_ASSERT(
            outputs.size() == fusion_->outputs().size(),
            __func__,
            " provided number of outputs does match fusion output");
      }
      {
        FUSER_PERF_SCOPE("ExecutorRunFusion::IntermediateBufferAlloc");
        for (const auto i : c10::irange(executor_entry->buffer_sizes.size())) {
          if (executor_entry->buffer_zero_init[i]) {
            global_buffers.buffers.push_back(at::zeros(
                executor_entry->buffer_sizes[i],
                at::TensorOptions()
                    .dtype(executor_entry->buffer_types[i])
                    .device(options_.device)));
          } else {
            global_buffers.buffers.push_back(at::native::empty_cuda(
                executor_entry->buffer_sizes[i],
                executor_entry->buffer_types[i],
                c10::nullopt,
                options_.device,
                c10::nullopt));
          }
        }
      }
    }
    rand_offset = executor_entry->rand_offset;
  } else {
    FUSER_PERF_SCOPE("ExecutorRunFusion::ValidateAndInitialize");
    // code path to take when either:
    //   1. no opt_code is provided or
    //   2. `executor_entry` is not initialized
    executor_utils::validateKernelInputs(fusion_, inputs, options_.device);

    if (!evaluator_precomputed_integers_) {
      evaluator_precomputed_integers_ =
          std::make_unique<KernelPrecomputedIntegers>(lowered_->kernel());
    }

    kir::ExpressionEvaluator expr_eval;
    evaluator_precomputed_integers_->bindKernelInputs(
        lowered_->kernel(), inputs);
    expr_eval.precomputedIntegers() = evaluator_precomputed_integers_.get();

    launch_params =
        computeLaunchParams(launch_constraints, expr_eval, warp_size_);

    // Recompile the kernel if the number of threads in the block has increased
    if (launch_params.nThreads() > block_size_high_water_mark) {
      const auto kernel = lowered_->kernel();
      const auto kernel_code =
          codegen::generateCudaKernel(kernel, kernelName());
      const auto structured_code = getStructuredCode(kernel_code);
      block_size_high_water_mark = launch_params.nThreads();
      compiled_kernel_ = executor_utils::nvrtcCompile(
          structured_code,
          (kernelNamespace() + "::" + kernelName()).c_str(),
          fusion_id_,
          block_size_high_water_mark);
    }

    if (kernel()->summary().has_cooperative_grid_reduction) {
#ifndef __HIP_PLATFORM_HCC__
      int num_blocks_per_SM = -1;
      at::globalContext().getNVRTC().cuOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_SM,
          compiled_kernel_.function,
          (int)(launch_params.bdimx() * launch_params.bdimy() * launch_params.bdimz()),
          (size_t)launch_params.smem());

      TORCH_INTERNAL_ASSERT(
          (int64_t)(
              num_blocks_per_SM *
              at::cuda::getDeviceProperties(options_.device.index())
                  ->multiProcessorCount) >= launch_params.gdimx() *
                  launch_params.gdimy() * launch_params.gdimz(),
          "Wanted to launch a cooperative kernel, however the number of blocks is greater than ",
          "what can be resident on the GPU at once. Need: ",
          launch_params.gdimx() * launch_params.gdimy() * launch_params.gdimz(),
          " but limited to ",
          num_blocks_per_SM,
          " * ",
          at::cuda::getDeviceProperties(options_.device.index())
              ->multiProcessorCount);
#else
      TORCH_INTERNAL_ASSERT(
          false, "Cross grid communication not supported with HIP.");
#endif
    }

    executor_utils::validateVectorizedTensors(
        lowered_.get()->kernel(),
        inputs,
        outputs,
        compileTimeDataCache(),
        expr_eval);

    auto alias_indices_entry =
        executor_utils::caching::ExecutorCompileTimeEntry<
            executor_utils::caching::InputAliasIndices>(
            compileTimeDataCache(), [&]() {
              return std::make_unique<std::vector<std::pair<int, int>>>(
                  fusion_->getInputAliasIndices());
            });

    auto& alias_indices = alias_indices_entry.get();

    // ditch pre-allocated outputs if the number doesn't match.
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (outputs.empty()) {
      auto output_alias_indices_entry =
          executor_utils::caching::ExecutorCompileTimeEntry<
              executor_utils::caching::OutputAliasIndices>(
              compileTimeDataCache(), [&]() {
                return std::make_unique<std::unordered_set<int>>(
                    fusion_->getOutputAliasIndices());
              });

      auto& output_alias_indices = output_alias_indices_entry.get();

      allocated_outputs = allocOutputs(inputs, expr_eval, output_alias_indices);

      for (const auto& entry : alias_indices) {
        TORCH_INTERNAL_ASSERT(
            inputs[entry.second].isTensor(), "alias io only supports tensor");
        allocated_outputs[entry.first] = inputs[entry.second].toTensor();
      }
    } else {
      // TODO: Update this as well;
      executor_utils::validateKernelOutputs(
          fusion_, allocated_outputs, options_.device);
    }

    global_buffers = allocGlobalVals(expr_eval);

    if (kernel()->summary().is_stochastic) {
      // NOTE: this is how we map offset to PW kernels in order to have
      // identical random number generator to match native PyTorch results.
      // But it doesn't really work as it takes assumption how threads are
      // binded but is not generally how we handle that in scheduler.
      // Refer to `Philox` in generated kernel to understand how the mapping
      // works.
      rand_offset = 4 *
          (std::ceil(
               allocated_outputs[0].numel() /
               (4.0 * 128 * launch_params.gdimx())) + // NOLINT
           1);
    }

    // This is the entry when we have provided `opt_code` but the entry has not
    // been initialized yet.
    if (executor_entry) {
      FUSER_PERF_SCOPE("ExecutorRunFusion::FillCacheEntry");
      // record the the short-cut executor entry for the given input set;
      executor_entry->launch_params = launch_params;
      executor_entry->io_alias_indices = alias_indices;
      for (const auto& output : allocated_outputs) {
        executor_entry->output_sizes.push_back(output.sizes().vec());
        executor_entry->output_types.push_back(output.scalar_type());
      }

      for (const auto& i : c10::irange(global_buffers.buffers.size())) {
        executor_entry->buffer_sizes.push_back(
            global_buffers.buffers[i].sizes().vec());
        executor_entry->buffer_types.push_back(
            global_buffers.buffers[i].scalar_type());
        executor_entry->buffer_zero_init.push_back(global_buffers.zero_init[i]);
      }
      executor_entry->rand_offset = rand_offset;
      executor_entry->init = true;
    }
  }

  KernelArgumentHolder kernel_arguments(options_.index_mode);
  {
    FUSER_PERF_SCOPE("ExecutorRunFusion::FillKernelArgStructure");
    kernel_arguments.push(inputs);
    kernel_arguments.push(allocated_outputs);
    kernel_arguments.push(global_buffers.buffers);
    if (lowered_->kernel()->summary().is_stochastic) {
      kernel_arguments.appendPhiloxRNGSeed(rand_offset);
    }
  }

  if (isDebugDumpEnabled(DebugDumpOption::LaunchParam)) {
    launch_params.print();
  }

  if (isDebugDumpEnabled(DebugDumpOption::PrintRuntimeArgs)) {
    std::cout << "Arguments for kernel" << fusion_id_ << ":" << std::endl
              << "Inputs:" << std::endl;
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        const auto& input_tensor = input.toTensor();
        std::cout << "  " << input_tensor.scalar_type() << " "
                  << input.toTensor().sizes()
                  << " (strides = " << input.toTensor().strides() << ")"
                  << std::endl;
      }
    }
    std::cout << "Outputs:" << std::endl;
    for (const auto& output : allocated_outputs) {
      std::cout << "  " << output.scalar_type() << " " << output.sizes()
                << " (strides = " << output.strides() << ")" << std::endl;
    }
    std::cout << "Reduction and semaphore buffers:" << std::endl;
    for (const auto& buffer : global_buffers.buffers) {
      std::cout << "  " << buffer.scalar_type() << " " << buffer.sizes()
                << std::endl;
    }
  }

  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};

  if (measure_kernel_time_ ||
      isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth)) {
    cudaEventCreate(&start_event);
    cudaEventCreate(&finish_event);
    cudaEventRecord(start_event);
  }

  if (execute_kernel_) {
    if (!kernel()->summary().has_cooperative_grid_reduction) {
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchKernel");
      AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
          compiled_kernel_.function,
          launch_params.gdimx(),
          launch_params.gdimy(),
          launch_params.gdimz(),
          launch_params.bdimx(),
          launch_params.bdimy(),
          launch_params.bdimz(),
          launch_params.smem(),
          stream,
          kernel_arguments.getBuffer(),
          nullptr));
    } else {
#ifndef __HIP_PLATFORM_HCC__
      FUSER_PERF_SCOPE("ExecutorRunFusion::cuLaunchCooperativeKernel");
      AT_CUDA_DRIVER_CHECK(
          at::globalContext().getNVRTC().cuLaunchCooperativeKernel(
              compiled_kernel_.function,
              launch_params.gdimx(),
              launch_params.gdimy(),
              launch_params.gdimz(),
              launch_params.bdimx(),
              launch_params.bdimy(),
              launch_params.bdimz(),
              launch_params.smem(),
              stream,
              kernel_arguments.getBuffer()));
#else
      TORCH_INTERNAL_ASSERT(
          false, "Cross grid communication not supported with HIP.");
#endif
    }
  }

  if (measure_kernel_time_ ||
      isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth)) {
    cudaEventRecord(finish_event);
    cudaEventSynchronize(start_event);
    cudaEventSynchronize(finish_event);
    cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event);
    cudaEventDestroy(start_event);
    cudaEventDestroy(finish_event);

    if (isDebugDumpEnabled(DebugDumpOption::EffectiveBandwidth)) {
      size_t bytes = 0;
      // Figure how many bytes are inputs, outputs, and temporary buffers
      for (auto input : inputs) {
        if (input.isTensor()) {
          bytes += input.toTensor().numel() *
              dataTypeSize(aten_to_data_type(input.toTensor().scalar_type()));
        }
      }
      for (const auto& output : allocated_outputs) {
        bytes += output.numel() *
            dataTypeSize(aten_to_data_type(output.scalar_type()));
      }
      double gb_per_s =
          ((double)bytes / ((double)kernel_time_ms_ / 1000)) / (double)1.0e9;
      std::cout << "kernel" << fusion_id_ << " run in " << kernel_time_ms_
                << " ms, achieved: " << gb_per_s << " GB/s" << std::endl;
    }
  }

  return allocated_outputs;
}

void FusionExecutor::compileRtc(
    const std::string& code,
    const std::string& name,
    bool structured) {
  FUSER_PERF_SCOPE("ExecutorRunFusion::compileRtc");
  std::string scode;
  if (!structured) {
    scode = getStructuredCode(code);
  } else {
    scode = code;
  }
  fusion_id_ = 1;
  options_ = CompileOptions();
  compiled_kernel_ = executor_utils::nvrtcCompile(scode, name, fusion_id_);
}

void FusionExecutor::runRtc(
    const LaunchParams& launch_params,
    const std::vector<at::Tensor>& args) {
  FUSER_PERF_SCOPE("runFusion");

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  KernelArgumentHolder kernel_arguments(options_.index_mode);
  kernel_arguments.push(args);
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
      compiled_kernel_.function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      launch_params.smem(),
      stream,
      kernel_arguments.getBuffer(),
      nullptr));
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
