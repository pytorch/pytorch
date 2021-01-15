#include <torch/csrc/jit/codegen/cuda/codegen.h>
#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <torch/csrc/jit/codegen/cuda/executor.h>

#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdlib>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int FusionExecutor::fusion_id_counter_ = 0;

std::string FusionExecutor::getStructuredCode(const std::string& kernel) {
  // generating cuda code;
  std::string code = "";
#ifdef __HIP_PLATFORM_HCC__
  code += std::string("#include <hip/hip_runtime.h>\n") +
      std::string("#include <hip/hip_fp16.h>\n");
#endif
  code += std::string("namespace ") + FusionExecutor::kernelNamespace() +
      " {\n" + executor_utils::kernelPreamble() + kernel + "}\n";

  const char* debug_env = std::getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "======================================\n"
              << std::endl;
  }

  return code;
}

void FusionExecutor::debugCompileFusionFromStr(
    Fusion* fusion,
    const std::string& code,
    const std::string& name,
    int id,
    CompileOptions options) {
  fusion_ = *fusion;
  FusionGuard fg(&fusion_);
  options_ = options;

  const char* debug_env = std::getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "======================================\n"
              << std::endl;
  }

  setUsedTVs();

  fusion_id_ = id;
  lowered_ = GpuLower(&fusion_);
  const auto kernel = lowered_.kernel();

  const char* dump_kir_env = std::getenv("PYTORCH_CUDA_FUSER_DUMP_KIR");
  if (dump_kir_env && atoi(dump_kir_env)) {
    kernel->print();
  }

  const auto& kernel_summary = kernel->summary();
  has_block_reductions = kernel_summary.has_block_reductions;
  has_grid_reductions = kernel_summary.has_grid_reductions;
  has_block_broadcasts = kernel_summary.has_block_broadcasts;

  if (!kernel_summary.static_smem_allocations.empty()) {
    StatefulExpressionEvaluator static_evaluator(&fusion_);
    unsigned static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_device_smem,
        "The static shared memory allocation is larger than available memory.");
  }

  compiled_kernel_ = executor_utils::nvrtcCompile(code, name, fusion_id_);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "assign a fusion_id_ <= 0 is not accepted.");
}

void FusionExecutor::compileFusion(Fusion* fusion, CompileOptions options) {
  FUSER_PERF_SCOPE("compileFusion");

  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs()) {
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");
  }

  // Clone the fusion so we can store it
  fusion_ = *fusion;
  FusionGuard fg(&fusion_);
  options_ = options;

  TORCH_INTERNAL_ASSERT(
      options.device.is_cuda(), "Provided device to CUDA fuser is the CPU.");
  max_device_smem =
      at::cuda::getDeviceProperties(options.device.index())->sharedMemPerBlock;

  setUsedTVs();

  fusion_id_ = ++fusion_id_counter_;
  lowered_ = GpuLower(&fusion_);
  const auto kernel = lowered_.kernel();

  const char* dump_kir_env = std::getenv("PYTORCH_CUDA_FUSER_DUMP_KIR");
  if (dump_kir_env && atoi(dump_kir_env)) {
    kernel->print();
  }

  const auto kernel_code = codegen::generateCudaKernel(kernel, kernelName());
  const auto structured_code = getStructuredCode(kernel_code);

  const auto& kernel_summary = kernel->summary();
  has_block_reductions = kernel_summary.has_block_reductions;
  has_grid_reductions = kernel_summary.has_grid_reductions;
  has_block_broadcasts = kernel_summary.has_block_broadcasts;

  if (!kernel_summary.static_smem_allocations.empty()) {
    StatefulExpressionEvaluator static_evaluator(&fusion_);
    unsigned static_smem_size = computeSharedMemory(
        static_evaluator, kernel_summary.static_smem_allocations);
    TORCH_INTERNAL_ASSERT(
        static_smem_size < max_device_smem,
        "The static shared memory allocation is larger than available memory.");
  }

  compiled_kernel_ = executor_utils::nvrtcCompile(
      structured_code,
      (kernelNamespace() + "::" + kernelName()).c_str(),
      fusion_id_);
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "failed to assign a fusion_id_ after compilation.");
}

namespace {

at::Tensor inferAndAlloc(
    const TensorView* tv,
    StatefulExpressionEvaluator& see,
    const CompileOptions& options,
    bool zero_init = false) {
  FUSER_PERF_SCOPE("inferAndAlloc");

  std::vector<int64_t> sizes;
  for (auto id : TensorDomain::noReductions(tv->getMaybeRFactorDomain())) {
    auto inferred_val = see.inferValue(id->rawExtent());
    TORCH_INTERNAL_ASSERT(
        inferred_val.has_value(),
        "Could not launch kernel as program could not infer ",
        id->rawExtent(),
        " for the buffer ",
        tv);
    sizes.push_back(inferred_val.value());
  }

  auto at_type = data_type_to_aten(tv->getDataType().value());

  if (zero_init) {
    auto tensor_options =
        at::TensorOptions().dtype(at_type).device(options.device);
    c10::IntArrayRef isizes(sizes);
    return at::zeros(isizes, tensor_options);
  } else {
    c10::IntArrayRef isizes(sizes);
    // Non Variable type guard for empty_cuda call
    at::AutoNonVariableTypeMode non_variable_type_mode;
    return at::native::empty_cuda(
        isizes, at_type, c10::nullopt, options.device, c10::nullopt);
  }
}

} // namespace

uint64_t FusionExecutor::computeSharedMemory(
    StatefulExpressionEvaluator& see,
    const std::vector<kir::Allocate*>& buffers,
    bool align_padding,
    uint64_t total) {
  FUSER_PERF_SCOPE("computeSharedMemory");
  for (auto smem_alloc : buffers) {
    // If this buffer aliases another buffer,
    // then do not allocate memory for this buffer.
    if (smem_alloc->alias() == nullptr) {
      auto inferred_val = see.inferValue(smem_alloc->size());
      if (inferred_val.has_value()) {
        const uint64_t data_size = dataTypeSize(smem_alloc->buffer_type());
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
    StatefulExpressionEvaluator& see) {
  FUSER_PERF_SCOPE("computeLaunchParams");

  LaunchParams launch_params;

  // Lets collect all IterDomains that are bound to a thread binding
  std::unordered_map<ParallelType, std::vector<IterDomain*>, TypeHash>
      parallel_iter_domains;
  for (auto tv : getUsedTVs()) {
    for (auto id : tv->domain()->domain()) {
      if (id->isThread() && !id->isBroadcast()) {
        if (parallel_iter_domains.find(id->getParallelType()) !=
            parallel_iter_domains.end()) {
          parallel_iter_domains.at(id->getParallelType()).push_back(id);
        } else {
          parallel_iter_domains[id->getParallelType()] =
              std::vector<IterDomain*>({id});
        }
      }
    }
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be inferred the inference matches what was set.
  if (launch_constraints.nBlocks() * launch_constraints.nThreads() != -1) {
    for (auto& entry : parallel_iter_domains) {
      auto p_type = entry.first;
      if (launch_constraints.hasDim(p_type)) {
        auto parallel_ids = entry.second;
        for (auto parallel_id : parallel_ids) {
          auto inferred_val = see.inferValue(parallel_id->rawExtent());
          if (inferred_val.has_value()) {
            // This value could have been inferred, make sure it was set right.
            TORCH_CHECK(
                inferred_val.value() == launch_constraints.getDim(p_type) ||
                    launch_constraints.getRawVal(p_type) == -1,
                "inferred that ",
                p_type,
                " should be set to ",
                inferred_val.value(),
                " but launch constraints specified ",
                launch_constraints.getDim(p_type));
          } else {
            // Bind the launch constraint into our evaluation context
            see.safeBind(
                parallel_id->rawExtent(),
                launch_constraints.getDim(entry.first),
                &lowered_);
            launch_params.bind(launch_constraints.getDim(p_type), p_type);
          }
        }
      }
    }
  }

  // Run through the rest of the parallel IterDomains and infer their size
  for (auto& entry : parallel_iter_domains) {
    auto p_type = entry.first;
    auto parallel_ids = entry.second;
    for (auto parallel_id : parallel_ids) {
      auto val = see.inferValue(parallel_id->rawExtent());
      TORCH_INTERNAL_ASSERT(
          val,
          "Tried to evaluate the extent of ",
          parallel_id,
          " to set launch bounds but could not.");
      launch_params.bind(val.value(), p_type);
    }
  }

  const auto kernel = lowered_.kernel();
  const auto& kernel_summary = kernel->summary();

  // Calculate Dynamic Shared Memory Size
  // Add workspace for reduction and broadcast
  uint64_t reduction_broadcast_workspace = 0;
  if (has_block_reductions || has_grid_reductions || has_block_broadcasts) {
    // Not using nThreads here since it does not handle uninitialized value
    reduction_broadcast_workspace =
        dataTypeSize(kernel_summary.largest_smem_data_type) *
        launch_params.bdimx() * launch_params.bdimy() * launch_params.bdimz();
  }

  const uint64_t dynamic_smem_size = computeSharedMemory(
      see,
      kernel_summary.dynamic_smem_allocations,
      true,
      reduction_broadcast_workspace);

  const uint64_t static_smem_size =
      computeSharedMemory(see, kernel_summary.static_smem_allocations);

  TORCH_INTERNAL_ASSERT(
      (dynamic_smem_size + static_smem_size) < max_device_smem,
      "The total shared memory allocation is larger than available memory.");
  launch_params.setSmem(dynamic_smem_size);

  return launch_params;
}

FusionExecutor::GlobalBuffers FusionExecutor::allocGlobalVals(
    StatefulExpressionEvaluator& see) {
  FUSER_PERF_SCOPE("allocGlobalVals");
  GlobalBuffers global_buffers;
  const auto& kernel_summary = lowered_.kernel()->summary();
  for (auto alloc : kernel_summary.global_allocations) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->getValType() == ValType::KirTensorView,
        "Cannot allocate global buffers that are not tensors.");
    if (!alloc->zeroInit()) {
      global_buffers.empty_buffers.push_back(inferAndAlloc(
          alloc->buffer()->as<kir::TensorView>()->fuserTv(),
          see,
          options_,
          false));
    } else {
      global_buffers.zero_buffers.push_back(inferAndAlloc(
          alloc->buffer()->as<kir::TensorView>()->fuserTv(),
          see,
          options_,
          true));
    }
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::allocOutputs(
    StatefulExpressionEvaluator& see) {
  FUSER_PERF_SCOPE("allocOutputs");
  std::vector<at::Tensor> outputs;
  for (auto output : fusion_.outputs()) {
    TORCH_INTERNAL_ASSERT(
        output->getValType() == ValType::TensorView,
        "Cannot allocate outputs that are not tensors.");
    outputs.push_back(
        inferAndAlloc(output->as<TensorView>(), see, options_, false));
  }
  return outputs;
}

void FusionExecutor::setUsedTVs() {
  used_tvs_.clear();
  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion_.inputs().begin(), fusion_.inputs().end()}, fusion_.outputs());
  for (auto val : used_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      used_tvs_.push_back(val->as<TensorView>());
    }
  }
}

std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    const LaunchParams& launch_constraints,
    const c10::optional<size_t>& opt_code) {
  FUSER_PERF_SCOPE("runFusion");

  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "Cannot run fusion, it was not compiled.");
  TORCH_INTERNAL_ASSERT(
      !opt_code.has_value() || outputs.empty(),
      "short cut input cache is not compatible with pre-allocated output");

  ExecutorEntry* executor_entry = nullptr;
  if (opt_code.has_value()) {
    executor_entry = &executor_entry_lookup_[*opt_code];
  }

  FusionGuard fg(&fusion_);
  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  LaunchParams launch_params;
  std::vector<at::Tensor> alloced_outputs = outputs;
  GlobalBuffers global_buffers;
  uint64_t rand_offset = 0;

  if (executor_entry && executor_entry->init) {
    {
      // context manager to disable auto grad for `empty_cuda` calls later;
      at::AutoNonVariableTypeMode non_variable_type_mode;
      // take the short-cut for launch if we see a recorded input set again;
      launch_params = executor_entry->launch_params;
      for (size_t i = 0; i < executor_entry->output_sizes.size(); i++) {
        alloced_outputs.push_back(at::native::empty_cuda(
            executor_entry->output_sizes[i],
            executor_entry->output_types[i],
            c10::nullopt,
            options_.device,
            c10::nullopt));
      }
      for (size_t i = 0; i < executor_entry->empty_buffer_sizes.size(); i++) {
        global_buffers.empty_buffers.push_back(at::native::empty_cuda(
            executor_entry->empty_buffer_sizes[i],
            executor_entry->empty_buffer_types[i],
            c10::nullopt,
            options_.device,
            c10::nullopt));
      }
    }
    for (size_t i = 0; i < executor_entry->zero_buffer_sizes.size(); i++) {
      auto tensor_options = at::TensorOptions()
                                .dtype(executor_entry->zero_buffer_types[i])
                                .device(options_.device);
      global_buffers.zero_buffers.push_back(
          at::zeros(executor_entry->zero_buffer_sizes[i], tensor_options));
    }
    rand_offset = executor_entry->rand_offset;
  } else {
    // code path to take when either:
    //   1. no opt_code is provided or;
    //   2. `executor_entry` is not initialized
    executor_utils::validateKernelInputs(&fusion_, inputs, options_.device);

    StatefulExpressionEvaluator evaluator =
        executor_utils::statefulBindInputs(inputs, &fusion_, &lowered_);

    launch_params = computeLaunchParams(launch_constraints, evaluator);

    if (outputs.empty() || outputs.size() != fusion_.outputs().size()) {
      alloced_outputs = allocOutputs(evaluator);
    } else {
      executor_utils::validateKernelOutputs(
          &fusion_, alloced_outputs, options_.device);
    }

    global_buffers = allocGlobalVals(evaluator);

    if (lowered_.kernel()->summary().is_stochastic) {
      // NOTE: this is how we map offset to PW kernels in order to have
      // identical random number generator to match native PyTorch results.
      // But it doesn't really work as it takes assumption how threads are
      // binded but is not generally how we handle that in scheduler.
      // Refer to `Philox` in generated kernel to understand how the mapping
      // works.
      rand_offset = 4 *
          (std::ceil(
               alloced_outputs[0].numel() /
               (4.0 * 128 * launch_params.gdimx())) + // NOLINT
           1);
    }

    // This is the entry when we have provided `opt_code` but the entry has not
    // been initialized yet.
    if (executor_entry) {
      // record the the short-cut executor entry for the given input set;
      executor_entry->launch_params = launch_params;
      for (const auto& output : alloced_outputs) {
        executor_entry->output_sizes.push_back(output.sizes().vec());
        executor_entry->output_types.push_back(output.scalar_type());
      }
      for (const auto& buffer : global_buffers.empty_buffers) {
        executor_entry->empty_buffer_sizes.push_back(buffer.sizes().vec());
        executor_entry->empty_buffer_types.push_back(buffer.scalar_type());
      }
      for (const auto& buffer : global_buffers.zero_buffers) {
        executor_entry->zero_buffer_sizes.push_back(buffer.sizes().vec());
        executor_entry->zero_buffer_types.push_back(buffer.scalar_type());
      }
      executor_entry->rand_offset = rand_offset;
      executor_entry->init = true;
    }
  }

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.push(inputs);
  kernel_arguments.push(alloced_outputs);
  kernel_arguments.push(global_buffers.empty_buffers);
  kernel_arguments.push(global_buffers.zero_buffers);
  if (lowered_.kernel()->summary().is_stochastic) {
    kernel_arguments.appendPhiloxRNGSeed(rand_offset);
  }

  {
    FUSER_PERF_SCOPE("cuLaunchKernel");
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
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return alloced_outputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
