
#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <torch/csrc/jit/codegen/cuda/executor.h>

#include <ATen/cuda/Exceptions.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

int FusionExecutor::fusion_id_counter_ = 0;

std::string FusionExecutor::getStructuredCode(const std::string& kernel) {
  // generating cuda code;
  std::string code = std::string("namespace ") +
      FusionExecutor::kernelNamespace() + " {\n" +
      executor_utils::kernelPreamble() + kernel + "}\n";

  const char* debug_env = getenv("PYTORCH_CUDA_FUSER_DEBUG");
  if (debug_env && atoi(debug_env)) {
    std::cout << "\n==== codegen output for kernel: " << kernelName()
              << " ====" << std::endl
              << code << std::endl
              << "=====*===============================" << std::endl;
  }

  return code;
}

void FusionExecutor::compileFusion(Fusion* fusion, CompileOptions options) {
  TORCH_INTERNAL_ASSERT(
      !fusion->outputs().empty(), "No output found for this kernel, aborting.");

  for (auto out : fusion->outputs()) {
    TORCH_INTERNAL_ASSERT(
        out->getValType() == ValType::TensorView,
        "Output types from fusions that are not tensors are not supported at this point.");
  }

  fusion_ = *fusion;
  FusionGuard fg(&fusion_);
  options_ = options;

  fusion_id_ = ++fusion_id_counter_;
  has_random_ = fusion->hasRNG();
  lowered_ = GpuLower(&fusion_);
  const auto kernel = lowered_.getKernel(kernelName());
  const auto structured_code = getStructuredCode(kernel);

  compiled_kernel_ = executor_utils::nvrtcCompile(
      structured_code,
      (kernelNamespace() + "::" + kernelName()).c_str(),
      fusion_id_);
  compiled_ = true;
}

namespace {

at::Tensor inferAndAlloc(
    const TensorView* tv,
    EvaluationContext& ec,
    const CompileOptions& options,
    bool zero_init = false) {
  std::vector<int64_t> sizes;
  for (auto id : TensorDomain::noReductions(tv->getRootDomain())) {
    auto infered_val = ExpressionEvaluator::evaluate(id->rawExtent(), &ec);
    TORCH_INTERNAL_ASSERT(
        infered_val.has_value(),
        "Could not launch kernel as program could not infer ",
        id->rawExtent(),
        " for the buffer ",
        tv);
    sizes.push_back(infered_val.value());
  }

  auto at_type = data_type_to_aten(tv->getDataType().value());
  auto tensor_options =
      at::TensorOptions().dtype(at_type).device(options.device);

  if (zero_init) {
    c10::IntArrayRef isizes(sizes);
    return at::zeros(isizes, tensor_options);
  } else {
    c10::IntArrayRef isizes(sizes);
    return at::empty(isizes, tensor_options);
  }
}

} // namespace

LaunchParams FusionExecutor::computeLaunchParams(
    const at::ArrayRef<IValue>& aten_inputs,
    const LaunchParams& launch_constraints,
    EvaluationContext& ec) {
  LaunchParams launch_params;

  // Grab all values that are actually used in the fusion
  auto unordered_vals = DependencyCheck::getAllValsBetween(
      {fusion_.inputs().begin(), fusion_.inputs().end()}, fusion_.outputs());

  // Lets collect all IterDomains that are bound to a thread binding
  std::unordered_map<ParallelType, std::vector<IterDomain*>, TypeHash>
      parallel_iter_domains;

  for (auto val : unordered_vals) {
    if (val->getValType().value() == ValType::TensorView) {
      TensorView* tv = val->as<TensorView>();
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
  }

  // If any dimension was set in launch constraints we need to run through
  // IterDomains that have been parallelized, and bind those values. Or make
  // sure if they could be infered the inference matches what was set.
  if (launch_constraints.nBlocks() * launch_constraints.nThreads() != -1) {
    for (auto& entry : parallel_iter_domains) {
      auto p_type = entry.first;
      if (launch_constraints.hasDim(p_type)) {
        auto parallel_ids = entry.second;
        for (auto parallel_id : parallel_ids) {
          auto infered_val =
              ExpressionEvaluator::evaluate(parallel_id->rawExtent(), &ec);
          if (infered_val.has_value()) {
            // This value could have been infered, make sure it was set right.
            TORCH_CHECK(
                infered_val.value() == launch_constraints.getDim(p_type) ||
                    launch_constraints.getRawVal(p_type) == -1,
                "Infered that ",
                p_type,
                " should be set to ",
                infered_val.value(),
                " but launch constraints specified ",
                launch_constraints.getDim(p_type));
          } else {
            // Bind the launch constraint into our evaluation context
            executor_utils::safeBind(
                ec,
                parallel_id->rawExtent(),
                launch_constraints.getDim(entry.first));
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
      auto val = ExpressionEvaluator::evaluate(parallel_id->rawExtent(), &ec);
      TORCH_INTERNAL_ASSERT(
          val,
          "Tried to evaluate the extent of ",
          parallel_id,
          " to set launch bounds but could not.");
      launch_params.bind(val.value(), p_type);
    }
  }

  return launch_params;
}

std::vector<at::Tensor> FusionExecutor::allocGlobalVals(EvaluationContext& ec) {
  std::vector<at::Tensor> global_buffers;
  for (auto alloc : lowered_.global_allocations()) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->getValType() == ValType::KirTensorView,
        "Cannot allocate global buffers that are not tensors.");
    global_buffers.push_back(inferAndAlloc(
        alloc->buffer()->as<kir::TensorView>()->fuserTv(),
        ec,
        options_,
        false));
  }

  for (auto alloc : lowered_.sync_allocations()) {
    TORCH_INTERNAL_ASSERT(
        alloc->buffer()->getValType() == ValType::KirTensorView,
        "Cannot allocate global buffers that are not tensors.");
    global_buffers.push_back(inferAndAlloc(
        alloc->buffer()->as<kir::TensorView>()->fuserTv(), ec, options_, true));
  }

  return global_buffers;
}

std::vector<at::Tensor> FusionExecutor::allocOutputs(EvaluationContext& ec) {
  std::vector<at::Tensor> outputs;
  for (auto output : fusion_.outputs()) {
    TORCH_INTERNAL_ASSERT(
        output->getValType() == ValType::TensorView,
        "Cannot allocate outputs that are not tensors.");
    outputs.push_back(
        inferAndAlloc(output->as<TensorView>(), ec, options_, false));
  }
  return outputs;
}

std::vector<at::Tensor> FusionExecutor::runFusion(
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    const LaunchParams& launch_constraints) {
  TORCH_INTERNAL_ASSERT(
      fusion_id_ > 0, "Cannot run fusion, it was not compiled.");

  FusionGuard fg(&fusion_);

  executor_utils::validateKernelInputs(&fusion_, inputs, options_.device);

  c10::DeviceGuard dg(options_.device);
  auto stream = at::cuda::getCurrentCUDAStream();

  EvaluationContext evaluation_context =
      executor_utils::bindInputs(inputs, &fusion_);

  LaunchParams launch_params =
      computeLaunchParams(inputs, launch_constraints, evaluation_context);

  std::vector<at::Tensor> alloced_outputs = outputs;
  if (outputs.empty() || outputs.size() != fusion_.outputs().size()) {
    alloced_outputs = allocOutputs(evaluation_context);
  }

  executor_utils::validateKernelOutputs(
      &fusion_, alloced_outputs, options_.device);

  KernelArgumentHolder kernel_arguments;
  kernel_arguments.push(inputs);
  kernel_arguments.push(alloced_outputs);
  auto buffers = allocGlobalVals(evaluation_context);
  kernel_arguments.push(buffers);

  if (has_random_) {
    const auto rand_offset = 4 *
        (std::ceil(
             alloced_outputs[0].numel() / (4.0 * 128 * launch_params.gdimx())) +
         1);
    kernel_arguments.appendPhiloxRNGSeed(rand_offset);
  }

  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLaunchKernel(
      compiled_kernel_.function,
      launch_params.gdimx(),
      launch_params.gdimy(),
      launch_params.gdimz(),
      launch_params.bdimx(),
      launch_params.bdimy(),
      launch_params.bdimz(),
      0, // smem
      stream,
      kernel_arguments.getBuffer(),
      nullptr));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  return alloced_outputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
