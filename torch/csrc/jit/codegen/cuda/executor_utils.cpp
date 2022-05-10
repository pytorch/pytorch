#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/irange.h>

#include <torch/csrc/jit/codegen/cuda/contiguity.h>
#include <torch/csrc/jit/codegen/cuda/executor_utils.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/fuser/cuda/fused_kernel.h>
#include <torch/csrc/jit/resource_guard.h>

#include <nvfuser_resources/PhiloxCudaStateRaw.h>
#include <nvfuser_resources/array.h>
#include <nvfuser_resources/bf16_support.h>
#include <nvfuser_resources/block_reduction.h>
#include <nvfuser_resources/block_sync_atomic.h>
#include <nvfuser_resources/block_sync_default.h>
#include <nvfuser_resources/broadcast.h>
#include <nvfuser_resources/fp16_support.h>
#include <nvfuser_resources/fused_reduction.h>
#include <nvfuser_resources/grid_broadcast.h>
#include <nvfuser_resources/grid_reduction.h>
#include <nvfuser_resources/grid_sync.h>
#include <nvfuser_resources/helpers.h>
#include <nvfuser_resources/index_utils.h>
#include <nvfuser_resources/random_numbers.h>
#include <nvfuser_resources/tensor.h>
#include <nvfuser_resources/tensorcore.h>
#include <nvfuser_resources/tuple.h>
#include <nvfuser_resources/type_traits.h>
#include <nvfuser_resources/warp.h>
#include <nvfuser_resources/welford.h>

#ifndef USE_ROCM
#include <cuda_occupancy.h>
#endif

#include <fstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace executor_utils {

std::string kernelPreamble() {
  std::stringstream ss;

#ifndef __HIP_PLATFORM_HCC__
  ss << nvfuser_resources::fp16_support_cu;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ss << nvfuser_resources::bf16_support_cu;
#endif
#else
  ss << R"(
#ifndef __noinline__
#define __noinline__ __attribute__((noinline))
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef assert
#define assert(expr) ((void)0)
#endif
#ifndef __align__
#define __align__(x) __attribute__((aligned(x)))
#endif
  )";
#endif

  // Base classes and helpers
  ss << nvfuser_resources::tensor_cu;
  ss << nvfuser_resources::type_traits_cu;
  ss << nvfuser_resources::array_cu;
  ss << nvfuser_resources::random_numbers_cu;
  ss << nvfuser_resources::helpers_cu;
  ss << nvfuser_resources::index_utils_cu;
  ss << nvfuser_resources::tuple_cu;

  // Synchronization classes
  if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
    ss << nvfuser_resources::block_sync_atomic_cu;
  } else {
    ss << nvfuser_resources::block_sync_default_cu;
  }
  ss << nvfuser_resources::grid_sync_cu;

  // Communication classes
  ss << nvfuser_resources::block_reduction_cu;
  ss << nvfuser_resources::grid_reduction_cu;
  ss << nvfuser_resources::grid_broadcast_cu;
  ss << nvfuser_resources::broadcast_cu;
  ss << nvfuser_resources::welford_cu;
  ss << nvfuser_resources::warp_cu;
  ss << nvfuser_resources::tensorcore_cu;
  ss << nvfuser_resources::fused_reduction_cu;

  // Random utilities
  ss << nvfuser_resources::PhiloxCudaStateRaw_cu;

  return ss.str();
}

namespace {

// return false if arg's type, number of dimensions, and device, doesn't match
// param and provided c10:device
bool validateKernelArgTensor(
    const at::Tensor& arg,
    const Val* param,
    const c10::Device& device,
    std::stringstream& msg) {
  // Arg is a tensor. Param must be a tensor too.
  if (*param->getValType() != ValType::TensorView) {
    msg << "Argument is a tensor, but the parameter is not.\n";
    return false;
  }

  if (is_cpu_scalar(arg) && !param->as<TensorView>()->isCpuScalar()) {
    msg << "Argument is CPU Scalar Tensor, but parameter is not.\n";
    return false;
  }

  if (!is_cpu_scalar(arg) && !arg.is_cuda()) {
    msg << "Argumnet is a CPU tensor which is not supported in fusions.\n";
    return false;
  }

  // Check the rank of the tensors.
  size_t arg_dim = arg.dim();
  // Note: This requires current Fusion to be active.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t param_dim = TensorDomain::noReductions(
                         param->as<TensorView>()->getMaybeRFactorDomain())
                         .size();
  // see [Note - broadcast support in integration]
  // Because of broadcasting support handled in integration, we relax the rank
  // check as necessary.
  if (arg_dim > param_dim) {
    msg << "Argument tensor's rank is " << arg_dim << ", but the parameter is "
        << param_dim << "\n";
    return false;
  }

  if (!is_cpu_scalar(arg) && arg.device() != device) {
    msg << "Argument is on device that is not compiled for."
        << "\n";
    return false;
  }
  // Check element type
  at::ScalarType arg_data_type = arg.scalar_type();
  DataType param_data_type = *param->getDataType();
  bool match = false;
  switch (arg_data_type) {
    case at::ScalarType::Double:
      match = param_data_type == DataType::Double;
      break;
    case at::ScalarType::Half:
      match = param_data_type == DataType::Half;
      break;
    case at::ScalarType::BFloat16:
      match = param_data_type == DataType::BFloat16;
      break;
    case at::ScalarType::Float:
      match = param_data_type == DataType::Float;
      break;
    case at::ScalarType::Long:
      match = param_data_type == DataType::Int;
      break;
    case at::ScalarType::Int:
      match = param_data_type == DataType::Int32;
      break;
    case at::ScalarType::Bool:
      match = param_data_type == DataType::Bool;
      break;
    case at::ScalarType::ComplexFloat:
      match = param_data_type == DataType::ComplexFloat;
      break;
    case at::ScalarType::ComplexDouble:
      match = param_data_type == DataType::ComplexDouble;
      break;
    default:
      msg << "Argument element type, " << arg_data_type << ", is not supported."
          << "\n";
      return false;
  }
  if (!match)
    msg << "Argument element type is " << arg_data_type
        << ", but the parameter is " << param_data_type << "\n";
  return match;
}

// Return false if  arg_type doesn't match the type in param
bool validateKernelArgScalar(
    const c10::IValue& arg,
    const Val* param,
    std::stringstream& msg) {
  if (!arg.isScalar()) {
    msg << "Argument is a scalar, but the parameter is not."
        << "\n";
    return false;
  }
  DataType param_type = *param->getDataType();
  bool match = false;
  switch (arg.toScalar().type()) {
    case c10::ScalarType::Long:
      match = param_type == DataType::Int || param_type == DataType::Int32;
      break;
    case c10::ScalarType::ComplexDouble:
      match = param_type == DataType::ComplexDouble ||
          param_type == DataType::ComplexFloat;
      break;
    case c10::ScalarType::Double:
      match = param_type == DataType::Double || param_type == DataType::Float ||
          param_type == DataType::Half || param_type == DataType::BFloat16;
      break;
    case c10::ScalarType::Bool:
      match = param_type == DataType::Bool;
      break;
    default:
      match = false;
  }
  if (!match) {
    msg << "Argument type is " << arg.toScalar().type()
        << ", but the parameter is " << param_type << "\n";
  }
  return match;
}

// Return false if arg and param don't match up and if arg's device (if a
// tensor) doesn't match provided device
bool validateKernelArg(
    const c10::IValue& arg,
    const Val* param,
    const c10::Device& device,
    std::stringstream& msg) {
  if (arg.isTensor()) {
    return validateKernelArgTensor(arg.toTensor(), param, device, msg);
  } else {
    return validateKernelArgScalar(arg, param, msg);
  }
}

// Return true if all the tensors have the same stride, assumes all tensors are
// contiguous
bool checkSameStride(const std::vector<c10::IValue>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }
  for (const auto idx : c10::irange(tensors.size() - 1)) {
    auto current = tensors[idx];
    auto next = tensors[idx + 1];
    if (!current.isTensor() || !next.isTensor()) {
      return false;
    }

    const auto& current_tensor = current.toTensor();
    const auto& next_tensor = next.toTensor();
    if (current_tensor.ndimension() != next_tensor.ndimension()) {
      return false;
    }

    for (const auto i : c10::irange(current_tensor.ndimension())) {
      if (current_tensor.stride(i) != next_tensor.stride(i)) {
        return false;
      }
    }
  }
  return true;
}

// Return true if all the tensors are contiguous and have the same striding
bool checkSameContiguity(const std::vector<c10::IValue>& tensors) {
  if (tensors.size() < 2) {
    return true;
  }

  auto reference = tensors.front();
  if (!reference.isTensor()) {
    return false;
  }

  // Determine if the reference tensor is contiguous
  const auto& reference_tensor = reference.toTensor();
  int64_t expected_stride = 1;
  for (const auto i : c10::irange(1, reference_tensor.ndimension() + 1)) {
    int64_t ind = reference_tensor.ndimension() - i;
    if (reference_tensor.size(ind) == 1) {
      continue;
    }
    if (reference_tensor.stride(ind) != expected_stride) {
      return false;
    }
    expected_stride *= reference_tensor.size(ind);
  }

  // Check if all the tensors have the same contiguity
  return checkSameStride(tensors);
}

bool checkValidMisalignedTensors(
    const std::unordered_set<TensorView*>& inp_tv,
    const std::unordered_set<TensorView*>& out_tv,
    const std::vector<c10::IValue>& inp_tensors,
    const std::vector<c10::IValue>& out_tensors) {
  if (out_tv.empty()) {
    // Only check input tensors
    return checkSameStride(inp_tensors);
  } else if (!out_tv.empty() && out_tensors.empty()) {
    // out_tensors is empty unless outputs are given to runFusion.
    // Assume out tensors are contiguous
    return checkSameContiguity(inp_tensors);
  } else {
    // Only check input and output tensors
    std::vector<c10::IValue> tensors;
    tensors.insert(tensors.end(), inp_tensors.begin(), inp_tensors.end());
    tensors.insert(tensors.end(), out_tensors.begin(), out_tensors.end());
    return checkSameStride(tensors);
  }
}

} // namespace

void validateKernelInputs(
    Fusion* fusion,
    const at::ArrayRef<IValue>& inputs,
    const c10::Device& device) {
  FUSER_PERF_SCOPE("executor_utils::ValidateKernelInputs");

  // This is necessary as we were traversing the fusion graph later in the check
  FusionGuard fg(fusion);
  // Check inputs
  TORCH_INTERNAL_ASSERT(
      inputs.size() == fusion->inputs().size(),
      "Wrong number of kernel inputs.");

  std::stringstream msg;
  bool mismatch = false;
  for (const auto i : c10::irange(inputs.size())) {
    const IValue& arg = inputs[i];
    const Val* param = fusion->inputs()[i];
    mismatch = !validateKernelArg(arg, param, device, msg) || mismatch;
  }
  TORCH_INTERNAL_ASSERT(
      !mismatch, "Found one or more invalid arguments: ", msg.str());
}

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    const c10::Device& device) {
  FUSER_PERF_SCOPE("executor_utils::ValidateKernelOutputs");

  TORCH_INTERNAL_ASSERT(
      fusion->outputs().size() != 0,
      "Kernel should have at least one output tensor.");

  TORCH_INTERNAL_ASSERT(
      outputs.size() == fusion->outputs().size(),
      "Wrong number of kernel outputs.");

  std::stringstream msg;
  bool mismatch = false;
  for (const auto i : c10::irange(outputs.size())) {
    const at::Tensor& arg = outputs[i];
    const Val* param = fusion->outputs()[i];
    mismatch = !validateKernelArg(arg, param, device, msg) || mismatch;
  }
  TORCH_INTERNAL_ASSERT(
      !mismatch, "Found one or more invalid arguments: ", msg.str());
}

namespace {

// Finds a fusion input or output tensor to validate its stides
// for vectorization.
// Returns a pair consisting of a flag indicating it's a fusion input
// and an integer position within in the input or output tensor list.
std::vector<std::pair<bool, int>> getVectorizedFusionInputOutput(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    Fusion* fusion) {
  std::vector<std::pair<bool, int>> vectorized_input_output;

  // When the producer is a fusion input, validate only the producer
  // and assume the consumer is contiguous. Similarly, when the
  // consumer is a fusion output, validate the consumer and assume the
  // producer is contiguous.

  if (producer_tv->isFusionInput()) {
    auto producer_it = std::find(
        fusion->inputs().begin(), fusion->inputs().end(), producer_tv);
    TORCH_INTERNAL_ASSERT(
        producer_it != fusion->inputs().end(),
        "Could not find ",
        producer_tv,
        " in fusion inputs.");
    auto pos = std::distance(fusion->inputs().begin(), producer_it);
    vectorized_input_output.push_back(
        std::make_pair<bool, int>(true, static_cast<int>(pos)));
  } else {
    // If not fusion input, assume it's fully contiguous, so nothing
    // to check with respect to strides.
    TORCH_INTERNAL_ASSERT(
        std::all_of(
            producer_tv->domain()->contiguity().begin(),
            producer_tv->domain()->contiguity().end(),
            [](bool contig) { return contig; }),
        "Unsupported pattern of vectorization: ",
        consumer_tv->definition()->toString());
  }

  if (consumer_tv->isFusionOutput()) {
    auto consumer_it = std::find(
        fusion->outputs().begin(), fusion->outputs().end(), consumer_tv);
    TORCH_INTERNAL_ASSERT(
        consumer_it != fusion->outputs().end(),
        "Could not find ",
        consumer_tv,
        " in fusion outputs.");
    auto pos = std::distance(fusion->outputs().begin(), consumer_it);
    vectorized_input_output.push_back(
        std::make_pair<bool, int>(false, static_cast<int>(pos)));
  } else {
    // If not fusion input, assume it's fully contiguous, so nothing
    // to check with respect to strides.
    TORCH_INTERNAL_ASSERT(
        std::all_of(
            consumer_tv->domain()->contiguity().begin(),
            consumer_tv->domain()->contiguity().end(),
            [](bool contig) { return contig; }),
        "Unsupported pattern of vectorization: ",
        consumer_tv->definition()->toString());
  }

  return vectorized_input_output;
}

//! Returns the information of vectorized input/output tensors
//! in the given fusion.
std::unique_ptr<caching::VectorizedTensorInfo> getVectorizedTensorValidationInfo(
    kir::Kernel* kernel) {
  auto vectorized_tensor_info_ptr =
      std::make_unique<caching::VectorizedTensorInfo>();

  for (const auto& vector_info : kernel->summary().vectorized_set_info) {
    auto consumer_tv = vector_info.consumer_tv;
    auto producer_tv = vector_info.producer_tv;

    auto vector_dim = vector_info.vectorized_leaf_id;
    const auto is_aligned =
        vector_dim->getParallelType() == ParallelType::Vectorize;

    // Find fusion inputs and outputs that are used with misaligned
    // vectorization.
    if (!is_aligned) {
      TORCH_INTERNAL_ASSERT(
          producer_tv->isFusionInput() || consumer_tv->isFusionOutput(),
          "MisalignedVectorize is assumed to be used with either input or output tensor");
      if (consumer_tv->getMemoryType() == MemoryType::Global &&
          producer_tv->getMemoryType() == MemoryType::Local) {
        vectorized_tensor_info_ptr->global_out_misaligned_tv.insert(
            consumer_tv);
      } else if (
          producer_tv->getMemoryType() == MemoryType::Global &&
          consumer_tv->getMemoryType() == MemoryType::Local) {
        vectorized_tensor_info_ptr->global_inp_misaligned_tv.insert(
            producer_tv);
      } else {
        TORCH_INTERNAL_ASSERT(
            false,
            "Unsupported memory configuration for misaligned vectorization.");
      }
    }

    // Collect information on corresponding fusion input and output
    // tensors to verify strides.
    auto inp_or_out_info =
        getVectorizedFusionInputOutput(producer_tv, consumer_tv, kernel);

    // If both producer and consumer are contig and intermediate,
    // nothing to validate with respect to strides.
    if (inp_or_out_info.empty()) {
      continue;
    }

    // Misaligned vectorize only allows from input to local or local
    // to output
    if (!is_aligned) {
      TORCH_INTERNAL_ASSERT(inp_or_out_info.size() == 1);
    }

    for (const auto& inp_or_out : inp_or_out_info) {
      const bool is_input = inp_or_out.first;
      const int pos = inp_or_out.second;

      if (is_aligned) {
        auto& pos_list = is_input
            ? vectorized_tensor_info_ptr->aligned_vectorized_inp_tensor_pos
            : vectorized_tensor_info_ptr->aligned_vectorized_out_tensor_pos;
        pos_list.push_back(pos);
      } else {
        auto& map = is_input
            ? vectorized_tensor_info_ptr->inp_misaligned_tensors_pos
            : vectorized_tensor_info_ptr->out_misaligned_tensors_pos;
        map.emplace_back(pos);
      }
    }
  }

  return vectorized_tensor_info_ptr;
}

// Make sure the root domain(s) comprising the vectorized leaf domain
// have the (merged) extent that is divisible by the vectorization
// word size.
void validateAlignedVectorizeExtents(
    const VectorizedSetInfo& info,
    kir::ExpressionEvaluator& expr_eval) {
  int64_t vectorized_merged_domain_extent = 1;
  for (auto id : info.contig_root_ids) {
    auto extent_val = expr_eval.evaluate(id->extent());
    TORCH_INTERNAL_ASSERT(
        extent_val.has_value(),
        "Error vectorizing, ",
        info.consumer_tv->toString(),
        " as the extent of a vectorized root domain, ",
        id->toString(),
        ", is unknown.");
    vectorized_merged_domain_extent *= extent_val.value();
  }

  TORCH_INTERNAL_ASSERT(
      vectorized_merged_domain_extent % info.word_size == 0,
      "Error vectorizing, ",
      info.consumer_tv->toString(),
      " as the extent of the indexed domain, ",
      vectorized_merged_domain_extent,
      ", is not divisible by vector word size ",
      info.word_size);
}

void validateAlignedVectorizedFusionInputOutput(
    const IValue& aten_val,
    int word_size,
    TensorView* tv) {
  TORCH_INTERNAL_ASSERT(aten_val.isTensor());

  const auto& aten_tensor = aten_val.toTensor();

  TORCH_INTERNAL_ASSERT(
      reinterpret_cast<size_t>(aten_tensor.data_ptr()) %
              (word_size * aten_tensor.dtype().itemsize()) ==
          0,
      "Vectorization of ",
      tv->toString(),
      " not possible as the memory address is not aligned. ",
      "Address: ",
      aten_tensor.data_ptr(),
      ", vector word size: ",
      word_size,
      ", data type: ",
      aten_tensor.dtype());

  // Traverse strides from the right-most domains. The rightmost
  // domain must have stride 1.
  int64_t cur_contig_stride = 1;
  bool still_rightmost = true;
  for (auto i = aten_tensor.ndimension() - 1; i >= 0; --i) {
    const auto stride = aten_tensor.strides().at(i);
    // If this domain is contiguous, then not necessary to check the
    // stride. Otherwise, stride must be 1 if it's rightmost or
    // divisible by word_size.
    TORCH_INTERNAL_ASSERT(
        stride == cur_contig_stride || (still_rightmost && stride == 1) ||
            (!still_rightmost && stride % word_size == 0),
        "Vectorization of ",
        tv->toString(),
        " with word size ",
        word_size,
        " not possible due to invalid stride.",
        " Domain: ",
        tv->axis(i)->toString(),
        ", stride: ",
        stride)
    // If the domain is size-1, the next domain is still considered
    // rightmost.
    const auto size = aten_tensor.sizes().at(i);
    still_rightmost = still_rightmost && size == 1;
    cur_contig_stride = stride * size;
  }
}

void validateAlignedVectorizedTensors(
    kir::Kernel* kernel,
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    kir::ExpressionEvaluator& expr_eval) {
  auto tensor_vectorization_validation_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::VectorizedTensorValidation>(
          data_cache, [kernel]() {
            return executor_utils::getVectorizedTensorValidationInfo(kernel);
          });

  // Verify extents of aligned vectorized tensors
  for (const auto& vec_info : kernel->summary().vectorized_set_info) {
    if (vec_info.vectorized_leaf_id->getParallelType() ==
        ParallelType::Vectorize) {
      validateAlignedVectorizeExtents(vec_info, expr_eval);
    }
  }

  // Validate input and output tensors with aligend
  // vectorization.
  for (auto pos : tensor_vectorization_validation_entry.get()
                      .aligned_vectorized_inp_tensor_pos) {
    auto tv = kernel->inputs().at(pos)->as<TensorView>();
    auto word_size = kernel->summary().vectorized_accesses.at(tv);
    validateAlignedVectorizedFusionInputOutput(inputs[pos], word_size, tv);
  }

  if (!outputs.empty()) {
    for (auto pos : tensor_vectorization_validation_entry.get()
                        .aligned_vectorized_out_tensor_pos) {
      auto tv = kernel->outputs().at(pos)->as<TensorView>();
      auto word_size = kernel->summary().vectorized_accesses.at(tv);
      validateAlignedVectorizedFusionInputOutput(outputs[pos], word_size, tv);
    }
  }
}

// Misaligned vectorization check. Currently misaligned vectorization is limited
// to global-register and register-global load/store patterns. However, this
// could be improved to include shared memory.
void validateMisalignedVectorizedTensors(
    kir::Kernel* kernel,
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    kir::ExpressionEvaluator& expr_eval) {
  auto tensor_vectorization_validation_entry =
      executor_utils::caching::ExecutorCompileTimeEntry<
          executor_utils::caching::VectorizedTensorValidation>(
          data_cache, [kernel]() {
            return executor_utils::getVectorizedTensorValidationInfo(kernel);
          });

  std::vector<c10::IValue> inp_misaligned_tensors;
  std::vector<c10::IValue> out_misaligned_tensors;

  const auto& inp_misaligned_tensors_pos =
      tensor_vectorization_validation_entry.get().inp_misaligned_tensors_pos;
  inp_misaligned_tensors.reserve(inp_misaligned_tensors_pos.size());
  std::transform(
      inp_misaligned_tensors_pos.begin(),
      inp_misaligned_tensors_pos.end(),
      std::back_inserter(inp_misaligned_tensors),
      [&inputs](int idx) { return inputs[idx]; });

  const auto& out_misaligned_tensors_pos =
      tensor_vectorization_validation_entry.get().out_misaligned_tensors_pos;
  if (outputs.size() > 0) {
    out_misaligned_tensors.reserve(out_misaligned_tensors_pos.size());
    std::transform(
        out_misaligned_tensors_pos.begin(),
        out_misaligned_tensors_pos.end(),
        std::back_inserter(out_misaligned_tensors),
        [&outputs](int idx) { return outputs[idx]; });
  }
  // If input stride is non-contiguous + no outputs, return false
  TORCH_INTERNAL_ASSERT(
      checkValidMisalignedTensors(
          tensor_vectorization_validation_entry.get().global_inp_misaligned_tv,
          tensor_vectorization_validation_entry.get().global_out_misaligned_tv,
          inp_misaligned_tensors,
          out_misaligned_tensors),
      "All global tensors must have the same stride for misaligned vectorization.");
}

// Check if there's any split that is non-divisible and vectorized. If
// found, Vectorize is illegal.
void validateVectorizedSplits(
    kir::Kernel* kernel,
    kir::ExpressionEvaluator& expr_eval) {
  for (const auto& extent_factor : kernel->summary().splits_to_validate) {
    auto input_extent = expr_eval.evaluate(extent_factor.first);
    auto split_factor = expr_eval.evaluate(extent_factor.second);
    TORCH_INTERNAL_ASSERT(
        input_extent.has_value(),
        "Could not check if a split with vectorization is divisible because the extent, ",
        extent_factor.first->toString(),
        ", is not possible to evaluate.");
    TORCH_INTERNAL_ASSERT(
        input_extent.has_value(),
        "Could not check if a split with vectorization is divisible because the split factor, ",
        extent_factor.second->toString(),
        ", is not possible to evaluate.");
    TORCH_INTERNAL_ASSERT(
        input_extent.value() % split_factor.value() == 0,
        "Non-divisible split with vectorization is detected. ",
        "Extent: ",
        input_extent.value(),
        ". Factor: ",
        split_factor.value());
  }
}

} // namespace

void validateVectorizedTensors(
    kir::Kernel* kernel,
    const at::ArrayRef<IValue>& inputs,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    kir::ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("FusionExecutor::validateVectorizedTensors");

  validateAlignedVectorizedTensors(
      kernel, inputs, outputs, data_cache, expr_eval);

  validateMisalignedVectorizedTensors(
      kernel, inputs, outputs, data_cache, expr_eval);

  validateVectorizedSplits(kernel, expr_eval);
}

kir::ExpressionEvaluator bindKernelInputs(
    const at::ArrayRef<IValue>& aten_inputs,
    kir::Kernel* kernel,
    bool check_consistency) {
  FUSER_PERF_SCOPE("executor_utils::BindKernelInputs");

  TORCH_INTERNAL_ASSERT(
      kernel->inputs().size() == aten_inputs.size(),
      "Something went wrong configuring launch. Inputs no longer match.");

  kir::ExpressionEvaluator expr_eval;
  const auto& inputs = kernel->inputs();

  for (const auto i : c10::irange(inputs.size())) {
    const auto input = inputs[i];

    if (auto tensor_input = dynamic_cast<TensorView*>(input)) {
      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].isTensor(),
          "Something went wrong configuring launch. Inputs no longer match at index:",
          i);

      const auto aten_tensor = aten_inputs[i].toTensor();
      const auto root_domain = TensorDomain::noReductions(
          tensor_input->domain()->getMaybeRFactorDomain());
      TORCH_INTERNAL_ASSERT(
          aten_tensor.ndimension() == static_cast<int>(root_domain.size()),
          "Something went wrong configuring launch. Inputs no longer match.");

      for (const auto dim : c10::irange(root_domain.size())) {
        const auto extent = root_domain[dim]->extent();
        const auto value = aten_tensor.sizes()[dim];
        if (value == 0 && tensor_input->uses().empty()) {
          // If there's no uses, ignore there's a size-0 dimension.
          continue;
        }
        TORCH_INTERNAL_ASSERT(value != 0, "Cannot handle size-0 dimensions");
        bool should_bind = true;
        if (check_consistency) {
          const auto prev_value = expr_eval.evaluate(extent);
          if (prev_value.has_value()) {
            TORCH_CHECK(
                *prev_value == value,
                "Attempting to bind ",
                extent->toString(),
                " to ",
                value,
                "but it's already set to ",
                *prev_value);
            should_bind = false;
          }
        }
        if (should_bind && !extent->isConstScalar()) {
          expr_eval.bind(extent, value);
        }
      }
      // NOLINTNEXTLINE: https://bugs.llvm.org/show_bug.cgi?id=48525
    } else if (input->isScalar() && input->dtype() == DataType::Int) {
      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].type()->kind() == c10::TypeKind::IntType,
          "kernel expected Scalar Int inputs, but found",
          aten_inputs[i].type()->str());
      expr_eval.bind(input, aten_inputs[i].toInt());
    }
  }

  return expr_eval;
}

ExpressionEvaluator bindFusionInputs(
    const at::ArrayRef<IValue>& aten_inputs,
    Fusion* fusion) {
  FUSER_PERF_SCOPE("executor_utils::BindFusionInputs");

  TORCH_INTERNAL_ASSERT(
      fusion->inputs().size() == aten_inputs.size(),
      "Something went wrong configuring launch. Inputs do not match.");

  ExpressionEvaluator evaluator(fusion);
  auto inputs = fusion->inputs();

  // This should probably move to EvaluationContext as we may want to bind
  // input values frequently. Bind fusion input values to runtime values.
  for (const auto i : c10::irange(fusion->inputs().size())) {
    if (inputs[i]->getValType() == ValType::TensorView) {
      TensorView* cg_tensor = inputs[i]->as<TensorView>();

      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].isTensor(),
          "Something went wrong configuring launch. Inputs do not match.");

      auto aten_tensor = aten_inputs[i].toTensor();
      auto root_dom =
          TensorDomain::noReductions(cg_tensor->getMaybeRFactorDomain());
      TORCH_INTERNAL_ASSERT(
          aten_tensor.ndimension() == (int64_t)root_dom.size(),
          "Something went wrong configuring launch. Inputs do not match.");
      for (const auto dim : c10::irange(root_dom.size())) {
        const auto extent = root_dom[dim]->extent();
        const auto value = aten_tensor.sizes()[dim];
        if (value == 0 && cg_tensor->uses().empty()) {
          // If there's no uses, ignore there's a size-0 dimension.
          continue;
        }
        TORCH_INTERNAL_ASSERT(value != 0, "Cannot handle size-0 dimensions");
        const auto prev_value = evaluator.evaluate(extent);
        if (prev_value.has_value()) {
          TORCH_CHECK(
              *prev_value == value,
              "Attempting to bind ",
              extent,
              " to ",
              value,
              "but it's already set to ",
              *prev_value);
        } else {
          evaluator.bind(extent, value);
        }
      }
    } else if (
        inputs[i]->getValType().value() == ValType::Scalar &&
        inputs[i]->getDataType().value() == DataType::Int) {
      TORCH_INTERNAL_ASSERT(
          aten_inputs[i].type()->kind() == c10::TypeKind::IntType,
          "fusion expected Scalar Int inputs, but found",
          aten_inputs[i].type()->str());
      evaluator.bind(inputs[i], aten_inputs[i].toInt());
    }
  }
  return evaluator;
}

void initializeCudaContext() {
  // lazily construct context if non-existing yet;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(nullptr);
  }
}

NvrtcFunction nvrtcCompile(
    const std::string& code,
    const std::string& func_name,
    int id,
    c10::optional<int> opt_block_size) {
  FUSER_PERF_SCOPE("executor_utils::NVRTC");
  initializeCudaContext();

  const auto prop = at::cuda::getCurrentDeviceProperties();

  int major = 0, minor = 0;
  bool compile_to_sass = false;
  codegenOutputQuery(prop, major, minor, compile_to_sass);

  nvrtcProgram program; // NOLINT(cppcoreguidelines-init-variables)

  {
    FUSER_PERF_SCOPE("executor_utils::NvrtcCreateProgram");
    AT_CUDA_NVRTC_CHECK(at::globalContext().getNVRTC().nvrtcCreateProgram(
        &program, code.c_str(), nullptr, 0, nullptr, nullptr));
  }

  ResourceGuard holdProgram([&] {
    FUSER_PERF_SCOPE("executor_utils::NvrtcDestroyProgram");
    AT_CUDA_NVRTC_CHECK(
        at::globalContext().getNVRTC().nvrtcDestroyProgram(&program));
  });

#ifdef __HIP_PLATFORM_HCC__
  std::vector<const char*> args = {"--std=c++14"};
#if ROCM_VERSION >= 40200
  args.push_back("-hip-pch");
#endif
#else
#if CUDA_VERSION < 11010
  // compile to sass is not allowed prior to CUDA 11.1
  compile_to_sass = false;
#endif
  // CUDA 11.1 allows going directly to SASS (sm_) instead of PTX (compute_)
  // which gives better backwards compatibility to work on older driver,
  // (since older driver doesn't necessrily recognize PTX emitted by new
  // toolkit);
  // Meanwhile, for forward compatibility (future device with
  // `unsupported_arch==True`), since SASS are not necessarily compatible,
  // we fallback to PTX instead.
  const std::string compute = std::string("--gpu-architecture=") +
      (compile_to_sass ? "sm_" : "compute_") + std::to_string(major) +
      std::to_string(minor);
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<const char*> args = {
      "--std=c++14", compute.c_str(), "-default-device"};
#endif

  const char* disable_fma = getenv("PYTORCH_NVFUSER_DISABLE_FMA");
#ifdef __HIP_PLATFORM_HCC__
  if (disable_fma && atoi(disable_fma)) {
    TORCH_WARN_ONCE(
        "PYTORCH_CUDA_FUSER_DISABLE_FMA is not supported on ROCm, ignoring");
  }
#else
  if (disable_fma && atoi(disable_fma)) {
    args.push_back("--fmad=false");
  } else {
    args.push_back("--fmad=true");
  }
#endif

#ifndef NDEBUG
  // Add line info to generated kernels
  args.push_back("-lineinfo");
#else
  // Avoid excessive register usage from assertion
  args.push_back("-DNDEBUG");
#endif

  const char* ptxas_opt_level = getenv("PYTORCH_NVFUSER_JIT_OPT_LEVEL");
  std::string jit_opt_level = "-O";

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<CUjit_option> options;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<void*> option_vals;
  std::vector<char> info_log;
  unsigned int log_size = 8196;

  if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
    // show register usage in compilation log
    if (compile_to_sass) {
      args.push_back("--ptxas-options");
      args.push_back("--verbose");
    } else {
      options.push_back(CU_JIT_LOG_VERBOSE);
      option_vals.push_back((void*)1);
      info_log.reserve(log_size);

      options.push_back(CU_JIT_INFO_LOG_BUFFER);
      option_vals.push_back((void*)info_log.data());

      options.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
      option_vals.push_back((void*)(long)log_size);
    }
  }

  if (ptxas_opt_level) {
    int val = atoi(ptxas_opt_level);
    if (val <= 4 && val >= 0) {
      if (compile_to_sass) {
        jit_opt_level += std::to_string(val);
        args.push_back("--ptxas-options");
        args.push_back(jit_opt_level.c_str());
      } else {
        options.push_back(CU_JIT_OPTIMIZATION_LEVEL);
        option_vals.push_back((void*)(intptr_t)val);
      }
    } else {
      TORCH_WARN_ONCE(
          "acceptable range for PYTORCH_NVFUSER_JIT_OPT_LEVEL is between 0 and 4, but received ",
          val,
          ", ignoring the option");
    }
  }

#ifndef USE_ROCM
  // keeping the string outside the loop for lifetime
  std::string max_register_usage = "--maxrregcount=";
  uint32_t max_register = 0;
  if (opt_block_size.has_value() && opt_block_size.value() > 0) {
    int num_partition = 0;
    int reg_allocation_granularity = 0;
    cudaOccDeviceProp occ_prop(*prop);
    cudaOccSubPartitionsPerMultiprocessor(&num_partition, &occ_prop);
    cudaOccRegAllocationGranularity(&reg_allocation_granularity, &occ_prop);
    int warp_size = prop->warpSize;
    int num_warps = ceilDiv(opt_block_size.value(), warp_size);

    // warps could be distributed unevenly across partition
    int max_warps_per_sm_partition = ceilDiv(num_warps, num_partition);
    // registers are evenly distributed across partitions, partition with most
    // wraps determins the maximum register available per warp
    int max_reg_per_warp =
        prop->regsPerBlock / num_partition / max_warps_per_sm_partition;
    // clamp down to register allocation granularity at warp level
    int effective_max_reg_per_warp = max_reg_per_warp /
        reg_allocation_granularity * reg_allocation_granularity;
    // The maximum possible count allowed by ptxas is 255
    max_register = static_cast<uint32_t>(
        std::min(effective_max_reg_per_warp / warp_size, 255));

    if (compile_to_sass) {
      max_register_usage += std::to_string(max_register);
      args.push_back("--ptxas-options");
      args.push_back(max_register_usage.c_str());
    } else {
      options.push_back(CU_JIT_MAX_REGISTERS);
      option_vals.push_back((void*)(intptr_t)max_register);
    }
  }
#endif

  at::globalContext().getNVRTC().nvrtcAddNameExpression(
      program, func_name.c_str());

  {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::CompileProgram");

    const auto result = at::globalContext().getNVRTC().nvrtcCompileProgram(
        program, args.size(), args.data());

    if (result != NVRTC_SUCCESS) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t logsize;
      at::globalContext().getNVRTC().nvrtcGetProgramLogSize(program, &logsize);
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      std::vector<char> log(logsize);
      at::globalContext().getNVRTC().nvrtcGetProgramLog(program, log.data());

      TORCH_INTERNAL_ASSERT(
          false, code.c_str(), "\nCUDA NVRTC compile error: ", log.data());
    } else if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t logsize;
      at::globalContext().getNVRTC().nvrtcGetProgramLogSize(program, &logsize);
      std::vector<char> log(logsize);
      at::globalContext().getNVRTC().nvrtcGetProgramLog(program, log.data());

      std::cout << log.data() << std::endl;
    }

    AT_CUDA_NVRTC_CHECK(result);
  }

  const char* lowered_kernel_name = nullptr;
  at::globalContext().getNVRTC().nvrtcGetLoweredName(
      program, func_name.c_str(), &lowered_kernel_name);

  size_t ptx_size = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<char> ptx;

  {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::GetPTX");
#if CUDA_VERSION >= 11010
    // compile_to_sass determines whether we are generating SASS or PTX, hence
    // the different API.
    const auto getSize = compile_to_sass
        ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
        : at::globalContext().getNVRTC().nvrtcGetPTXSize;
    const auto getFunc = compile_to_sass
        ? at::globalContext().getNVRTC().nvrtcGetCUBIN
        : at::globalContext().getNVRTC().nvrtcGetPTX;
#else
    const auto getSize = at::globalContext().getNVRTC().nvrtcGetPTXSize;
    const auto getFunc = at::globalContext().getNVRTC().nvrtcGetPTX;
#endif
    AT_CUDA_NVRTC_CHECK(getSize(program, &ptx_size));
    ptx.resize(ptx_size);
    AT_CUDA_NVRTC_CHECK(getFunc(program, ptx.data()));
  }

  NvrtcFunction compiled_kernel_;

  // TODO: We do go through different code path, should investigate whether this
  // has an impact on generated binary.
#ifndef __HIP_PLATFORM_HCC__
  const char* prefix_env = getenv("PYTORCH_NVFUSER_CUBIN");
  if (prefix_env) {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::LoadCUBIN");

    // Output ptx file
    std::stringstream output_file_name;
    output_file_name << prefix_env << "_" << id
                     << (compile_to_sass ? ".cubin" : ".ptx");
    std::ofstream outputFile(output_file_name.str().c_str(), std::ios::out);
    if (outputFile.is_open()) {
      outputFile.write(ptx.data(), ptx.size());
      outputFile.close();
    }

    if (compile_to_sass) {
      FUSER_PERF_SCOPE("executor_utils::Nvrtc::LoadPTX");

      // load sass directly
      AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadDataEx(
          &(compiled_kernel_.module),
          ptx.data(),
          options.size(),
          options.data(),
          option_vals.data()));
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      CUlinkState linkState;

      AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkCreate(
          // 0, nullptr, nullptr, &linkState));
          options.size(),
          options.data(),
          option_vals.data(),
          &linkState));

      AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkAddData(
          linkState,
          CU_JIT_INPUT_PTX,
          ptx.data(),
          ptx_size,
          "compiling PTX",
          0,
          nullptr,
          nullptr));

      if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
        std::cout << info_log.data() << std::endl;
      }

      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      size_t cubinSize;
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      void* cubin;
      AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuLinkComplete(
          linkState, &cubin, &cubinSize));

      // Output binary file
      std::stringstream cubin_file_name;
      cubin_file_name << prefix_env << "_" << id << ".cubin";

      std::ofstream myCubinFile(
          cubin_file_name.str().c_str(), std::ios::out | std::ios::binary);

      if (myCubinFile.is_open()) {
        myCubinFile.write(static_cast<const char*>(cubin), cubinSize);
        myCubinFile.close();
      }
      // load compiled cubin
      // AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadData(
      //     &(compiled_kernel_.module), cubin));
      AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadDataEx(
          &(compiled_kernel_.module),
          cubin,
          options.size(),
          options.data(),
          option_vals.data()));
    }
  } else {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::LoadPTX");

    // load ptx directly
    AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadDataEx(
        &(compiled_kernel_.module),
        ptx.data(),
        options.size(),
        options.data(),
        option_vals.data()));

    if (!compile_to_sass &&
        isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
      std::cout << info_log.data() << std::endl;
    }
  }
#else
  // load ptx directly
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleLoadData(
      &(compiled_kernel_.module), ptx.data()));

#endif
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuModuleGetFunction(
      &(compiled_kernel_.function),
      compiled_kernel_.module,
      lowered_kernel_name));

  return compiled_kernel_;
}

namespace caching {

//! CompileTimeInfo is the actual subclass of CompileTimeInfoBase that will
//!  be stored in the data cache. It owns a data_ state internally of the
//!  dataType defined within the entry class, which are listed in header file.
template <typename EntryClass>
class CompileTimeInfo : public CompileTimeInfoBase {
 public:
  CompileTimeInfo(std::unique_ptr<typename EntryClass::DataType> data)
      : CompileTimeInfoBase(EntryClass::EntryType), data_(std::move(data)) {}

  typename EntryClass::DataType* get() {
    return data_.get();
  }

 private:
  std::unique_ptr<typename EntryClass::DataType> data_;
};

void ExecutorCompileTimeInfoCache::insert(EntryOwningPtr new_entry) {
  // Just overwrite when insertion duplicates, equality not checked.
  entry_type_map_[new_entry->type()] = new_entry.get();
  entries_.emplace_back(std::move(new_entry));
}

template <typename EntryClass>
ExecutorCompileTimeEntry<EntryClass>::ExecutorCompileTimeEntry(
    ExecutorCompileTimeInfoCache* data_cache,
    MakerFnType fn) {
  using InfoType = CompileTimeInfo<EntryClass>;

  if (!data_cache || !data_cache->has(EntryClass::EntryType)) {
    owned_data_ = fn();
    data_ptr_ = owned_data_.get();

    if (data_cache) {
      std::unique_ptr<CompileTimeInfoBase> new_entry =
          std::make_unique<InfoType>(std::move(owned_data_));
      data_cache->insert(std::move(new_entry));
    }
  } else {
    data_ptr_ =
        data_cache->at(EntryClass::EntryType)->template as<InfoType>()->get();
  }
}

// Template instantiation
template class ExecutorCompileTimeEntry<ParallelBindingIterDomains>;
template class ExecutorCompileTimeEntry<ParallelIterExtentMap>;
template class ExecutorCompileTimeEntry<SimplifiedParallelIterExtentMap>;
template class ExecutorCompileTimeEntry<WarpPaddedParallelExtents>;
template class ExecutorCompileTimeEntry<VectorizedTensorValidation>;
template class ExecutorCompileTimeEntry<InputAliasIndices>;
template class ExecutorCompileTimeEntry<OutputAliasIndices>;

} // namespace caching

std::vector<IterDomain*> getParallelBindingsIterDomains(
    GpuLower* lower,
    const std::vector<TensorView*>& used_tvs) {
  std::vector<IterDomain*> parallel_ids;
  for (auto tv : used_tvs) {
    for (auto id : tv->domain()->domain()) {
      if (id->isThread()) {
        if (id->isBroadcast()) {
          // Want to keep the broadcast dimensions if they are not resolved
          // TODO: piping down the parallel dimension map here would
          //  be helpful
          auto& parallel_map = lower->caParallelMap();
          if (parallel_map.getConcreteMappedID(id) == id) {
            parallel_ids.push_back(id);
          }
        } else {
          // Non broadcast ids are directly added to the binding
          //  ids.
          parallel_ids.push_back(id);
        }
      }
    }
  }
  return parallel_ids;
}

namespace {

void insertParallelExtent(
    IterDomain* binding_id,
    const std::unique_ptr<ParallelExtentMap>& parallel_iter_extents_ptr) {
  auto extent = binding_id->extent();
  const auto it =
      parallel_iter_extents_ptr->find(binding_id->getParallelType());
  if (it != parallel_iter_extents_ptr->end()) {
    it->second.push_back(extent);
  } else {
    parallel_iter_extents_ptr->operator[](binding_id->getParallelType()) = {
        extent};
  }
}

} // namespace

std::unique_ptr<ParallelExtentMap> getParallelIterExtents(
    std::vector<IterDomain*>& parallel_binding_ids) {
  auto parallel_iter_extents_ptr = std::make_unique<ParallelExtentMap>();
  for (auto id : parallel_binding_ids) {
    insertParallelExtent(id, parallel_iter_extents_ptr);
  }

  return parallel_iter_extents_ptr;
}

std::unique_ptr<ParallelExtentMap> getSimplifiedParallelIterExtents(
    GpuLower* lower,
    std::vector<IterDomain*>& parallel_binding_ids) {
  auto parallel_iter_extents_ptr = std::make_unique<ParallelExtentMap>();
  auto& parallel_map = lower->caParallelMap();
  std::vector<IterDomain*> mapped;
  bool is_tidx_warp_padded = lower->getWarpPaddedParallelInfo().is_tidx_padded;

  for (auto id : parallel_binding_ids) {
    if (std::any_of(
            mapped.begin(),
            mapped.end(),
            [id, &parallel_map](IterDomain* mapped_id) {
              return parallel_map.areMapped(mapped_id, id);
            })) {
      if (id->getParallelType() != ParallelType::TIDx || !is_tidx_warp_padded) {
        continue;
      }
    }

    insertParallelExtent(
        parallel_map.getConcreteMappedID(id), parallel_iter_extents_ptr);
    mapped.push_back(id);
  }

  return parallel_iter_extents_ptr;
}

std::unique_ptr<caching::WarpPaddedExtentsInfo> getWarpPaddedExtentsInfo(
    kir::Kernel* kernel,
    std::vector<IterDomain*>& parallel_binding_ids) {
  auto warp_padded_extent_info_ptr =
      std::make_unique<caching::WarpPaddedExtentsInfo>();
  auto& warp_padded_extent_set =
      warp_padded_extent_info_ptr->warp_padded_extent_set;
  auto& warp_padded_constant =
      warp_padded_extent_info_ptr->warp_padded_constant;
  bool has_warp_reduction =
      kernel->getWarpPaddedParallelInfo().has_warp_reduction;

  for (auto id : parallel_binding_ids) {
    // Apply warp padding only when there're warp reductions in
    //  the kernel.
    if (has_warp_reduction) {
      if (id->hasPaddingToMultipleOfWarp() ||
          kernel->isParallelTypePadded(id->getParallelType())) {
        auto extent = id->extent();
        warp_padded_extent_set.insert(extent);
        auto padded_value = id->getMaybeSizeAfterPadding();
        if (padded_value.has_value()) {
          warp_padded_constant[extent] = padded_value.value();
        }
      }
    }
  }
  return warp_padded_extent_info_ptr;
}

} // namespace executor_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
