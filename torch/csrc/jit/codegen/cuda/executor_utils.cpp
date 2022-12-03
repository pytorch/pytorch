#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>

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
#include <nvfuser_resources/fused_welford_helper.h>
#include <nvfuser_resources/fused_welford_impl.h>
#include <nvfuser_resources/grid_broadcast.h>
#include <nvfuser_resources/grid_reduction.h>
#include <nvfuser_resources/grid_sync.h>
#include <nvfuser_resources/helpers.h>
#include <nvfuser_resources/index_utils.h>
#include <nvfuser_resources/memory.h>
#include <nvfuser_resources/random_numbers.h>
#include <nvfuser_resources/swizzle.h>
#include <nvfuser_resources/tensor.h>
#include <nvfuser_resources/tensorcore.h>
#include <nvfuser_resources/tuple.h>
#include <nvfuser_resources/type_traits.h>
#include <nvfuser_resources/warp.h>
#include <nvfuser_resources/welford.h>

#ifdef USE_ROCM
#include <nvfuser_resources/array_rocm.h>
#include <nvfuser_resources/bf16_support_rocm.h>
#include <nvfuser_resources/block_sync_default_rocm.h>
#include <nvfuser_resources/warp_rocm.h>
#endif

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

#ifndef USE_ROCM
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
  // fp16 support is automatic, bf16 is not
  ss << nvfuser_resources::bf16_support_rocm_cu;
#endif

  // Base classes and helpers
  ss << nvfuser_resources::tensor_cu;
  ss << nvfuser_resources::type_traits_cu;
#ifndef USE_ROCM
  ss << nvfuser_resources::array_cu;
#else
  ss << nvfuser_resources::array_rocm_cu;
#endif
  ss << nvfuser_resources::random_numbers_cu;
  ss << nvfuser_resources::helpers_cu;
  ss << nvfuser_resources::index_utils_cu;
  ss << nvfuser_resources::tuple_cu;

  // Synchronization classes
  if (std::getenv("PYTORCH_NVFUSER_USE_BLOCK_SYNC_ATOMIC")) {
    ss << nvfuser_resources::block_sync_atomic_cu;
  } else {
#ifndef USE_ROCM
    ss << nvfuser_resources::block_sync_default_cu;
#else
    ss << nvfuser_resources::block_sync_default_rocm_cu;
#endif
  }
  ss << nvfuser_resources::grid_sync_cu;

  // Communication classes
  ss << nvfuser_resources::block_reduction_cu;
  ss << nvfuser_resources::grid_reduction_cu;
  ss << nvfuser_resources::grid_broadcast_cu;
  ss << nvfuser_resources::broadcast_cu;
  ss << nvfuser_resources::welford_cu;
#ifndef USE_ROCM
  ss << nvfuser_resources::warp_cu;
  ss << nvfuser_resources::tensorcore_cu;
  ss << nvfuser_resources::memory_cu;
#else
  ss << nvfuser_resources::warp_rocm_cu;
#endif
  ss << nvfuser_resources::fused_welford_helper_cu;
  ss << nvfuser_resources::fused_reduction_cu;
  ss << nvfuser_resources::fused_welford_impl_cu;
  ss << nvfuser_resources::swizzle_cu;

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
    msg << "Argument is a CPU tensor which is not supported in fusions.\n";
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
  // TODO: remove this switch with `aten_to_data_type`
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
    const ArgAbstract* arg,
    const Val* param,
    std::stringstream& msg) {
  TORCH_INTERNAL_ASSERT(
      param->getDataType().has_value(), "kernel param should have data type");
  DataType param_type = *param->getDataType();
  bool match = false;
  switch (arg->type()) {
    case ArgType::Long:
      match = param_type == DataType::Int || param_type == DataType::Int32;
      break;
    case ArgType::Double:
      match = param_type == DataType::Double || param_type == DataType::Float ||
          param_type == DataType::Half || param_type == DataType::BFloat16;
      break;
    case ArgType::Bool:
      match = param_type == DataType::Bool;
      break;
    case ArgType::ComplexDouble:
      match = param_type == DataType::ComplexDouble ||
          param_type == DataType::ComplexFloat;
      break;
    default:
      // TODO: We need to verify that param is actually a scalar
      msg << "Argument is not a scalar, but the parameter is."
          << "\n";
      return false;
  }
  if (!match) {
    msg << "Argument type is " << argTypeToString(arg->type())
        << ", but the parameter is " << param_type << "\n";
  }
  return match;
}

// Return false if arg and param don't match up and if arg's device (if a
// tensor) doesn't match provided device
bool validateKernelArg(
    const ArgAbstract* arg,
    const Val* param,
    const c10::Device& device,
    std::stringstream& msg) {
  if (auto tensor_arg_abstract = dynamic_cast<const TensorArgAbstract*>(arg)) {
    // TODO: don't use get tensor here. We would want to remove tensor reference
    // for async compilation
    return validateKernelArgTensor(
        tensor_arg_abstract->getTensor(), param, device, msg);
  } else if (arg->isType(ArgType::CpuScalarTensor)) {
    // TODO: merge this one with above
    // TODO: we need to check cpu scalar dtyp matches param
    bool match = param->as<TensorView>()->isCpuScalar();
    if (!match) {
      msg << "Argument is scalar type, but kernel parameter is not\n";
    }
    return match;
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
    const KernelArgumentHolder& args,
    const c10::Device& device) {
  FUSER_PERF_SCOPE("executor_utils::ValidateKernelInputs");

  // This is necessary as we were traversing the fusion graph later in the check
  FusionGuard fg(fusion);
  // Check inputs
  TORCH_INTERNAL_ASSERT(
      args.size() == fusion->inputs().size(), "Wrong number of kernel inputs.");

  std::stringstream msg;
  bool mismatch = false;
  for (const auto i : c10::irange(args.size())) {
    const ArgAbstract* arg = args[i];
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
    mismatch = !validateKernelArgTensor(arg, param, device, msg) || mismatch;
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
  TORCH_INTERNAL_ASSERT(
      !info.contig_root_ids.empty(),
      "No root ID found for vectorization with ",
      info.consumer_tv->toString(),
      " and ",
      info.producer_tv->toString());

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
    vectorized_merged_domain_extent *= extent_val->as<int64_t>();
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
    const at::Tensor& aten_tensor,
    int word_size,
    TensorView* tv) {
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
    const auto size = aten_tensor.sizes().at(i);
    // If this domain is contiguous or size == 1, then not necessary to check
    // the stride. Otherwise, stride must be 1 if it's rightmost or
    // divisible by word_size
    TORCH_INTERNAL_ASSERT(
        stride == cur_contig_stride || size == 1 ||
            (still_rightmost && stride == 1) ||
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
    still_rightmost = still_rightmost && size == 1;
    // We do not update cur_contig_stride for size==1 dimensions,
    // since we have specialized vectorization stride check for them
    if (size != 1) {
      cur_contig_stride = stride * size;
    }
  }
}

void validateAlignedVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
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
    auto tensor_arg_abstract =
        dynamic_cast<const TensorArgAbstract*>(args[pos]);
    TORCH_INTERNAL_ASSERT(tensor_arg_abstract, "alias io only supports tensor");
    validateAlignedVectorizedFusionInputOutput(
        tensor_arg_abstract->getTensor(), word_size, tv);
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
    const KernelArgumentHolder& args,
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
      [&args](int idx) {
        auto tensor_arg_abstract =
            dynamic_cast<const TensorArgAbstract*>(args[idx]);
        TORCH_INTERNAL_ASSERT(
            tensor_arg_abstract, "alias io only supports tensor");
        return tensor_arg_abstract->getTensor();
      });

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
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    kir::ExpressionEvaluator& expr_eval) {
  FUSER_PERF_SCOPE("FusionExecutor::validateVectorizedTensors");

  validateAlignedVectorizedTensors(
      kernel, args, outputs, data_cache, expr_eval);

  validateMisalignedVectorizedTensors(
      kernel, args, outputs, data_cache, expr_eval);

  validateVectorizedSplits(kernel, expr_eval);
}

namespace {

template <typename EXPR_EVALUATOR>
void bindInputForExprEvaluation(
    Val* val,
    const ArgAbstract* arg,
    bool check_consistency,
    EXPR_EVALUATOR& expr_eval) {
  if (val->getValType() == ValType::TensorView) {
    TensorView* cg_tensor = val->as<TensorView>();
    auto root_domain =
        TensorDomain::noReductions(cg_tensor->getMaybeRFactorDomain());

    if (root_domain.size() == 0) {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::CpuScalarTensor) ||
              (arg->isType(ArgType::Tensor) &&
               dynamic_cast<const TensorArgAbstract*>(arg)->getRank() == 0),
          "Something went wrong configuring launch. Inputs is not rank 0 tensor");
    } else {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::Tensor),
          "Something went wrong configuring launch. Inputs do not match.");

      auto tensor_arg_abstract = dynamic_cast<const TensorArgAbstract*>(arg);
      TORCH_INTERNAL_ASSERT(
          tensor_arg_abstract &&
              tensor_arg_abstract->getRank() == (int64_t)root_domain.size(),
          "Something went wrong configuring launch. Inputs rank does not match.");

      for (const auto dim : c10::irange(root_domain.size())) {
        const auto tensor_arg_size = tensor_arg_abstract->getSize(dim);
        const auto tensor_arg_stride = tensor_arg_abstract->getStride(dim);
        const auto extent = root_domain[dim]->extent();
        if (root_domain[dim]->hasExpandedExtent()) {
          TORCH_INTERNAL_ASSERT(
              tensor_arg_stride == 0,
              "Expecting an expanded dimension on dimension ",
              dim,
              " but found stride ",
              tensor_arg_stride);
          // Could support dynamic size on expanded dimension, so may not have
          // an inferable expanded extent here. This check might be better to do
          // once all values are bound.
          auto maybe_expanded_size =
              expr_eval.evaluate(root_domain[dim]->expandedExtent());
          if (maybe_expanded_size.has_value()) {
            TORCH_CHECK(
                *maybe_expanded_size == tensor_arg_size,
                "Expecting expanded extent of ",
                *maybe_expanded_size,
                " but received value of ",
                tensor_arg_size);
          }
        }

        const auto value =
            root_domain[dim]->hasExpandedExtent() ? 1 : tensor_arg_size;
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
                " but it's already set to ",
                *prev_value);
            should_bind = false;
          }
        }
        if (should_bind && !extent->isConstScalar()) {
          expr_eval.bind(extent, value);
        }
      }
    }
  } else if (val->getValType().value() == ValType::Scalar) {
    if (val->getDataType().value() == DataType::Int) {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::Long),
          "fusion expected Scalar Int inputs, but found ",
          argTypeToString(arg->type()));
      expr_eval.bind(val, *static_cast<const int64_t*>(arg->arg()));
    } else if (val->getDataType().value() == DataType::Double) {
      TORCH_INTERNAL_ASSERT(
          arg->isType(ArgType::Double),
          "fusion expected Scalar Double inputs, but found ",
          argTypeToString(arg->type()));
      expr_eval.bind(val, *static_cast<const double*>(arg->arg()));
    }
  }
}

} // namespace

kir::ExpressionEvaluator bindKernelInputs(
    const KernelArgumentHolder& args,
    kir::Kernel* kernel,
    bool check_consistency) {
  FUSER_PERF_SCOPE("executor_utils::BindKernelInputs");

  TORCH_INTERNAL_ASSERT(
      kernel->inputs().size() == args.size(),
      "Something went wrong configuring launch. Inputs no longer match.");

  kir::ExpressionEvaluator expr_eval;
  const auto& inputs = kernel->inputs();

  for (const auto i : c10::irange(inputs.size())) {
    bindInputForExprEvaluation(
        inputs[i], args[i], check_consistency, expr_eval);
  }
  return expr_eval;
}

ExpressionEvaluator bindFusionInputs(
    const KernelArgumentHolder& args,
    Fusion* fusion) {
  FUSER_PERF_SCOPE("executor_utils::BindFusionInputs");

  auto inputs = fusion->inputs();
  TORCH_INTERNAL_ASSERT(
      inputs.size() == args.size(),
      "Something went wrong configuring launch. Inputs do not match.\n",
      "inputs: ",
      ir_utils::toString(inputs),
      " args size: ",
      args.size());

  ExpressionEvaluator expr_eval(fusion);

  // This should probably move to EvaluationContext as we may want to bind
  // input values frequently. Bind fusion input values to runtime values.
  for (const auto i : c10::irange(inputs.size())) {
    bindInputForExprEvaluation(inputs[i], args[i], true, expr_eval);
  }
  return expr_eval;
}

void initializeCudaContext() {
  // lazily construct context if non-existing yet;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  CUcontext pctx = nullptr;
  AT_CUDA_DRIVER_CHECK(at::globalContext().getNVRTC().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::getFreeMutex()));
    C10_CUDA_CHECK(cudaFree(nullptr));
  }
}

namespace {

// Dump PTX or CUBIN to a file
#if CUDA_VERSION >= 11010
void dumpCompiledCodeToFile(
    const nvrtcProgram& program,
    int fusion_id,
    bool dump_cubin) {
  const auto getSize = dump_cubin
      ? at::globalContext().getNVRTC().nvrtcGetCUBINSize
      : at::globalContext().getNVRTC().nvrtcGetPTXSize;
  const auto getCode = dump_cubin ? at::globalContext().getNVRTC().nvrtcGetCUBIN
                                  : at::globalContext().getNVRTC().nvrtcGetPTX;
  size_t size = 0;
  AT_CUDA_NVRTC_CHECK(getSize(program, &size));
  std::vector<char> code(size);
  AT_CUDA_NVRTC_CHECK(getCode(program, code.data()));
  std::stringstream file_name;
  file_name << "__tmp_kernel" << fusion_id << "."
            << (dump_cubin ? "cubin" : "ptx");
  std::cout << "PRINTING: " << file_name.str() << std::endl;
  std::ofstream out(file_name.str());
  TORCH_INTERNAL_ASSERT(out.is_open());
  out.write(code.data(), size);
  out.close();
}
#endif

} // namespace

std::pair<NvrtcFunction, std::string> nvrtcCompile(
    const std::string& code,
    const std::string& func_name,
    int id,
    c10::optional<int> opt_block_size) {
  FUSER_PERF_SCOPE("executor_utils::NVRTC");
  if (isOptionDisabled(DisableOption::ArchCheck)) {
    TORCH_WARN(
        "NVFuser Compile: arch check disabled, should not compile any kernel");
  }

  initializeCudaContext();

  std::stringstream ptxas_log;

  const auto prop = at::cuda::getCurrentDeviceProperties();

  int major = 0, minor = 0;
  bool compile_to_sass = false;
  codegenOutputQuery(prop, major, minor, compile_to_sass);

  nvrtcProgram program; // NOLINT(cppcoreguidelines-init-variables)

  {
    std::stringstream ss;
    ss << "__tmp_kernel" << id << ".cu";
    std::string name = ss.str();
    FUSER_PERF_SCOPE("executor_utils::NvrtcCreateProgram");
    AT_CUDA_NVRTC_CHECK(at::globalContext().getNVRTC().nvrtcCreateProgram(
        &program, code.c_str(), name.c_str(), 0, nullptr, nullptr));
  }

  ResourceGuard holdProgram([&] {
    FUSER_PERF_SCOPE("executor_utils::NvrtcDestroyProgram");
    AT_CUDA_NVRTC_CHECK(
        at::globalContext().getNVRTC().nvrtcDestroyProgram(&program));
  });

#ifdef USE_ROCM
  std::vector<const char*> args = {"--std=c++14"};
#if ROCM_VERSION >= 40200
  args.push_back("-hip-pch");
#endif
#else
#if CUDA_VERSION < 11010
  // compile to sass is not allowed prior to CUDA 11.1
  compile_to_sass = false;
#endif

  if (isOptionDisabled(DisableOption::CompileToSass)) {
    // Allows manually disabling compilation to sass
    //  so the intermediate ptx could be checked.
    compile_to_sass = false;
  }
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

  const bool disable_fma = isOptionDisabled(DisableOption::Fma);
#ifdef USE_ROCM
  if (disable_fma) {
    TORCH_WARN_ONCE(
        "PYTORCH_CUDA_FUSER_DISABLE_FMA is not supported on ROCm, ignoring");
  }
#else
  if (disable_fma) {
    args.push_back("--fmad=false");
  } else {
    args.push_back("--fmad=true");
  }
#endif
  // Add line info to generated kernels
  if (isDebugDumpEnabled(DebugDumpOption::DebugInfo)) {
    args.push_back("-lineinfo");
  }
#ifdef NDEBUG
  // Avoid excessive register usage from assertion
  args.push_back("-DNDEBUG");
#endif

  if (isOptionEnabled(EnableOption::KernelProfile)) {
    args.push_back("-DPYTORCH_NVFUSER_PROFILE_KERNEL");
  }

  const char* ptxas_opt_level = getenv("PYTORCH_NVFUSER_JIT_OPT_LEVEL");
  std::string jit_opt_level = "-O";

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<CUjit_option> options;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<void*> option_vals;
  std::vector<char> info_log;
  unsigned int log_size = 8196;

  if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog) ||
      isDebugDumpEnabled(DebugDumpOption::PerfDebugVerbose)) {
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
      if (val < 4) {
        TORCH_WARN(
            "ptxas optimization level manually set as ",
            val,
            ", which could negatively affect performance. Try removing env variable PYTORCH_NVFUSER_JIT_OPT_LEVEL for optimal performance.");
      }
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

    ptxas_log << "\nCompile options: ";
    for (auto arg : args) {
      ptxas_log << arg << " ";
    }
    ptxas_log << " ; block size=" << opt_block_size.value() << "\n";
  }
#endif

  at::globalContext().getNVRTC().nvrtcAddNameExpression(
      program, func_name.c_str());

  {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::CompileProgram");

    const auto result = at::globalContext().getNVRTC().nvrtcCompileProgram(
        program, args.size(), args.data());

    size_t logsize = 0;
    at::globalContext().getNVRTC().nvrtcGetProgramLogSize(program, &logsize);

    std::vector<char> log(logsize);
    at::globalContext().getNVRTC().nvrtcGetProgramLog(program, log.data());

    if (result != NVRTC_SUCCESS) {
      TORCH_INTERNAL_ASSERT(
          false, code.c_str(), "\nCUDA NVRTC compile error: ", log.data());
    }

    ptxas_log << log.data() << std::endl;
    if (isDebugDumpEnabled(DebugDumpOption::PrintPtxasLog)) {
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

#ifndef USE_ROCM

#if CUDA_VERSION >= 11010
  if (isDebugDumpEnabled(DebugDumpOption::Ptx)) {
    dumpCompiledCodeToFile(program, id, false);
  }

  if (isDebugDumpEnabled(DebugDumpOption::Cubin)) {
    TORCH_INTERNAL_ASSERT(
        compile_to_sass,
        "CUBIN not available as the kernel was compiled only to PTX");
    dumpCompiledCodeToFile(program, id, true);
  }
#endif

  {
    FUSER_PERF_SCOPE("executor_utils::Nvrtc::LoadPTX");

    // load ptx or cubin directly
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

  TORCH_CHECK(
      !isOptionDisabled(DisableOption::ArchCheck),
      "NVFuser Compile: arch check disabled, should not return any compiled kernel");

  return {compiled_kernel_, ptxas_log.str()};
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
          if (lower->caMap()->getConcreteMappedID(id, IdMappingMode::LOOP) ==
              id) {
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
  const auto& ca_map = lower->caMap();
  std::vector<IterDomain*> mapped;
  bool is_tidx_warp_padded = lower->getWarpPaddedParallelInfo().is_tidx_padded;

  for (auto id : parallel_binding_ids) {
    if (std::any_of(
            mapped.begin(), mapped.end(), [id, &ca_map](IterDomain* mapped_id) {
              return ca_map->areMapped(mapped_id, id, IdMappingMode::LOOP);
            })) {
      if (id->getParallelType() != ParallelType::TIDx || !is_tidx_warp_padded) {
        continue;
      }
    }

    insertParallelExtent(
        ca_map->getConcreteMappedID(id, IdMappingMode::LOOP),
        parallel_iter_extents_ptr);
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
