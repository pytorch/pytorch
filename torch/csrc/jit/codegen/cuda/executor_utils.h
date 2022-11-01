#pragma once

#include <ATen/core/ivalue.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <cuda.h>

#include <torch/csrc/jit/ir/ir.h>

#include <torch/csrc/jit/codegen/cuda/executor_kernel_arg.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>

#include <string>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace executor_utils {

// Include all the functions we might need in generated code
std::string kernelPreamble();

void validateKernelInputs(
    Fusion* fusion,
    const KernelArgumentHolder& args,
    const c10::Device& device);

void validateKernelOutputs(
    Fusion* fusion,
    const std::vector<at::Tensor>& outputs,
    const c10::Device& device);

//! Bind kernel input values to runtime values
ExpressionEvaluator bindKernelInputs(
    const KernelArgumentHolder& args,
    kir::Kernel* kernel,
    bool check_consistency = true);

//! Bind fusion input values to runtime values
TORCH_CUDA_CU_API ExpressionEvaluator
bindFusionInputs(const KernelArgumentHolder& args, Fusion* fusion);

struct NvrtcFunction {
  CUmodule module = CUmodule();
  CUfunction function = CUfunction();
};

void initializeCudaContext();

// Returns executable function and the ptxas log from compilation
std::pair<NvrtcFunction, std::string> nvrtcCompile(
    const std::string& code,
    const std::string& func_name,
    int id,
    c10::optional<int> opt_block_size = c10::nullopt);

namespace caching {
// TODO: Could consider putting some of
//  the logic in the common space and re-use

//! List of all the possible entry types in
//!  `FusionExecutor` compile-time data cache.
enum class CompileTimeEntryType {
  PARALLEL_BINDING_ITERDOMAINS,
  PARALLEL_ITER_EXTENT_MAP,
  SIMPLIFIED_PARALLEL_ITER_EXTENT_MAP,
  WARP_PADDED_PARALLEL_EXTENTS,
  VECTORIZED_TENSOR_VALIDATION,
  INPUT_ALIAS_INDICES,
  OUTPUT_ALIAS_INDICES
};

//! Entry class definitions for each entry type:
//!  each class defines the data type for each entry type

//! Compile-time info to be cached in each FusionExecutor:
//!  ParallelBindingIterDomains:
//!    Stores all the iterdomains that are parallelized
//!    on the scheduled Fusion graph. They will be used
//!    in launch param iteration and their extents may
//!    come from launch constraints.
class ParallelBindingIterDomains {
 public:
  using DataType = std::vector<IterDomain*>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::PARALLEL_BINDING_ITERDOMAINS;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  ParallelIterExtentMap
//!    Stores the symbolic extents of all the parallelized
//!    iterdomains corresponding to each used parallel type.
class ParallelIterExtentMap {
 public:
  using DataType =
      std::unordered_map<ParallelType, std::vector<const Val*>, TypeHash>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::PARALLEL_ITER_EXTENT_MAP;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  SimplifiedParallelIterExtentMap
//!    This entry type is a simplified version of ParallelIterExtentMap.
//!
//!    For launch parameter binding we only need the most concrete iterdomain
//!      in each disjoint set stored in CaParallelMap. This entry stores the
//!      remaining list of extents for binding after this simplification.
//!
//!    We still need ParallelIterExtentMap since we want to bind the concrete
//!      values to the extents of all parallelized iterdomains. We would be
//!      able to save these bindings if the integer machine has a notion of
//!      equality and could be configured compile time. But that'd be a longer
//!      term target.
class SimplifiedParallelIterExtentMap {
 public:
  using DataType =
      std::unordered_map<ParallelType, std::vector<const Val*>, TypeHash>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::SIMPLIFIED_PARALLEL_ITER_EXTENT_MAP;
};

//!  WarpPaddedExtentsInfo:
//!    Auxiliary data type for entry class WarpPaddedParallelExtents
struct WarpPaddedExtentsInfo {
  std::unordered_set<const Val*> warp_padded_extent_set;
  std::unordered_map<const Val*, int64_t> warp_padded_constant;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  WarpPaddedParallelExtents
//!    Stores the symbolic and constant extents of warp
//!    padded parallel iterdomains.
class WarpPaddedParallelExtents {
 public:
  using DataType = WarpPaddedExtentsInfo;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::WARP_PADDED_PARALLEL_EXTENTS;
};

//!  VectorizedTensorInfo:
//!    Auxiliary data type for entry class VectorizedTensorValidation
struct VectorizedTensorInfo {
  //! Aligned vectorized fusion inputs
  std::vector<int> aligned_vectorized_inp_tensor_pos;
  //! Aligned vectorized fusion outputs
  std::vector<int> aligned_vectorized_out_tensor_pos;
  //! Misaligned vectorized input tensors
  std::unordered_set<TensorView*> global_inp_misaligned_tv;
  //! Misaligned vectorized output tensors
  std::unordered_set<TensorView*> global_out_misaligned_tv;
  //! Positions of misaligned input tensors
  std::vector<int> inp_misaligned_tensors_pos;
  //! Positions of misaligned output tensors
  std::vector<int> out_misaligned_tensors_pos;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  VectorizedTensorValidation
//!    Stores position info and vector word sizes of
//!    vectorized input/output tensors, to be used
//!    in misaligned vectorization validation.
class VectorizedTensorValidation {
 public:
  using DataType = VectorizedTensorInfo;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::VECTORIZED_TENSOR_VALIDATION;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  InputAliasIndices
//!    Stores position info of aliased input tensors
class InputAliasIndices {
 public:
  using DataType = std::vector<std::pair<int, int>>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::INPUT_ALIAS_INDICES;
};

//! Compile-time info to be cached in each FusionExecutor:
//!  OutputAliasIndices
//!    Stores position info of aliased output tensors
class OutputAliasIndices {
 public:
  using DataType = std::unordered_set<int>;
  static const CompileTimeEntryType EntryType =
      CompileTimeEntryType::OUTPUT_ALIAS_INDICES;
};

//! Base abstract class for unified storage in `ExecutorCompileTimeInfoCache`,
//!  each entry in `ExecutorCompileTimeInfoCache` will be a subclass.
class CompileTimeInfoBase : public PolymorphicBase {
 public:
  CompileTimeInfoBase(CompileTimeEntryType entry_type)
      : entry_type_(entry_type) {}
  CompileTimeEntryType type() {
    return entry_type_;
  }

 private:
  CompileTimeEntryType entry_type_;
};

// Note: Do NOT export this class. MSVC issue with exported class that contains
// std::vector<unique_ptr<xxx>>: https://godbolt.org/z/3E4e8T1P1
//! Compile-time information cache
class ExecutorCompileTimeInfoCache {
  using Entry = CompileTimeInfoBase;
  using EntryOwningPtr = std::unique_ptr<Entry>;
  using EntryPtr = Entry*;
  using EntryType = CompileTimeEntryType;

 public:
  void insert(EntryOwningPtr new_entry);

  EntryPtr at(EntryType entry_type) {
    return entry_type_map_.at(entry_type);
  }

  bool has(EntryType entry_type) {
    return entry_type_map_.count(entry_type);
  }

 private:
  std::vector<EntryOwningPtr> entries_;
  std::unordered_map<EntryType, EntryPtr> entry_type_map_;
};

//! A utility class to facilitate accessing ExecutorCompileTimeInfoCache.
template <typename EntryClass>
class ExecutorCompileTimeEntry {
  using EntryDataType = typename EntryClass::DataType;
  using EntryDataTypeOwnPtr = std::unique_ptr<EntryDataType>;
  using MakerFnType = std::function<EntryDataTypeOwnPtr()>;

 public:
  //! Creates a data entry with type defined in EntryClass,
  //!  eg. EntryClass = VectorizableInputsAndOutputs;
  //!
  //! @param data_cache, a pointer to an instantiated compile-time
  //!  info cache. The info data will be
  //!    1. read from data cache if data cache has the corresponding entry.
  //!    2. written into data cache if data cache doesn't have the entry.
  //!    3. managed by owned_data_ if data cache is nullptr
  //! @param fn:
  //!   The factory function that needs to return a owning pointer
  //!  i.e. std::unique_ptr<EntryClass::DataType>. It will only
  //!  be called either when data cache is missing an entry or when no data
  //!  cache is given.
  ExecutorCompileTimeEntry(
      ExecutorCompileTimeInfoCache* data_cache,
      MakerFnType fn);

  //! Unified interface to get actual data, either from cache
  //!  or from factory function.
  EntryDataType& get() {
    return *data_ptr_;
  }

 private:
  //! Internal data owing pointer that will manage the computed
  //!  data where there is no data cache.
  EntryDataTypeOwnPtr owned_data_ = nullptr;

  //! Pointer to the valid data entry that could be accessed.
  EntryDataType* data_ptr_ = nullptr;
};

} // namespace caching

//! Returns the vector of tensorviews that will be used to bind parallel
//!  dimensions.
std::vector<IterDomain*> getParallelBindingsIterDomains(
    GpuLower* lower,
    const std::vector<TensorView*>& used_tvs);

using ParallelExtentMap =
    std::unordered_map<ParallelType, std::vector<const Val*>, TypeHash>;

//! Returns the extents of all parallel binding iterdomains corresponding
//!  to each parallel type.
std::unique_ptr<ParallelExtentMap> getParallelIterExtents(
    std::vector<IterDomain*>& parallel_binding_ids);

//! Returns the simplified set of extents necessary for launch parameter
//!  binding.
std::unique_ptr<ParallelExtentMap> getSimplifiedParallelIterExtents(
    GpuLower* lower,
    std::vector<IterDomain*>& parallel_binding_ids);

//! Returns the symbolic or constant extetns of warp padded parallel
//!  iterdomains in the given vector.
std::unique_ptr<caching::WarpPaddedExtentsInfo> getWarpPaddedExtentsInfo(
    kir::Kernel* lower,
    std::vector<IterDomain*>& parallel_binding_ids);

void validateVectorizedTensors(
    kir::Kernel* kernel,
    const KernelArgumentHolder& args,
    const std::vector<at::Tensor>& outputs,
    caching::ExecutorCompileTimeInfoCache* data_cache,
    ExpressionEvaluator& expr_eval);

} // namespace executor_utils
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
