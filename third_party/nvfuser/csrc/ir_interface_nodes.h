#pragma once

#include <c10/macros/Export.h>

#include <fusion.h>
#include <ir_base_nodes.h>
#include <ir_internal_nodes.h>
#include <mma_type.h>

#include <torch/csrc/jit/ir/ir.h>

//! Nodes in here are intended to be "user facing" users in this sense being
//! those that want to be able to generate CUDA code.

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class WelfordResult;
class ViewTransform;

class IrCloner;
class IrBuilderPasskey;

//! A Bool value
//!
//! This value can be a symbolic value (defined after the kernel
//! is compiled) or a constant value (inlined into the kernel definition).
//!
class TORCH_CUDA_CU_API Bool : public Val {
 public:
  Bool(IrBuilderPasskey passkey);

  explicit Bool(IrBuilderPasskey passkey, bool value);

  explicit Bool(IrBuilderPasskey passkey, c10::optional<bool> value);

  Bool(const Bool* src, IrCloner* ir_cloner);

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const final {
    return maybe_value_.has_value();
  }
  c10::optional<bool> value() const {
    return maybe_value_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const c10::optional<bool> maybe_value_;
};

//! A Float64 value. This value can be a symbolic value (defined after the
//! kernel is compiled) or a constant value (inlined into the kernel
//! definition).
class TORCH_CUDA_CU_API Double : public Val {
 public:
  using ScalarType = double;

  Double(IrBuilderPasskey passkey);

  explicit Double(IrBuilderPasskey passkey, ScalarType value);

  explicit Double(IrBuilderPasskey passkey, c10::optional<ScalarType> value);

  Double(const Double* src, IrCloner* ir_cloner);

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const final {
    return maybe_value_.has_value();
  }
  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const c10::optional<ScalarType> maybe_value_;
};

//! An Int64 value. If used for indexing it's set as size_t. Otherwise it's an
//! inlined literal in the kernel.
class TORCH_CUDA_CU_API Int : public Val {
 public:
  using ScalarType = int64_t;

  Int(IrBuilderPasskey passkey);

  explicit Int(IrBuilderPasskey passkey, ScalarType value);

  explicit Int(IrBuilderPasskey passkey, c10::optional<ScalarType> value);

  Int(const Int* src, IrCloner* ir_cloner);

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const final {
    return maybe_value_.has_value();
  }
  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const c10::optional<ScalarType> maybe_value_;
};

//! An c10::complex<double> value. This value can be a symbolic value (defined
//! after the kernel is compiled) or a constant value (inlined into the kernel
//! definition).
class TORCH_CUDA_CU_API ComplexDouble : public Val {
 public:
  using ScalarType = c10::complex<double>;

  ComplexDouble(IrBuilderPasskey passkey);

  explicit ComplexDouble(IrBuilderPasskey passkey, ScalarType value);

  explicit ComplexDouble(
      IrBuilderPasskey passkey,
      c10::optional<ScalarType> value);

  ComplexDouble(const ComplexDouble* src, IrCloner* ir_cloner);

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const final {
    return maybe_value_.has_value();
  }
  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const c10::optional<ScalarType> maybe_value_;
};

//! Mode during propagation of computeAt, standard will throw an error if
//! computeAt position provided can't be satisfied, best effort will lower the
//! computeAt position as needed during traversal, most inlined will increase
//! the compute at position to maximum possible through traversal.
enum class ComputeAtMode { Standard, BestEffort, MostInlined };

class TransformPropagator;
struct MostInlinedTransformPropagator;
class TransformIter;
class TransformReplay;
class OptOutMutator;
class TensorDomain;

class MaxPosCalculator;

namespace ir_utils {
class TVDomainGuard;
}

//! TensorView is our primitive Tensor Type used in code generation. It can be
//! thought of as representing physical memory, however, its dimensionality is
//! modifed as split/merge/computeAt functions are called. The history of
//! these transformations are kept and used for generating actual code
//! referncing physical memory. Generally when users are thinking of code
//! generation in reference to a Tensor, this is the class they should be
//! interacting with.
//!
//! The reason we need both TensorView and TensorDomain is that we need to have
//! a record of both what is being computed and how it is being computed. For
//! example we may have the operation:
//!
//!   TV3[I, J, K] = TV2[I, J, K] + TV1[I, J, K]
//!
//! The mathematical operations here are on the tensor views TV1, TV2, and
//! TV3. This operation is a pointwise operation. To compute this pointwise
//! operation we iterate over the 3D TensorDomain [I, J, K], where K is the
//! fastest changing dimension.
//!
//! \todo Need to work on the const model for TensorView, making all functions
//! that should be const, const. Gave this a try but expanded really quickly.
//! getComputeAtAxis not being const because it can return a TV that some expect
//! to be non-const is the biggest headache.
//!
class TORCH_CUDA_CU_API TensorView : public Val {
 public:
  TensorView(
      IrBuilderPasskey passkey,
      TensorDomain* domain,
      DataType dtype,
      MemoryType mtype = MemoryType::Local);

  explicit TensorView(
      IrBuilderPasskey passkey,
      const std::shared_ptr<c10::TensorType>& tensor_type);

  explicit TensorView(
      IrBuilderPasskey passkey,
      const std::shared_ptr<Value>& jit_value);

  TensorView(const TensorView* src, IrCloner* ir_cloner);

  TensorDomain* domain() const {
    return domain_;
  }

  //! This is for a TensorView with an rFactor domain that is an input to a
  //! fusion segment. We convert the rfactor domain into a new root domain.
  //! Any dynamic-sized rfactor iterDomains are given a new symbolic extent.
  //! Concrete integer extents are kept. Output TensorViews of any subsequent
  //! expressions that use this TensorView are also updated.
  void convertRfactorToRootDomain();

  void setContiguity(const std::vector<bool>& contig) {
    domain()->setContiguity(contig);
  }

  void setContiguity(bool contig) {
    setContiguity(std::vector<bool>(domain()->contiguity().size(), contig));
  }

  bool hasReduction() const;
  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBroadcast() const;
  bool hasRFactor() const;

  //! This is the previous hasReduction logic,
  //! kept here exclusively for lower loop pass will
  //! deprecate when Fusion IR pass can convert
  //! trivial reductions
  bool hasAnyReduction() const;

  //! Returns true if this tensor is zero dimensional,
  //!  i.e. a wrapped scalar or an empty placeholder.
  bool isZeroDim() const {
    return nDims() == 0;
  }

  //! Returns true if this tensor does not contain
  //!  any value.
  bool isEmptyTensor() const;

  c10::optional<unsigned int> getReductionAxis() const;

  const std::vector<IterDomain*>& getRootDomain() const;

  const std::vector<IterDomain*>& getRFactorDomain() const;

  // If rfactor domain exists in domain() return it, otherwise return root
  // domain.
  const std::vector<IterDomain*>& getMaybeRFactorDomain() const;

  IterDomain* axis(int pos) const;

  // Does it share outer axes with other tensors?
  bool hasComputeAt() const {
    return compute_at_pos_ > 0;
  }

  bool hasMaxProducerPosition() const {
    return max_producer_pos_ > 0;
  }

  size_t nDims() const;

  // sets cpu_scalar_ value, which is special handling for CPU based zero-dim
  // tensors (i.e. CPU Tensors that only have one value). This is only used if
  // on an input value, otherwise ignored. This is important as special handling
  // because these "scalars" should be type promoted as a tensor, but we want to
  // avoid explicit copying of the data, so we want to pass the data value as a
  // standard kernel argument value.
  void setCpuScalar(bool is_cpu_scalar);

  // returns cpu_scalar_ value, which is special handling for CPU based zero-dim
  // tensors (i.e. CPU Tensors that only have one value). This is only used if
  // on an input value, otherwise ignored. This is important as special handling
  // because these "scalars" should be type promoted as a tensor, but we want to
  // avoid explicit copying of the data, so we want to pass the data value as a
  // standard kernel argument value.
  bool isCpuScalar() const {
    return cpu_scalar_;
  }

  // Returns the position that this tensor is produced at relative to its axes.
  unsigned int getComputeAtPosition() const {
    return compute_at_pos_;
  }

  // Returns the maximum position of producers are being computed at relative to
  // this tensor. This position dictates the clear expectations of producers.
  unsigned int getMaxProducerPosition() const {
    return max_producer_pos_;
  }

  //! This is used when we disconnect a tensorview from a reduction
  //!  operation and connect it to a non-reduction operator. We need
  //!  to remove the reduction ids on the tv in this case.
  //! Currently only used in translate welford, and this function may
  //!  be refactored or extended if any more use cases appear.
  void clearReductionIterDomains();

  //! Compute this TensorView relative to a consumer position, -1 will
  //! compute tensors inline with each other, 0 doesn't share
  //! any loop nests between the tensors. It's an error when the given
  //! position is not legally viable. Alternatively, when the mode
  //! parameter is ComputeAtMode::BestEffort, the position is lowered
  //! one by one until a valid position is found. When
  //! ComputeAtMode::MostInlined is given, the position parameter is
  //! ignored, and the deepest possible position is searched.
  TensorView* computeAt(
      TensorView* consumer,
      int position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  //! Compute this tensor to consumer, at local position, -1 will compute
  //! tensors inline with eachother, 0 doesn't share any loop nests between the
  //! tensors. The mode parameter can be used in the same manner as computeAt.
  TensorView* computeWith(
      TensorView* consumer,
      int position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  // Split "axis" into 2 axes
  //! inner_split dictates if the factor section of the split should be inside
  //! the
  //! remainer or outside.
  //! e.g. split(0, 4, inner_split = true) will result in:
  //! tv[id{extent}] -> tv[id{ceilDiv(extent, factor)}, id{factor}]
  //! e.g. split(0, 4, inner_split = false) will result in:
  //! tv[id{extent}] -> tv[id{factor}, id{ceilDiv(extent, factor)}]
  //!
  //! When trim_out_of_bounds is true, only the inner domain defined by the
  //! start and stop positions is split.
  TensorView* split(
      int axis,
      unsigned int factor,
      bool inner_split = true,
      bool trim_out_of_bounds = false);

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor. Factor can be a symbolic
  // value instead of constant. This requires setting the symbolic value as an
  // input, or using a parallel dim from NamedScalar::getParallelDim
  TensorView* split(
      int axis,
      Val* factor,
      bool inner_split = true,
      bool trim_out_of_bounds = false);

  // Merge axis_o and axis_i into 1 IterDomain
  TensorView* merge(int axis_o, int axis_i);

  // Merge axis and axis+1 into 1 IterDomain
  TensorView* merge(int axis) {
    return merge(axis, axis + 1);
  }

  // Reorder axes according to old2new[old_pos] = new_pos
  TensorView* reorder(const std::unordered_map<int, int>& old2new);

  //! Swizzle indices to improve memory access efficiency.
  //!
  //! Swizzle::Transpose is a pattern commonly used to avoid bank
  //! conflicts in shared memory. It takes two axes and shifts the
  //! second axis by the first axis as ((axis1 + axis2) % extent). The
  //! memory type must be Shared.
  //!
  //! \input type Swizzle pattern such as transpose.
  //! \input axes Axes to swizzle
  TensorView* swizzle(SwizzleType type, const std::vector<int>& axes);

  //! Swizzle the rectangular tile defined by the iterdomains corresponding
  //!  to the 2 given indices.
  TensorView* swizzle(
      Swizzle2DType swizzle_type,
      int x,
      int y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  // WARNING: rFactor does not return this TensorView, ir returns a new
  //  tensorview consumed by this!
  //
  // Take reduction axes out of this domain, and create a new
  // domain. New domain will be used to create this domain.
  //
  // For example:
  //  TV1[I0, R1, R2, I3] = TV0[I0, I1, I2, I3]
  //
  // After:
  //  TV1->rfactor({1}), TV1 is transformed to -> TV1[I0, R2, I3]
  //
  // The TensorView returned is: TV2[I0, R1, I2, I3]
  //
  // The reduction will now beset as:
  //  TV2[I0, R1, I2, I3] = TV0[I0, I1, I2, I3]
  //  TV1[I0, R2, I3] = TV2[I0, R1, I2, I3]
  //
  TensorView* rFactor(const std::vector<int>& axes);

  //! Multi-output version of rFactor, semantically similar with
  //! the reduction version except that the rfactor is done
  //! for all outputs in a consistent way
  std::vector<TensorView*> rFactor(
      const std::vector<int>& axes,
      const std::vector<TensorView*>& tvs);

  //! Create a TensorView before the original tensor. A common use case is to
  //! write results into shared memory or registers before moving to global
  //! memory. Analogous to TVM Cache_Write
  //!
  //! @param cache_op: memory operator to use for the inserted op between
  //!   the the data tensor and the cache tensor
  TensorView* cacheBefore(
      c10::optional<LoadStoreOpType> cache_op = c10::nullopt);

  //! Create a TensorView after the original tensor. A common use case is to
  //! read tensor into shared memory or registers. Analogous to TVM Cache_Read
  //!
  //! @param cache_op: memory operator to use for the inserted op between
  //!   the the data tensor and the cache tensor
  TensorView* cacheAfter(
      c10::optional<LoadStoreOpType> cache_op = c10::nullopt);

  // For a fusion output with other uses, we want to avoid writing to global
  // memory and then reading the output again. We write to global memory
  // separately after an operation. We replace this fusion output with the
  // direct write TensorView.
  TensorView* cacheFork();

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  void setMemoryType(MemoryType mt);

  SwizzleType swizzleType() const {
    return swizzle_type_;
  }

  const std::vector<IterDomain*>& axesToSwizzle() const {
    return axes_to_swizzle_;
  }

  // Apply double buffering transformation
  void doubleBuffer();

  // Apply circular buffering transformation
  void circularBuffer(unsigned int number_of_stage);

  // Returns true if this tensor is double buffered.
  bool isDoubleBuffered() const {
    return is_double_buffered_;
  }

  // Returns true if this tensor is circular buffered.
  bool isCircularBuffered() const {
    return is_circular_buffered_;
  }

  // Returns the depth of circular buffering if applicable.
  unsigned int circularBufferDepth() const {
    TORCH_INTERNAL_ASSERT(
        is_circular_buffered_, toString(), "not circular buffered");
    return circular_buffer_stage_;
  }

  //! Transforms the innermost iterdomains according to the given mma swizzle,
  //!  this should be used on the tvs that are either inputs/outputs of an
  //!  MmaOp, or any tv's that are involved in prolog/epilog fusions and need to
  //!  have a matching thread swizzle with the mma operand/result.
  //! More detail on usage see [WarpMmaSwizzler] in scheduler/mma_utils.h .
  void applyMmaSwizzle(MmaOptions options);

  //! Returns if this tensor view has swizzle operator on its tensor domain.
  //!  This is the temporary flag for indicating that the new swizzle
  //!  implementation is used and will be removed in follow ups.
  bool hasSwizzleOp() const {
    return has_swizzle_op_;
  }

  friend TORCH_CUDA_CU_API TransformPropagator;
  friend TORCH_CUDA_CU_API MostInlinedTransformPropagator;
  friend TORCH_CUDA_CU_API TransformReplay;
  friend TORCH_CUDA_CU_API OptOutMutator;
  friend class InlineBatchingGuard;
  friend class ir_utils::TVDomainGuard;

  // Inline the computation of this tensor into its consumer at the given
  // position. If this tensor is already inlined in a higher position, then this
  // call is a no-op. If the right most dimensions before `pos` are
  // broadcasting, then will not inline into these broadcastings. If
  // best_effort, then will inline into the highest allowed position that is <=
  // `pos`.
  void inlineAt(
      int64_t pos,
      bool best_effort = false,
      MaxPosCalculator* calc = nullptr);

  // Update the max producer position of the current tensor. This is required
  // when we modify producer-consumer relationship of a scheduled tensor, for
  // example, grouping multiple reductions.
  void updateMaxProducerPosition();

 protected:
  void setDomain(TensorDomain* td) {
    domain_ = td;
  }

 private:
  int normalizeAxisPos(int pos) const {
    if (pos < 0) {
      pos += nDims();
    }
    return pos;
  }

  //! A helper function to maintain the consistency of schedules of
  //! multiple outputs wheen doing rfactor on multi-output reduction ops.
  TensorView* multiOutputRfactorHelper(
      TensorView* tv,
      const std::vector<int>& axes);

 private:
  TensorDomain* domain_ = nullptr;
  unsigned int compute_at_pos_ = 0;
  unsigned int max_producer_pos_ = 0;
  MemoryType memory_type_ = MemoryType::Local;
  SwizzleType swizzle_type_ = SwizzleType::NoSwizzle;
  std::vector<IterDomain*> axes_to_swizzle_;
  bool is_double_buffered_ = false;

  //! Indicates if the tensor is circular buffered.
  bool is_circular_buffered_ = false;

  //! Indicates the circular buffering stage depth if applicable.
  unsigned int circular_buffer_stage_ = 0;

  // special handling for CPU based zero-dim tensors (i.e. CPU Tensors that only
  // have one value). This is only used if on an input value, otherwise ignored.
  // This is important as special handling because these "scalars" should be
  // type promoted as a tensor, but we want to avoid explicit copying of the
  // data, so we want to pass the data value as a standard kernel argument
  // value.
  bool cpu_scalar_ = false;

  //! Indicates if this tensor view has swizzle operator on its tensor domain.
  //!  This is the temporary flag for indicating that the new swizzle
  //!  implementation is used and will be removed in follow ups.
  bool has_swizzle_op_ = false;
};

//! A simple TensorView builder
//!
//! Example usage:
//!
//!   auto tv = TensorViewBuilder()
//!       .ndims(ndims)
//!       .dtype(dtype)
//!       .contiguity(contiguity)
//!       .build();
//!
class TORCH_CUDA_CU_API TensorViewBuilder {
 public:
  //! Set the number of dimensions of the tensor (default 0, meaning scalar)
  TensorViewBuilder& ndims(size_t ndims);

  //! Set the data type of the tensor (default DataType::Float)
  TensorViewBuilder& dtype(DataType dtype);

  //! Set the contiguity information (default non-contiguous)
  TensorViewBuilder& contiguity(std::vector<bool> contiguity);

  //! Set the shape (default 0 dimensional, ie. scalar)
  TensorViewBuilder& shape(std::vector<Val*> shape);
  TensorViewBuilder& shape(const std::vector<int64_t>& shape);

  //! Creates a new TensorView with the specified options
  TensorView* build() const;

 private:
  size_t ndims_ = 0;
  DataType dtype_ = DataType::Float;
  std::vector<bool> contiguity_;
  std::vector<Val*> shape_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
