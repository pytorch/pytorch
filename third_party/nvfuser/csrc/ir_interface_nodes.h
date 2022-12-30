#pragma once

#include <c10/macros/Export.h>

#include <fusion.h>
#include <ir_base_nodes.h>
#include <ir_builder_passkey.h>
#include <ir_internal_nodes.h>
#include <mma_type.h>

#include <torch/csrc/jit/ir/ir.h>

#include <complex>
#include <limits>
#include <sstream>

//! Nodes in here are intended to be "user facing" users in this sense being
//! those that want to be able to generate CUDA code.

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class WelfordResult;
class ViewTransform;

class IrCloner;

namespace ir_utils {
TORCH_CUDA_CU_API std::string varName(const Val* val);
}

template <typename T>
inline bool __toBool(T x) {
  return static_cast<bool>(x);
}

template <>
inline bool __toBool<std::complex<double>>(std::complex<double> x) {
  return x != std::complex<double>(0, 0);
}

//! A scalr value. This value can be a symbolic value (defined after
//! the kernel is compiled) or a constant value (inlined into the kernel
//! definition).
template <typename UnderlyingType>
class TORCH_CUDA_CU_API Scalar : public Val {
 public:
  using ScalarType = UnderlyingType;
  static constexpr DataType kDefaultDataType =
      NativeTypeToDataType<UnderlyingType>::type;

  explicit Scalar(IrBuilderPasskey passkey, DataType dtype = kDefaultDataType)
      : Val(passkey, ValType::Scalar, dtype), maybe_value_{c10::nullopt} {
    TORCH_INTERNAL_ASSERT(
        (std::is_integral<UnderlyingType>::value && isIntegralType(dtype)) ||
            (std::is_same<UnderlyingType, bool>::value &&
             isBooleanType(dtype)) ||
            (std::is_floating_point<UnderlyingType>::value &&
             isFloatingPointType(dtype)) ||
            (c10::is_complex<UnderlyingType>::value && isComplexType(dtype)),
        "Invalid data type: ",
        dtype);
  }

  explicit Scalar(
      IrBuilderPasskey passkey,
      c10::optional<UnderlyingType> value,
      DataType dtype = kDefaultDataType)
      : Val(passkey, ValType::Scalar, dtype), maybe_value_{value} {
    TORCH_INTERNAL_ASSERT(
        (std::is_integral<UnderlyingType>::value && isIntegralType(dtype)) ||
            (std::is_same<UnderlyingType, bool>::value &&
             isBooleanType(dtype)) ||
            (std::is_floating_point<UnderlyingType>::value &&
             isFloatingPointType(dtype)) ||
            (c10::is_complex<UnderlyingType>::value && isComplexType(dtype)),
        "Invalid data type: ",
        dtype);
  }

  Scalar(const Scalar* src, IrCloner* ir_cloner)
      : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

  NVFUSER_DECLARE_CLONE

  std::string toString(int indent_size = 0) const override {
    std::stringstream ss;
    if (isSymbolic()) {
      ss << ir_utils::varName(this);
      return ss.str();
    }
    if (*getDataType() == DataType::Bool) {
      ss << "(" << (__toBool(value().value()) ? "true" : "false") << ")";
    } else if (isIntegralType(*getDataType())) {
      ss << *(value());
    } else if (isFloatingPointType(*getDataType())) {
      ss << getDataType().value() << "(";
      if (getDataType() == DataType::Double) {
        ss << std::setprecision(std::numeric_limits<double>::max_digits10)
           << *(value()) << ")";
      } else if (getDataType() == DataType::Float) {
        ss << std::setprecision(std::numeric_limits<float>::max_digits10)
           << *(value()) << ")";
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Invalid data type: ", getDataType().value());
      }
    } else if (isComplexType(*getDataType())) {
      ss << getDataType().value() << "(";
      if (getDataType() == DataType::ComplexDouble) {
        ss << std::setprecision(std::numeric_limits<double>::max_digits10)
           << *(value()) << ")";
      } else if (getDataType() == DataType::ComplexFloat) {
        ss << std::setprecision(std::numeric_limits<float>::max_digits10)
           << *(value()) << ")";
      } else {
        TORCH_INTERNAL_ASSERT(
            false, "Invalid data type: ", getDataType().value());
      }
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unknown scalar type: ", *getDataType());
    }
    return ss.str();
  }

  std::string toInlineString(int indent_size = 0) const override {
    if (definition() != nullptr) {
      std::stringstream ss;
      ss << "( " << definition()->toInlineString(indent_size) << " )";
      return ss.str();
    } else {
      return toString(indent_size);
    }
  }

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const final {
    return maybe_value_.has_value();
  }
  c10::optional<UnderlyingType> value() const {
    return maybe_value_;
  }

  bool sameAs(const Statement* other) const override {
    if (this == other) {
      return true;
    }
    if (!other->isA<Scalar>()) {
      return false;
    }
    const auto other_val = other->as<Scalar>();
    if (isConst() && other_val->isConst()) {
      return *value() == *(other_val->value());
    }
    return Val::sameAs(other);
  }

 private:
  const c10::optional<UnderlyingType> maybe_value_;
};

using Bool = Scalar<bool>;
using Int = Scalar<int64_t>;
using Double = Scalar<double>;
using ComplexDouble = Scalar<std::complex<double>>;

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

  NVFUSER_DECLARE_CLONE

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

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

  unsigned int getMaybeMaxProducerPosition() const {
    return maybe_max_producer_pos_;
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

  //! Inline the computation of this tensor into a consumer at the given
  //! position. The consumer to compute with is determined when the
  //! fusion is lowered. Specifically, it is the first consumer tensor
  //! in the topologically ordered dependency graph. Before the
  //! lowering, its compute-with consumer is considered unresolved,
  //! which is then resolved by resolveComputeWith below.
  //!
  //! The position is relative to its own domain. It is an
  //! error if the position is smaller than the compute-at position. If this
  //! tensor is already inlined in a higher position with the same
  //! consumer, then this call is a no-op. The actual position is
  //! computed in the same way as inlineAt, except that computeWith
  //! does not have the constraint of the persistent data-dependency pattern.
  void computeWith(int pos, bool best_effort = false);

  //! Set the actual consumer tensors that this tensor is
  //! computed with. Requires a topologically sorted list expressions,
  //! which can be obtained reorderExprsForComputeAt. Return true if
  //! resolution is actually done. This should only be done in the
  //! Kernel container.
  bool resolveComputeWith(const std::vector<Expr*>& sorted_exprs);

  bool hasComputeWith() const {
    return getComputeWithPosition() > getComputeAtPosition();
  }

  bool hasResolvedComputeWith() const {
    return !compute_with_consumers_.empty();
  }

  //! Query if this tensor is computed with a given consumer.
  bool isComputedWith(const TensorView* consumer) const;

  //! Return the tensors with which this tensor is computed. It is an
  //! error to use this function without first resolving computeWith.
  const std::vector<TensorView*>& getComputeWithConsumers() const;

  unsigned int getComputeWithPosition() const {
    return compute_with_pos_;
  }

  unsigned int getMaxComputePosition() const {
    return std::max(getComputeWithPosition(), getComputeAtPosition());
  }

  //! Returns the position that this tensor is produced at for a given
  //! consumer. If this tensor is computed with the given consumer,
  //! which also means its computeWith needs to have been resolved, the
  //! computeWith position is returned. Otherwise, the default computeAt
  //! position is retured.
  unsigned int getComputePosition(const TensorView* consumer) const;

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

  void clearComputeWith();

 private:
  TensorDomain* domain_ = nullptr;
  unsigned int compute_at_pos_ = 0;
  unsigned int max_producer_pos_ = 0;
  MemoryType memory_type_ = MemoryType::Local;
  bool is_double_buffered_ = false;

  //! Indicates if the tensor is circular buffered.
  bool is_circular_buffered_ = false;

  //! Indicates the circular buffering stage depth if applicable.
  unsigned int circular_buffer_stage_ = 0;

  // special handling for CPU based zero-dim tensors (i.e. CPU Tensors that
  // only have one value). This is only used if on an input value, otherwise
  // ignored. This is important as special handling because these "scalars"
  // should be type promoted as a tensor, but we want to avoid explicit
  // copying of the data, so we want to pass the data value as a standard
  // kernel argument value.
  bool cpu_scalar_ = false;

  //! Indicates if this tensor view has swizzle operator on its tensor domain.
  //!  This is the temporary flag for indicating that the new swizzle
  //!  implementation is used and will be removed in follow ups.
  bool has_swizzle_op_ = false;

  //! Direct consumer tensors that this tensor is computed with
  std::vector<TensorView*> compute_with_consumers_;

  //! Position where this tensor is computed with the compute-with
  //! consumer tensors. It should be always be equal or greater than
  //! the computeAt position
  unsigned int compute_with_pos_ = 0;

  //! Maximum position where producers may be computed at, including
  //! unresolved computeWith. This is equal to max_producer_pos_ when
  //! no producer has unresolved computeWith. It is only used before
  //! resolving computeWith so that no IterDomain should never be
  //! transformed when there may actually be a producer tensor that
  //! may be computed at.
  unsigned int maybe_max_producer_pos_ = 0;
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

  //! Set if a dimension is expanded
  TensorViewBuilder& expanded(std::vector<bool> expanded);

  //! Creates a new TensorView with the specified options
  TensorView* build() const;

 private:
  size_t ndims_ = 0;
  DataType dtype_ = DataType::Float;
  std::vector<bool> contiguity_;
  std::vector<Val*> shape_;
  std::vector<bool> expanded_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
