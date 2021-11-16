#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>

#include <torch/csrc/jit/ir/ir.h>

//! Nodes in here are intended to be "user facing" users in this sense being
//! those that want to be able to generate CUDA code.

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class WelfordResult;

//! A Bool value
//!
//! This value can be a symbolic value (defined after the kernel
//! is compiled) or a constant value (inlined into the kernel definition).
//!
class TORCH_CUDA_CU_API Bool : public Val {
 public:
  Bool() : Val(ValType::Scalar, DataType::Bool), maybe_value_{c10::nullopt} {}

  explicit Bool(bool value)
      : Val(ValType::Scalar, DataType::Bool), maybe_value_{value} {}

  Bool(const Bool* src, IrCloner* ir_cloner);

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<bool> value() const {
    return maybe_value_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const c10::optional<bool> maybe_value_;
};

//! A Float64 value. For now we don't have any other type besides
//! Float64. This value can be a symbolic value (defined after the kernel
//! is compiled) or a constant value (inlined into the kernel definition).
class TORCH_CUDA_CU_API Double : public Val {
 public:
  using ScalarType = double;

  Double()
      : Val(ValType::Scalar, DataType::Double), maybe_value_{c10::nullopt} {}

  explicit Double(ScalarType value)
      : Val(ValType::Scalar, DataType::Double), maybe_value_{value} {}

  Double(const Double* src, IrCloner* ir_cloner);

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
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

  Int() : Val(ValType::Scalar, DataType::Int), maybe_value_{c10::nullopt} {}

  explicit Int(ScalarType value)
      : Val(ValType::Scalar, DataType::Int), maybe_value_{value} {}

  Int(const Int* src, IrCloner* ir_cloner);

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
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

class ComputeAt;
class TransformPropagator;
class TransformIter;
class TransformReplay;
class OptOutMutator;

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
      TensorDomain* domain,
      DataType dtype,
      MemoryType mtype = MemoryType::Local);

  explicit TensorView(const std::shared_ptr<c10::TensorType>& tensor_type);

  explicit TensorView(const std::shared_ptr<Value>& jit_value)
      : TensorView(jit_value->type()->cast<c10::TensorType>()) {}

  TensorView(const TensorView* src, IrCloner* ir_cloner);

  TensorDomain* domain() const {
    return domain_;
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
  TensorView* split(int axis, unsigned int factor, bool inner_split = true);

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor. Factor can be a symbolic
  // value instead of constant. This requires setting the symbolic value as an
  // input, or using a parallel dim from NamedScalar::getParallelDim
  TensorView* split(int axis, Val* factor, bool inner_split = true);

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

  //! Welford Version of rFactor, semantically similar with
  //!  the reduction version except that the rfactor is done
  //!  in a multi-output scan pattern
  WelfordResult rFactor(
      const std::vector<int>& axes,
      TensorView* avg,
      TensorView* var,
      TensorView* n);

  // Create a TensorView before the original tensor. A common use case is to
  // write results into shared memory or registers before moving to global
  // memory. Analogous to TVM Cache_Write
  TensorView* cache_before();

  // Create a TensorView after the original tensor. A common use case is to
  // read tensor into shared memory or registers. Analogous to TVM Cache_Read
  TensorView* cache_after();

  // For a fusion output with other uses, we want to avoid writing to global
  // memory and then reading the output again. We write to global memory
  // separately after an operation. We replace this fusion output with the
  // direct write TensorView.
  TensorView* cache_fork();

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

  friend TORCH_CUDA_CU_API TransformPropagator;
  friend TORCH_CUDA_CU_API TransformReplay;
  friend TORCH_CUDA_CU_API OptOutMutator;
  friend ComputeAt;
  friend void adjustMemoryTypes(Fusion* fusion);
  friend class ir_utils::TVDomainGuard;

 protected:
  void setDomain(TensorDomain* td) {
    domain_ = td;
  }

  void setComputeAt(unsigned int this_pos, bool decrease = false);

  void setMaxProducer(unsigned int this_pos, bool decrease = false);

 private:
  int normalizeAxisPos(int pos) const {
    if (pos < 0) {
      pos += nDims();
    }
    return pos;
  }

  //! A helper function to maintain the consistency of welford output
  //! schedules when doing rfactor on welford ops.
  TensorView* welfordRfactorHelper(
      TensorView* tv,
      const std::vector<int>& axes);

 private:
  TensorDomain* domain_ = nullptr;
  unsigned int compute_at_pos_ = 0;
  unsigned int max_producer_pos_ = 0;
  MemoryType memory_type_ = MemoryType::Local;
  SwizzleType swizzle_type_ = SwizzleType::NoSwizzle;
  std::vector<IterDomain*> axes_to_swizzle_;
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
  TensorViewBuilder& shape(std::vector<int64_t> shape);

  //! Creates a new TensorView with the specified options
  TensorView* build() const;

 private:
  size_t ndims_ = 0;
  DataType dtype_ = DataType::Float;
  std::vector<bool> contiguity_;
  std::vector<int64_t> shape_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
