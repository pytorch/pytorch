#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>

#include <torch/csrc/jit/ir/ir.h>

/*
 * Nodes in here are intended to be "user facing" users in this sense being
 * those that want to be able to generate CUDA code.
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * A Bool value.
 * This value can be a symbolic value (defined after the kernel
 * is compiled) or a constant value (inlined into the kernel definition).
 */
class TORCH_CUDA_CU_API Bool : public Val {
 public:
  // NOLINTNEXTLINE(modernize-use-override)
  ~Bool() = default;

  Bool() : Val(ValType::Scalar, DataType::Bool), maybe_value_{c10::nullopt} {}

  explicit Bool(bool _value)
      : Val(ValType::Scalar, DataType::Bool), maybe_value_{_value} {}

  Bool(const Bool* src, IrCloner* ir_cloner);

  Bool(const Bool& other) = delete;
  Bool& operator=(const Bool& other) = delete;

  Bool(Bool&& other) = delete;
  Bool& operator=(Bool&& other) = delete;

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<bool> value() const {
    return maybe_value_;
  }

  bool sameAs(const Bool* const other) const;

 private:
  const c10::optional<bool> maybe_value_;
};

/*
 * A Float32 value. For now we don't have any other type besides
 * Float32. This value can be a symbolic value (defined after the kernel
 * is compiled) or a constant value (inlined into the kernel definition).
 */
class TORCH_CUDA_CU_API Float : public Val {
 public:
  using ScalarType = double;

  // NOLINTNEXTLINE(modernize-use-override)
  ~Float() = default;

  Float() : Val(ValType::Scalar, DataType::Float), maybe_value_{c10::nullopt} {}

  explicit Float(ScalarType _value)
      : Val(ValType::Scalar, DataType::Float), maybe_value_{_value} {}

  Float(const Float* src, IrCloner* ir_cloner);

  Float(const Float& other) = delete;
  Float& operator=(const Float& other) = delete;

  Float(Float&& other) = delete;
  Float& operator=(Float&& other) = delete;

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

  bool sameAs(const Float* const other) const;

 private:
  const c10::optional<ScalarType> maybe_value_;
};

/*
 * An IEEE 754 Float16 value.
 * This value can be a symbolic value (defined after the kernel
 * is compiled) or a constant value (inlined into the kernel definition).
 */
class TORCH_CUDA_CU_API Half : public Val {
 public:
  // NOLINTNEXTLINE(modernize-use-override)
  ~Half() = default;

  Half() : Val(ValType::Scalar, DataType::Half), maybe_value_{c10::nullopt} {}

  explicit Half(float _value)
      : Val(ValType::Scalar, DataType::Half), maybe_value_{_value} {}

  Half(const Half* src, IrCloner* ir_cloner);

  Half(const Half& other) = delete;
  Half& operator=(const Half& other) = delete;

  Half(Half&& other) = delete;
  Half& operator=(Half&& other) = delete;

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<float> value() const {
    return maybe_value_;
  }

  bool sameAs(const Half* const other) const;

 private:
  const c10::optional<float> maybe_value_;
};

// An Int64 value. If used for indexing it's set as size_t. Otherwise it's an
// inlined literal in the kernel.
class TORCH_CUDA_CU_API Int : public Val {
 public:
  using ScalarType = int64_t;

  // NOLINTNEXTLINE(modernize-use-override)
  ~Int() = default;

  Int() : Val(ValType::Scalar, DataType::Int), maybe_value_{c10::nullopt} {}

  explicit Int(ScalarType _value)
      : Val(ValType::Scalar, DataType::Int), maybe_value_{_value} {}

  Int(const Int* src, IrCloner* ir_cloner);

  Int(const Int& other) = delete;
  Int& operator=(const Int& other) = delete;

  Int(Int&& other) = delete;
  Int& operator=(Int&& other) = delete;

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

  bool sameAs(const Int* const other) const;

 private:
  const c10::optional<ScalarType> maybe_value_;
};

class ComputeAt;
class TransformReplay;
class TransformIter;
class OptOutMutator;
class LoopNestGenerator;

namespace ir_utils {
class TVDomainGuard;
}

// TensorView is our primitive Tensor Type used in code generation. It can be
// thought of as representing physical memory, however, its dimensionality is
// modifed as split/merge/computeAt functions are called. The history of
// these transformations are kept and used for generating actual code referncing
// physical memory. Generally when users are thinking of code generation in
// reference to a Tensor, this is the class they should be interacting with.
//
// The reason we need both TensorView and TensorDomain is that we need to have a
// record of both what is being computed and how it is being computed. For
// example we may have the operation: TV3[I, J, K] = TV2[I, J, K] + TV1[I, J, K]
// The mathematical operations here are on the tensor views TV1, TV2, and TV3.
// This operation is a pointwise operation. To compute this pointwise operation
// we iterate over the 3D TensorDomain [I, J, K], where K is the fastest
// changing dimension.
//
// TODO: Need to work on the const model for TensorView, making all functions
// that should be const, const. Gave this a try but expanded really quickly.
// getComputeAtAxis not being const because it can return a TV that some expect
// to be non-const is the biggest headache.
class TORCH_CUDA_CU_API TensorView : public Val {
 public:
  // NOLINTNEXTLINE(modernize-use-override)
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView(
      TensorDomain* _domain,
      DataType dtype,
      MemoryType mtype = MemoryType::Local);

  TensorView(const std::shared_ptr<c10::TensorType>& tensor_type);

  TensorView(const std::shared_ptr<Value>& jit_value)
      : TensorView(jit_value->type()->cast<c10::TensorType>()) {}

  TensorView(const TensorView* src, IrCloner* ir_cloner);

  TensorDomain* domain() const {
    return domain_;
  }

  bool hasReduction() const;
  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBlockBroadcast() const;
  bool hasBroadcast() const;
  bool hasRFactor() const;

  c10::optional<unsigned int> getReductionAxis() const;

  const std::vector<IterDomain*>& getRootDomain() const;

  const std::vector<IterDomain*>& getRFactorDomain() const;

  // If rfactor domain exists in domain() return it, otherwise return root
  // domain.
  const std::vector<IterDomain*>& getMaybeRFactorDomain() const;

  IterDomain* axis(int pos) const;

  // Is there an active computeAt TensorView/Axis
  bool hasComputeAt() const {
    return compute_at_view_ != nullptr;
  }

  // Return the TensorView we're computing at
  TensorView* getComputeAtView() const {
    return compute_at_view_;
  }

  size_t nDims() const;

  // Return compute at axis relative to this domain
  unsigned int getThisComputeAtAxis() const {
    return this_compute_at_axis_;
  }

  // Return compute at axis relative to compute at view
  unsigned int getRelativeComputeAtAxis() const {
    return relative_compute_at_axis_;
  }

  // Return position in compute_at_view that lines up with this->axis(pos)?
  int getComputeAtRelPos(int pos);

  // Will check if an axis is inside computeAtAxis and will fetch the reference
  // to be used in code generation.
  std::pair<int, TensorView*> getComputeAtPos(int pos) {
    pos = normalizeAxisPos(pos);
    TORCH_INTERNAL_ASSERT(
        nDims() > 0, "Tried to access a computeAt axis in a 0-dim TensorView");
    if (!hasComputeAt() || getThisComputeAtAxis() <= (unsigned int)pos)
      return std::make_pair(pos, this);
    return compute_at_view_->getComputeAtPos(getComputeAtRelPos(pos));
  }

  std::pair<IterDomain*, TensorView*> getComputeAtAxis(int pos) {
    const auto computeAtPos = getComputeAtPos(pos);
    return std::make_pair(
        computeAtPos.second->axis(computeAtPos.first), computeAtPos.second);
  }

  // Compute this TensorView relative to another tensor at axis
  TensorView* computeAt(TensorView* consumer, int axis);

  void clearComputeAt() {
    this_compute_at_axis_ = 0;
    relative_compute_at_axis_ = 0;
    compute_at_view_ = nullptr;
  }

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor
  TensorView* split(int axis, unsigned int factor);

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor. Factor can be a symbolic
  // value instead of constant. This requires setting the symbolic value as an
  // input, or using a parallel dim from NamedScalar::getParallelDim
  TensorView* split(int axis, Val* factor);

  // Merge axis_o and axis_i into 1 IterDomain
  TensorView* merge(int axis_o, int axis_i);

  // Merge axis and axis+1 into 1 IterDomain
  TensorView* merge(int axis) {
    return merge(axis, axis + 1);
  }

  // Reorder axes according to old2new[old_pos] = new_pos
  TensorView* reorder(const std::unordered_map<int, int>& old2new);

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

  // Create a TensorView before the original tensor. A common use case is to
  // write results into shared memory or registers before moving to global
  // memory. Analogous to TVM Cache_Write
  TensorView* cache_before();

  // Create a TensorView after the original tensor. A common use case is to
  // read tensor into shared memory or registers. Analogous to TVM Cache_Read
  TensorView* cache_after();

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  void setMemoryType(MemoryType mt);

  friend TORCH_CUDA_CU_API TransformReplay;
  friend TORCH_CUDA_CU_API OptOutMutator;
  friend TORCH_CUDA_CU_API LoopNestGenerator;
  friend ComputeAt;
  friend void IrFixComputeAt(Fusion*);
  friend void adjustMemoryTypes(Fusion* fusion);
  friend class ir_utils::TVDomainGuard;

 protected:
  // Make an exact copy of this tensor (similar to clone()), however, also grabs
  // the same name. Current use of this is for initialization of reductions.
  // This will break our dependency chain as it is a literal clone of a
  // TensorView but it has a different dependency chain. We need to improve our
  // dependency model to allow for initailziation of reduction buffers. The only
  // reason we can get away with this for now is because we don't use dependency
  // analysis for the IR after we call this.
  TensorView* unsafeClone() const;

  void setDomain(TensorDomain* td) {
    domain_ = td;
  }

  void setComputeAt(TensorView* computeAtView, int axis);

  // Set all computeAt members without checking any correctness. Useful for
  // computeAt with outputs relative to eachother
  void setComputeAt(TensorView* computeAtView, int thisPos, int relPos);

 private:
  int normalizeAxisPos(int pos) const {
    if (pos < 0) {
      pos += nDims();
    }
    return pos;
  }

  // In Cache Before, for the origin expr of the original tensor,
  // we create a new operation where the original tensor is replaced
  // with the new cache tensor. This function creates a new expr
  // given the consumer, the output of the expression.
  void createExprConsumer(Expr* expr, TensorView* consumer);

  // In Cache After, for all the uses of the original tensor, we create
  // a new operation where the original tensor is replaced with the new
  // cache tensor. This function creates a new expr given a producer,
  // an input for the expression.
  void createExprProducer(
      Expr* expr,
      TensorView* current,
      TensorView* producer);

  void setThisComputeAtAxis();

 private:
  TensorDomain* domain_ = nullptr;
  TensorView* compute_at_view_ = nullptr;
  // compute at axis in compute at view
  unsigned int relative_compute_at_axis_ = 0;
  unsigned int this_compute_at_axis_ = 0;
  MemoryType memory_type_ = MemoryType::Local;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
