#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>

#include <torch/csrc/jit/ir/ir.h>

/*
 * Nodes in here are intended to be "user facing" users in this sense being
 * those that want to be able to generate CUDA code.
 */

namespace torch {
namespace jit {
namespace fuser {

/*
 * A Bool value.
 * This value can be a symbolic value (defined after the kernel
 * is compiled) or a constant value (inlined into the kernel definition).
 */
struct TORCH_CUDA_API Bool : public Val {
  ~Bool() = default;

  Bool() : Val(ValType::Scalar, DataType::Bool), maybe_value_{c10::nullopt} {}

  Bool(bool _value)
      : Val(ValType::Scalar, DataType::Bool), maybe_value_{_value} {}

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
  c10::optional<bool> value() const noexcept {
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
struct TORCH_CUDA_API Float : public Val {
  ~Float() = default;

  Float() : Val(ValType::Scalar, DataType::Float), maybe_value_{c10::nullopt} {}

  Float(float _value)
      : Val(ValType::Scalar, DataType::Float), maybe_value_{_value} {}

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
  c10::optional<float> value() const noexcept {
    return maybe_value_;
  }

  bool sameAs(const Float* const other) const;

 private:
  const c10::optional<float> maybe_value_;
};

/*
 * An IEEE 754 Float16 value.
 * This value can be a symbolic value (defined after the kernel
 * is compiled) or a constant value (inlined into the kernel definition).
 */
struct TORCH_CUDA_API Half : public Val {
  ~Half() = default;

  Half() : Val(ValType::Scalar, DataType::Half), maybe_value_{c10::nullopt} {}

  Half(float _value)
      : Val(ValType::Scalar, DataType::Half), maybe_value_{_value} {}

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
  c10::optional<float> value() const noexcept {
    return maybe_value_;
  }

  bool sameAs(const Half* const other) const;

 private:
  const c10::optional<float> maybe_value_;
};

// An Int64 value. If used for indexing it's set as size_t. Otherwise it's an
// inlined literal in the kernel.
struct TORCH_CUDA_API Int : public Val {
  ~Int() = default;

  Int() : Val(ValType::Scalar, DataType::Int), maybe_value_{c10::nullopt} {}

  Int(int _value) : Val(ValType::Scalar, DataType::Int), maybe_value_{_value} {}

  Int(const Int& other) = delete;
  Int& operator=(const Int& other) = delete;

  Int(Int&& other) = delete;
  Int& operator=(Int&& other) = delete;

  virtual bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  virtual bool isConst() const {
    return maybe_value_.has_value();
  }
  virtual c10::optional<int> value() const noexcept {
    return maybe_value_;
  }

  virtual bool sameAs(const Int* const other) const;

 private:
  const c10::optional<int> maybe_value_;
};

struct TransformReplay;
struct TransformIter;
struct OptOutMutator;
struct LoopNestGenerator;
struct GPULower;
/*
 * TensorView is our primitive Tensor Type used in code generation. It can be
 * thought of as representing physical memory, however, its dimensionality is
 * modifed as split/merge/reorder/computeAt functions are called. The history of
 * these transformations are kept and used for generating actual code referncing
 * physical memory. Generally when users are thinking of code generation in
 * reference to a Tensor, this is the class they should be interacting with.
 *
 * The reason we need both TensorView and TensorDomain is that we need to have a
 * record of both what is being computed and how it is being computed. For
 * Example we may have the operation: TV3[I, J, K] = TV2[I, J, K] + TV1[I, J, K]
 * The mathematical operationss here are on the tensor views TV1, TV2, and TV3.
 * This operation is a pointwise operation. To compute this pointwise operation
 * we iterate over the 3D TensorDomain [I, J, K], where K is the fastest
 * changing dimension.
 */
struct TORCH_CUDA_API TensorView : public Val {
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView(TensorDomain* _domain, DataType dtype);

  TensorView(const std::shared_ptr<c10::TensorType>& tensor_type);

  TensorView(const std::shared_ptr<Value>& jit_value)
      : TensorView(jit_value->type()->cast<c10::TensorType>()) {}

  // Make a new tensor with the given dtype, and the same domain as this tensor
  // (minus reduction IterDomains).
  TensorView* newForOutput(DataType dtype) const;

  // Make a new tensor with the given dtype, same domain as this tensor, minus
  // reduction IterDomains, with new reduced axes marked as so.
  TensorView* newForReduction(std::vector<unsigned int> axes) const;

  // Make an exact copy of this tensor with the same dtype and same domain
  TensorView* clone() const;

  TensorDomain* domain() const noexcept {
    return domain_;
  }

  bool hasReduction() const;

  // Is there an active computeAt TensorView/Axis
  bool hasComputeAt() const {
    return compute_at_view_ != nullptr;
  }

  // Return the TensorView we're computing at
  TensorView* getComputeAtView() const noexcept {
    return compute_at_view_;
  }

  // domain() accessors
  std::vector<IterDomain*>::size_type nDims() const;
  IterDomain* axis(int pos) const;

  unsigned int getComputeAtAxis() const noexcept {
    return compute_at_axis_;
  }

  // Will check if an axis is inside computeAtAxis and will fetch the reference
  // to be used in code generation.
  std::pair<IterDomain*, TensorView*> getComputeAtAxis(int pos) {
    if (!hasComputeAt() || getComputeAtAxis() <= (unsigned int)pos)
      return std::pair<IterDomain*, TensorView*>(axis(pos), this);
    return compute_at_view_->getComputeAtAxis(pos);
  }

  TensorDomain* getRootDomain() const;
  // Return the TensorView to its original state, before all
  // transformations/computeAt calls.
  void resetView();

  // Compute this TensorView relative to another tensor at axis
  TensorView* computeAt(TensorView* consumer, int axis);

  void clearComputeAt() {
    compute_at_axis_ = 0;
    compute_at_view_ = nullptr;
  }

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor
  TensorView* split(int axis, int factor);

  // Merge "axis" and "axis+1" into 1 dimension
  TensorView* merge(int axis);

  // Reorder axes according to old2new[old_pos] = new_pos
  TensorView* reorder(const std::unordered_map<int, int>& old2new);

  /*
   * WARNING: Does not return this TensorView, returns a new tensorview consumed
   * to create this!! Take reduction axes out of this domain, and create a new
   * domain. New domain will be used to create this domain. For example: TV1[I0,
   * I1] = TV0[I0, R0, R1, I1] TV0->rfactor({1}) TV0 is transformed to ->
   * TV0[I0, R1, I1] The TensorView returned is: TV2[I0, R0, I3, I1] The
   * reduction will now beset as: TV1[I0, R1, I1] = TV2[I0, R0, I3, I1] TV0[I0,
   * I1] = TV1[I0, R1, I1]
   */
  TensorView* rFactor(const std::vector<int>& axes);

  MemoryType getMemoryType() {
    return memory_type_;
  }

  friend TORCH_CUDA_API TransformReplay;
  friend TORCH_CUDA_API TransformIter;
  friend TORCH_CUDA_API OptOutMutator;
  friend TORCH_CUDA_API GPULower;
  friend TORCH_CUDA_API LoopNestGenerator;

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

  void setComputeAt(TensorView* computeAtView, int axis) {
    TORCH_INTERNAL_ASSERT(
        axis >= 0 && (unsigned int)axis <= nDims(),
        "Invalid computeAt on ",
        this,
        " tried to set to ",
        axis);
    compute_at_view_ = computeAtView;
    compute_at_axis_ = axis;
  }

  void setMemoryType(MemoryType mt) {
    memory_type_ = mt;
    bool is_inp_or_out =
        this->fusion()->hasInput(this) || this->fusion()->hasOutput(this);
    if (is_inp_or_out)
      TORCH_INTERNAL_ASSERT(
          mt == MemoryType::Global,
          "Tried to set an input or output to the fusion to a non-global memory type.");
  }

 private:
  // Transform this view like consumer, mark compute_at_(viw,axis)
  void computeAt_impl(TensorView* consumer, int axis);

  // Transform this view like producer, mark producer as compute_at_(this, axis)
  void forwardComputeAt_impl(TensorView* producer, int axis);

  TensorDomain* domain_;
  TensorView* compute_at_view_ = nullptr;
  unsigned int compute_at_axis_ = 0;
  MemoryType memory_type_ = MemoryType::Global;

  // Make a copy of the domain (used for Tensor based constructor), likely to be
  // removed soon.
  void copyDomain(const TensorDomain* td);
};

} // namespace fuser
} // namespace jit
} // namespace torch
