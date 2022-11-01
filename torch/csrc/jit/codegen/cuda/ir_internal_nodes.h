#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/mma_type.h>
#include <torch/csrc/jit/codegen/cuda/parallel_type_bitmap.h>

//! Nodes in here should generally not be used by users. They should be behind
//! the scenes and users shouldn't have to be aware of what they do to use the
//! code generator
//!
//! \todo improve implementation bool IterDomain::sameAs(const IterDomain*)
//! \todo Add testing of sameAs functions for these nodes
//!

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ViewTransform;
class Scope;
class IrCloner;
struct AnalyzeViewResult;

//! Returns true if both v1 and v2 are scalars, are the same type of scalars,
//! and dispatches to the inherited Val type's `->sameAs` call. e.g. if both
//! vals are `Int` will dispatch to v1->as<Int>()->sameAs(v2.as<Int>())
bool areEqualScalars(Val* v1, Val* v2);

class TORCH_CUDA_CU_API FullOp : public Expr {
 public:
  FullOp(IrBuilderPasskey, Val* out, Val* fill_value, DataType dtype);

  FullOp(const FullOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  bool sameAs(const Statement* other) const override;

  DataType dtype() const {
    return dtype_;
  }

  Val* getFillValue() const {
    return fill_value_;
  }

 private:
  const DataType dtype_;
  Val* fill_value_;
};

class TORCH_CUDA_CU_API ARangeOp : public Expr {
 public:
  ARangeOp(
      IrBuilderPasskey,
      Val* out,
      Val* start,
      Val* end,
      Val* step,
      DataType dtype,
      Val* linear_index = nullptr);

  ARangeOp(const ARangeOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  bool sameAs(const Statement* other) const override;

  DataType dtype() const {
    return dtype_;
  }

  Val* start() const {
    return start_;
  }

  Val* end() const {
    return end_;
  }

  Val* step() const {
    return step_;
  }

  Val* getLinearLogicalIndex() const {
    return linear_index_;
  }

  void setLinearIndex(Val* index) {
    linear_index_ = index;
  }

 private:
  const DataType dtype_;
  Val* start_;
  Val* end_;
  Val* step_;
  Val* linear_index_ = nullptr;
};

// Tensor factory for generating identity matrices like
//
// [[1, 0, 0],
//  [0, 1, 0],
//  [0, 0, 1]]
//
// or
//
// [[1, 0, 0],
//  [0, 1, 0],
//  [0, 0, 1],
//  [0, 0, 0]]
//
// or
//
// [[1, 0, 0, 0],
//  [0, 1, 0, 0],
//  [0, 0, 1, 0]]
class TORCH_CUDA_CU_API EyeOp : public Expr {
 public:
  EyeOp(
      IrBuilderPasskey,
      Val* out,
      DataType dtype,
      Val* index1 = nullptr,
      Val* index2 = nullptr);

  EyeOp(const EyeOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  bool sameAs(const Statement* other) const override;

  DataType dtype() const {
    return dtype_;
  }

  Val* getIndex1() const {
    return index1_;
  }

  void setIndex1(Val* index) {
    index1_ = index;
  }

  Val* getIndex2() const {
    return index2_;
  }

  void setIndex2(Val* index) {
    index2_ = index;
  }

 private:
  const DataType dtype_;
  Val* index1_ = nullptr;
  Val* index2_ = nullptr;
};

//! A specialization for Unary operations. Unary operations take in a single
//! input and produce a single output. Examples include:
//!   1) Casting operation i.e. float(a_val)
//!   2) Negation i.e. val * -1
//!   3) Reduction across a dimension i.e. val.sum(axis=2)
//!   4) split/merge
class TORCH_CUDA_CU_API UnaryOp : public Expr {
 public:
  UnaryOp(
      IrBuilderPasskey,
      UnaryOpType type,
      Val* out,
      Val* in,
      int rng_offset = -1);

  UnaryOp(const UnaryOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  UnaryOpType getUnaryOpType() const {
    return unary_op_type_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const UnaryOpType unary_op_type_;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

//! A specialization for Binary operations. Binary operations take in two inputs
//! and produce a single output. Examples include:
//!  1) Add/mul/div/mod/sub (A * B)
//!  2) LT (A < B)
class TORCH_CUDA_CU_API BinaryOp : public Expr {
 public:
  BinaryOp(IrBuilderPasskey, BinaryOpType type, Val* out, Val* lhs, Val* rhs);

  BinaryOp(const BinaryOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }
  Val* lhs() const {
    return lhs_;
  }
  Val* rhs() const {
    return rhs_;
  }

  BinaryOpType getBinaryOpType() const {
    return binary_op_type_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const BinaryOpType binary_op_type_;
  Val* const out_ = nullptr;
  Val* const lhs_ = nullptr;
  Val* const rhs_ = nullptr;
};

//! A specialization for random number generator (RNG) operations. RNG
//! operations take in no tensor input and produce a single output.
class TORCH_CUDA_CU_API RNGOp : public Expr {
 public:
  RNGOp(
      IrBuilderPasskey,
      RNGOpType type,
      Val* out,
      DataType dtype,
      std::vector<Val*> parameters = {},
      int rng_offset = 0,
      Val* philox_index = nullptr);

  RNGOp(const RNGOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  RNGOpType getRNGOpType() const {
    return rng_op_type_;
  }

  DataType dtype() const {
    return dtype_;
  }

  int getRNGOffset() const {
    return rng_offset_;
  }

  void setRNGOffset(int val) {
    rng_offset_ = val;
  }

  const std::vector<Val*>& getParameters() const {
    return parameters_;
  }

  const std::vector<Val*>& getShape() const {
    return shape_;
  }

  Val* getPhiloxIndex() const {
    return philox_index_;
  }

  void setPhiloxIndex(Val* index) {
    philox_index_ = index;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const RNGOpType rng_op_type_;
  const DataType dtype_;
  std::vector<Val*> parameters_;
  std::vector<Val*> shape_;
  int rng_offset_ = -1;
  // The index used to feed philox's subsequence and component
  Val* philox_index_ = nullptr;
};

//! Broadcast in to match out. is_broadcast_dims are relative to out. Where
//! is_broadcast_dims.size() == out->nDims().
class TORCH_CUDA_CU_API BroadcastOp : public Expr {
 public:
  //! \param out The output tensor
  //! \param in The input tensor
  //! \param is_broadcast_dims True when output dim is a new broadcast domain
  BroadcastOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<bool> is_broadcast_dims);

  BroadcastOp(const BroadcastOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  bool isBroadcastDim(size_t dim) const {
    return is_broadcast_dims_.at(dim);
  }

  const std::vector<bool>& getBroadcastDimFlags() const {
    return is_broadcast_dims_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;

  //! The same list passed to the broadcast arithmetic op. Each
  //! element corresponds to an IterDomain of the output tensor and is
  //! true when the IterDomain is a new broadcast domain. Note
  //! that the output tensor may have other broadcast domains whose
  //! flags are false because the input tensor may already have
  //! broadcast domains.
  const std::vector<bool> is_broadcast_dims_;
};

//! Squeeze in to match out. is_squeeze_dims are relative to in. Where
//! is_squeeze_dims.size() == in->nDims(). Squeeze is the opposite of
//! broadcast.
class TORCH_CUDA_CU_API SqueezeOp : public Expr {
 public:
  //! \param out The output tensor
  //! \param in The input tensor
  //! \param is_squeeze_dims True when input dim is a removed broadcast domain
  SqueezeOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<bool> is_broadcast_dims);

  SqueezeOp(const SqueezeOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  bool isSqueezeDim(size_t dim) const {
    return is_squeeze_dims_.at(dim);
  }

  const std::vector<bool>& getSqueezeDimFlags() const {
    return is_squeeze_dims_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;

  //! The same list passed to the squeeze arithmetic op. Each
  //! element corresponds to an IterDomain of the input tensor and is
  //! true when the IterDomain is a broadcast domain that is removed in the
  //! output. Note that the output tensor may still contain broadcast domains
  //! because the input tensor may have broadcast domains that we don't want to
  //! remove (false flag).
  const std::vector<bool> is_squeeze_dims_;
};

//! Reduction operation. Out is first initialized to _init. Then
//! reduction_op_type is used to update out as out = reductionOp(out, in).
//! Output's axes marked as reduction will be reduced to produce an output
//! tensor. The output tensors size will be the size of all
//! non-reduction/non-broadcast dimensions.
class TORCH_CUDA_CU_API ReductionOp : public Expr {
 public:
  ReductionOp(
      IrBuilderPasskey,
      BinaryOpType reduction_op_type,
      Val* init,
      Val* out,
      Val* in,
      bool is_allreduce = false,
      ExprType expr_type = ExprType::ReductionOp);

  ReductionOp(const ReductionOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }
  Val* init() const {
    return init_;
  }

  BinaryOpType getReductionOpType() const {
    return reduction_op_type_;
  }

  bool isAllreduce() const {
    return is_allreduce_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const BinaryOpType reduction_op_type_;
  Val* const init_ = nullptr;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
  //! True if broadcast is fused
  bool is_allreduce_ = false;
};

//! Grouped reduction operation for horizontal fusions. It works like
//! batched GEMMs in the sense that multiple independent reductions are
//! performed together. The main benefit is when reducing tensors across thread
//! blocks, a single grid sync can be done for all individual
//! reductions. As grid sync is very expensive, this can be a
//! significant performance impact.
class TORCH_CUDA_CU_API GroupedReductionOp : public Expr {
 public:
  GroupedReductionOp(
      IrBuilderPasskey,
      std::vector<BinaryOpType> reduction_op_type,
      std::vector<Val*> init,
      std::vector<Val*> out,
      std::vector<Val*> in,
      bool is_allreduce = false,
      ExprType expr_type = ExprType::GroupedReductionOp);

  GroupedReductionOp(const GroupedReductionOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  //! Number of expressions grouped horizontally. It does not reflect
  //! iteration grouping.
  size_t numExprs() const {
    return reduction_op_types_.size();
  }

  const std::vector<Val*>& initVals() const {
    return init_vals_;
  }

  Val* initVal(size_t index) const {
    return init_vals_.at(index);
  }

  const std::vector<BinaryOpType>& getReductionOpTypes() const {
    return reduction_op_types_;
  }

  BinaryOpType getReductionOpType(size_t index) const {
    return reduction_op_types_.at(index);
  }

  bool isAllreduce() const {
    return is_allreduce_;
  }

  //! Return the index of the corresponding reduction expression for
  //! a given output val.
  int getExprIndexOfOutput(Val* output_val) const;

  bool sameAs(const Statement* other) const override;

 private:
  //! Reduction ops of grouped reductions
  const std::vector<BinaryOpType> reduction_op_types_;
  //! Initial values of grouped reductions
  const std::vector<Val*> init_vals_;
  //! True if using the fused reduction kernel
  bool is_allreduce_ = false;
};

//! Average, variance and N (count) vals for Welford
class TORCH_CUDA_CU_API WelfordTriplet {
 public:
  //! Names of the Welford triplet vals
  enum class ValName { Avg, Var, N };

  WelfordTriplet() = default;

  WelfordTriplet(Val* avg, Val* var, Val* N) : vals_({avg, var, N}) {}

  Val* const& avg() const {
    return get(ValName::Avg);
  }

  Val*& avg() {
    return get(ValName::Avg);
  }

  TensorView* avgTv() const {
    TORCH_INTERNAL_ASSERT(avg()->isA<TensorView>());
    return avg()->as<TensorView>();
  }

  Val* const& var() const {
    return get(ValName::Var);
  }

  Val*& var() {
    return get(ValName::Var);
  }

  TensorView* varTv() const {
    TORCH_INTERNAL_ASSERT(var()->isA<TensorView>());
    return var()->as<TensorView>();
  }

  Val* const& N() const {
    return get(ValName::N);
  }

  Val*& N() {
    return get(ValName::N);
  }

  TensorView* NTv() const {
    TORCH_INTERNAL_ASSERT(N()->isA<TensorView>());
    return N()->as<TensorView>();
  }

  //! Get the i-th val. Ordering is defined by ValName.
  Val* const& get(int i) const {
    return vals_.at(i);
  }

  //! Get the i-th val. Ordering is defined by ValName.
  Val*& get(int i) {
    return vals_.at(i);
  }

  Val* const& get(ValName name) const {
    return get(valNameToIndex(name));
  }

  Val*& get(ValName name) {
    return get(valNameToIndex(name));
  }

  //! Get the name of a given val in this triplet. None is returned if
  //! not found.
  c10::optional<ValName> getNameOf(Val* val) const;

  //! Return a new triplet with outputs produced by a function applied
  //! to each of this triplet
  template <typename Func>
  WelfordTriplet transform(Func func) const {
    return WelfordTriplet(func(avg()), func(var()), func(N()));
  }

  bool sameAs(const WelfordTriplet& other) const;

  WelfordTriplet clone(IrCloner* ir_cloner) const;

  //! Clone a vector of triplets
  static std::vector<WelfordTriplet> clone(
      const std::vector<WelfordTriplet>& src,
      IrCloner* ir_cloner);

  auto begin() {
    return vals_.begin();
  }

  auto begin() const {
    return vals_.begin();
  }

  auto end() {
    return vals_.end();
  }

  auto end() const {
    return vals_.end();
  }

 private:
  //! Convert a given val name to an index
  static int valNameToIndex(ValName name) {
    return static_cast<int>(name);
  }

  //! Convert a given index to a name
  static ValName indexToValName(int index) {
    TORCH_INTERNAL_ASSERT(index >= 0 && index < 3, "Invalid index: ", index);
    return static_cast<ValName>(index);
  }

 private:
  //! Holds avg, var and N in this order
  std::array<Val*, 3> vals_ = {{nullptr, nullptr, nullptr}};
};

//! Welford Scan operation.
class TORCH_CUDA_CU_API WelfordOp : public Expr {
 public:
  WelfordOp(
      IrBuilderPasskey,
      const WelfordTriplet& output,
      const WelfordTriplet& input,
      const WelfordTriplet& init,
      bool is_fused = false);

  WelfordOp(
      IrBuilderPasskey,
      Val* out_avg,
      Val* out_var,
      Val* out_N,
      Val* in_avg,
      Val* in_var,
      Val* in_N,
      Val* init_avg,
      Val* init_var,
      Val* init_N,
      bool is_fused = false);

  WelfordOp(const WelfordOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return output().avg();
  }

  Val* in() const {
    return input().avg();
  }

  bool sameAs(const Statement* const other) const override;

  const WelfordTriplet& output() const {
    return output_;
  }

  Val* outAvg() const {
    return output().avg();
  }

  Val* outVar() const {
    return output().var();
  }

  Val* outN() const {
    return output().N();
  }

  const WelfordTriplet& input() const {
    return input_;
  }

  Val* inAvg() const {
    return input().avg();
  }

  Val* inVar() const {
    return input().var();
  }

  Val* inN() const {
    return input().N();
  }

  const WelfordTriplet& init() const {
    return init_;
  }

  Val* initAvg() const {
    return init().avg();
  }

  Val* initVar() const {
    return init().var();
  }

  Val* initN() const {
    return init().N();
  }

  bool singleValue() const {
    return inN()->isOneInt();
  }

  bool hasInit() const {
    return !initN()->isZeroInt();
  }

  bool isAllreduce() const {
    return is_allreduce_;
  }

  std::vector<Val*> getInitVals() const;

  //! Return the init val for an output val
  Val* getInitValOfOutput(Val* output_val) const;

 private:
  const WelfordTriplet output_;
  const WelfordTriplet input_;
  const WelfordTriplet init_;
  //! True if using the fused reduction kernel (not implemented yet)
  bool is_allreduce_ = false;
};

class TORCH_CUDA_CU_API GroupedWelfordOp : public Expr {
 public:
  GroupedWelfordOp(
      IrBuilderPasskey,
      std::vector<WelfordTriplet> output_vals,
      std::vector<WelfordTriplet> input_vals,
      std::vector<WelfordTriplet> init_vals,
      bool is_allreduce = false,
      ExprType expr_type = ExprType::GroupedWelfordOp);

  GroupedWelfordOp(const GroupedWelfordOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  //! Number of expressions grouped horizontally. It does not reflect
  //! iteration grouping. As horizontal grouping is not supported,
  //! this always returns 1.
  size_t numExprs() const {
    return 1;
  }

  Val* out(size_t index) const {
    return outAvg(index);
  }

  Val* in(size_t index) const {
    return inAvg(index);
  }

  bool sameAs(const Statement* const other) const override;

  const std::vector<WelfordTriplet>& outputVals() const {
    return output_vals_;
  }

  const std::vector<WelfordTriplet>& inputVals() const {
    return input_vals_;
  }

  const std::vector<WelfordTriplet>& initVals() const {
    return init_vals_;
  }

  Val* outAvg(size_t index) const {
    return outputVals().at(index).avg();
  }

  Val* outVar(size_t index) const {
    return outputVals().at(index).var();
  }

  Val* outN(size_t index) const {
    return outputVals().at(index).N();
  }

  Val* inAvg(size_t index) const {
    return inputVals().at(index).avg();
  }

  Val* inVar(size_t index) const {
    return inputVals().at(index).var();
  }

  Val* inN(size_t index) const {
    return inputVals().at(index).N();
  }

  Val* initAvg(size_t index) const {
    return initVals().at(index).avg();
  }

  Val* initVar(size_t index) const {
    return initVals().at(index).var();
  }

  Val* initN(size_t index) const {
    return initVals().at(index).N();
  }

  //! Return the index of the corresponding welford expression for
  //! a given output val
  int getExprIndexOfOutput(Val* output_val) const;

  //! Return the init val for an output val
  Val* getInitValOfOutput(Val* output_val) const;

  bool singleValue(size_t index) const {
    return inN(index)->isOneInt();
  }

  bool hasInit(size_t index) const {
    return !initN(index)->isZeroInt();
  }

  bool isAllreduce() const {
    return is_allreduce_;
  }

 private:
  const std::vector<WelfordTriplet> output_vals_;
  const std::vector<WelfordTriplet> input_vals_;
  const std::vector<WelfordTriplet> init_vals_;
  //! True if using the fused reduction kernel
  bool is_allreduce_ = false;
};

//! Fused Matmul operation
class TORCH_CUDA_CU_API MmaOp : public Expr {
 public:
  // This is a temporary data structure to for the
  //  scheduling specific parameters that we still need
  //  to store on an mma node. Eventually will only be
  //  the mma macro type that will stay on the IR node
  //  after additional cleaning ups.
  struct OptionsInMma {
    MmaOptions::MacroType macro = MmaOptions::MacroType::NoMMA;
    MmaOptions::MmaInputLayout operand_layout = MmaOptions::MmaInputLayout::TT;
    int accumulator_stride = 0;

    bool operator==(const OptionsInMma& other) const {
      return macro == other.macro && operand_layout == other.operand_layout &&
          accumulator_stride == other.accumulator_stride;
    }
  };

  MmaOp(IrBuilderPasskey, Val* out, Val* in_a, Val* in_b, Val* init);

  MmaOp(
      IrBuilderPasskey,
      Val* out,
      Val* in_a,
      Val* in_b,
      Val* init,
      OptionsInMma options);

  MmaOp(const MmaOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }

  Val* inA() const {
    return in_a_;
  }

  Val* inB() const {
    return in_b_;
  }

  Val* init() const {
    return init_;
  }

  const auto& options() const {
    TORCH_INTERNAL_ASSERT(options_.has_value(), "MmaOp not configured:", this);
    return options_.value();
  }

  bool sameAs(const Statement* const other) const override;

  auto accStride() const {
    TORCH_INTERNAL_ASSERT(options_.has_value(), "MmaOp not configured:", this);
    return options_->accumulator_stride;
  }

  void configureOptions(MmaOptions options) {
    options_ = OptionsInMma();
    TORCH_INTERNAL_ASSERT(
        options.macro != MmaOptions::MacroType::NoMMA,
        "Un-configured mma type from options.");
    TORCH_INTERNAL_ASSERT(
        options.accumulator_stride > 0, "Un-configured accumulator stride.");
    options_->accumulator_stride = options.accumulator_stride;
    options_->macro = options.macro;
    options_->operand_layout = options.operand_layout;
  }

 private:
  Val* const out_ = nullptr;
  Val* const in_a_ = nullptr;
  Val* const in_b_ = nullptr;
  Val* const init_ = nullptr;
  c10::optional<OptionsInMma> options_ = c10::nullopt;
};

class TORCH_CUDA_CU_API TransposeOp : public Expr {
 public:
  TransposeOp(
      IrBuilderPasskey,
      TensorView* out,
      TensorView* in,
      std::vector<int64_t> new2old);

  TransposeOp(const TransposeOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  TensorView* out() const {
    return out_;
  }

  TensorView* in() const {
    return in_;
  }

  const std::vector<int64_t>& new2old() const {
    return new2old_;
  }

  std::vector<int64_t> old2new() const;

 private:
  TensorView* const out_ = nullptr;
  TensorView* const in_ = nullptr;
  const std::vector<int64_t> new2old_;
};

class TORCH_CUDA_CU_API ExpandOp : public Expr {
 public:
  ExpandOp(
      IrBuilderPasskey,
      TensorView* out,
      TensorView* in,
      std::vector<Val*> _expanded_extents);

  ExpandOp(const ExpandOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  TensorView* out() const {
    return out_;
  }

  TensorView* in() const {
    return in_;
  }

  const std::vector<Val*>& expanded_extents() const {
    return expanded_extents_;
  }

 private:
  TensorView* const out_ = nullptr;
  TensorView* const in_ = nullptr;
  std::vector<Val*> expanded_extents_;
};

class TORCH_CUDA_CU_API TernaryOp : public Expr {
 public:
  TernaryOp(
      IrBuilderPasskey,
      TernaryOpType type,
      Val* out,
      Val* in1,
      Val* in2,
      Val* in3);

  TernaryOp(const TernaryOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }

  Val* in1() const {
    return in1_;
  }
  Val* in2() const {
    return in2_;
  }
  Val* in3() const {
    return in3_;
  }

  TernaryOpType getTernaryOpType() const {
    return ternary_op_type_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  const TernaryOpType ternary_op_type_;
  Val* const out_ = nullptr;
  Val* const in1_ = nullptr;
  Val* const in2_ = nullptr;
  Val* const in3_ = nullptr;
};

//! Shift
class TORCH_CUDA_CU_API ShiftOp : public Expr {
 public:
  //! \param out
  //! \param in
  //! \param offsets
  ShiftOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<int> offsets,
      std::vector<int> pad_width);

  ShiftOp(const ShiftOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  int offset(size_t dim) const {
    return offsets_.at(dim);
  }

  const std::vector<int>& offsets() const {
    return offsets_;
  }

  const std::vector<int>& padWidth() const {
    return pad_width_;
  }

  bool hasPadding() const {
    return std::any_of(pad_width_.begin(), pad_width_.end(), [](const auto p) {
      return p > 0;
    });
  }

  bool sameAs(const Statement* other) const override;

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
  //! Each of the root axes is shifted by the corresponding value of
  //! offsets_. The sign of each value indicates the direction of
  //! shifting.
  const std::vector<int> offsets_;
  const std::vector<int> pad_width_;
};

//! Gather a window around each element.
class TORCH_CUDA_CU_API GatherOp : public Expr {
 public:
  GatherOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<int> window_shape,
      std::vector<std::vector<int>> pad_width);

  GatherOp(const GatherOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  const auto& windowShape() const {
    return window_shape_;
  }

  //! Returns the gather axis that corresponds to an input axis
  int gatherAxis(int axis) const;

  const auto& padWidth() const {
    return pad_width_;
  }

  bool hasPadding() const {
    return std::any_of(pad_width_.begin(), pad_width_.end(), [](const auto& p) {
      return p[0] > 0 || p[1] > 0;
    });
  }

  bool sameAs(const Statement* other) const override;

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
  //! Shape of a window gathered for each element.
  std::vector<int> window_shape_;
  //! The size of zero-padding of each axis.
  std::vector<std::vector<int>> pad_width_;
};

class TORCH_CUDA_CU_API ViewAsScalar : public Expr {
 public:
  ViewAsScalar(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      IterDomain* vector_id,
      Val* index = nullptr);

  ViewAsScalar(const ViewAsScalar* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

  IterDomain* vector_id() const {
    return vector_id_;
  }

  Val* index() const {
    return index_;
  }

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;

  // The IterDomain of type VectorComponent newly appended to the output
  IterDomain* vector_id_ = nullptr;

  // The index that vector_id_ is lowered into
  Val* index_ = nullptr;
};

class TORCH_CUDA_CU_API ViewOp : public Expr {
 public:
  ViewOp(IrBuilderPasskey, TensorView* out, TensorView* in);

  ViewOp(const ViewOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  TensorView* out() const {
    return out_;
  }

  TensorView* in() const {
    return in_;
  }

 private:
  TensorView* const out_ = nullptr;
  TensorView* const in_ = nullptr;
};

//! This operator explicitly models data movement between
//!   state spaces on GPU. Currently the modeled state spaces include
//!   global memory, shared memory and register.
//!
//! The main usage of this op is to facilitate generation of hardware
//!   accelerated memory ops, i.e. ldmatrix, cp.async and more to come.
class TORCH_CUDA_CU_API LoadStoreOp : public Expr {
 public:
  LoadStoreOp(IrBuilderPasskey, LoadStoreOpType op_type, Val* out, Val* in);

  LoadStoreOp(const LoadStoreOp* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

  LoadStoreOpType opType() const {
    return load_store_type_;
  }

 private:
  LoadStoreOpType load_store_type_ = LoadStoreOpType::LdMatrix;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

// Convenience utility to initialize IterDomain's without having to sort through
// all the default values. Intended to be used with
// IterDomain::IterDomain(IrBuilderPasskey IterDomainBuildArgs)
class TORCH_CUDA_CU_API IterDomainBuilder {
 public:
  // Match legacy constructor
  IterDomainBuilder(Val* _start, Val* _extent);

  // Grab all the parameters from id to set the IterDomainBuilder
  IterDomainBuilder(const IterDomain* id);

  // Resets defaults for rfactor, is padded dim, padded to size, and is mma
  // swizzle which should only be set during scheduling.
  IterDomainBuilder& resetSchedulingParams();

  // Resets is_rfactor_domain
  IterDomainBuilder& resetRfactor();

  IterDomainBuilder& start(Val* _start);
  IterDomainBuilder& extent(Val* _extent);
  IterDomainBuilder& expanded_extent(Val* _expanded_extent);
  IterDomainBuilder& stop_offset(Val* _stop_offset);
  IterDomainBuilder& parallel_type(ParallelType _parallel_type);
  IterDomainBuilder& iter_type(IterType _iter_type);
  IterDomainBuilder& is_rfactor_domain(bool _is_rfactor_domain);
  IterDomainBuilder& is_padded_dimension(bool _is_padded_dimension);
  IterDomainBuilder& padded_to_size(c10::optional<int64_t> _padded_to_size);
  IterDomainBuilder& is_mma_swizzled(bool _is_mma_swizzled);

  IterDomain* build() const;

  // Must have start and extent at least
  IterDomainBuilder() = delete;

  Val* start_ = nullptr;
  Val* extent_ = nullptr;
  Val* expanded_extent_ = nullptr;
  Val* stop_offset_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;

  // Only relevant at scheduling time or compile time.
  bool is_rfactor_domain_ = false;
  bool is_padded_dimension_ = false;
  c10::optional<int64_t> padded_to_size_ = c10::nullopt;
  bool is_mma_swizzled_ = false;
};

// Friends for direct access to split
class TensorDomain;
class ReplayTransformations;
class IndexReferenceReplay;
//! Simply a representation of an annotated 1D iterable from start to extent.
//! TensorDomains which represent how to iterate over a tensor is made up of
//! IterDomains to form an ND iterable. We directly set parallization strategies
//! on IterDomains.
class TORCH_CUDA_CU_API IterDomain : public Val {
 public:
  IterDomain(IrBuilderPasskey, const IterDomainBuilder& args);

  // Legacy constructor, TODO: should start moving to use IterDomainBuildArgs
  // constructor Same as the above but can set the offset of the stop point
  IterDomain(
      IrBuilderPasskey,
      Val* start,
      Val* extent,
      Val* expanded_extent,
      Val* stop_offset,
      ParallelType parallel_type,
      IterType iter_type,
      bool is_rfactor_domain,
      bool is_padded_dimension,
      c10::optional<int64_t> padded_to_size_,
      bool is_mma_swizzled);

  IterDomain(const IterDomain* src, IrCloner* ir_cloner);

  bool sameAs(const Statement* other) const override;

  //! Returns a new IterDomain matching properties of this
  //!
  //! This does NOT copy the is_rfactor_domain flag.
  IterDomain* cloneWithoutRFactor() const;

  //! Clone a vector domains
  static std::vector<IterDomain*> clone(
      const std::vector<IterDomain*>& domains);

  static IterDomain* merge(IterDomain* outer, IterDomain* inner);

  //! start_offset and stop_offset defines partial split. Only root
  //! domains are allowed to have non-zero start and stop offsets.
  static std::pair<IterDomain*, IterDomain*> split(
      IterDomain* in,
      Val* factor,
      bool inner_split,
      Val* start_offset = nullptr,
      Val* stop_offset = nullptr);

  //! trim_out_of_bounds controls how the values outside start and stop
  //! positions are treated. The option is only valid with root
  //! domains as non-root domains do not have valid start and stop
  //! positions.
  //!
  //! \param trim_out_of_bounds Trims [0, start_] and [-stop_offset_, extent_]
  static std::pair<IterDomain*, IterDomain*> split(
      IterDomain* in,
      Val* factor,
      bool inner_split,
      bool trim_out_of_bounds);

  bool isReduction() const {
    return getIterType() == IterType::Reduction;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return getIterType() == IterType::Broadcast;
  }

  bool isGather() const {
    return getIterType() == IterType::Gather;
  }

  bool isStride() const {
    return getIterType() == IterType::Stride;
  }

  bool isVectorComponent() const {
    return getIterType() == IterType::VectorComponent;
  }

  bool isParallelized() const {
    return getParallelType() != ParallelType::Serial;
  }

  //! Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return isParallelTypeBlockDim(getParallelType());
  }

  //! Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return isParallelTypeThreadDim(getParallelType());
  }

  //! Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  void parallelize(ParallelType t);

  ParallelType getParallelType() const {
    return parallel_type_;
  }

  IterType getIterType() const {
    return iter_type_;
  }

  Val* start() const {
    return start_;
  }

  Val* stop() const;

  Val* stopOffset() const;

  Val* extent() const {
    TORCH_INTERNAL_ASSERT(extent_ != nullptr);
    return extent_;
  }

  bool hasExpandedExtent() const {
    return expanded_extent_ != nullptr;
  }

  // Returns the expanded extent of a strided broadcast entry.
  Val* expandedExtent() const {
    TORCH_INTERNAL_ASSERT(
        hasExpandedExtent(),
        "Requested expanded extent, but none found on this dimension.");
    return expanded_extent_;
  }

  Val* getMaybeExpandedExtent() const {
    if (hasExpandedExtent()) {
      return expandedExtent();
    }
    return extent();
  }

  //! Dimension padding interface:
  //!  2 modes are currently supported:
  //!
  //!   - mode 1: if to_size is given as a positive number,
  //!      the dimension will be padded to the size so that
  //!      this iterdomain will be compile-time constant
  //!      size and it is the scheduler's responsibility
  //!      to ensure no input larger than the padded size
  //!      will be observed
  //!
  //!   - mode 2: if no to_size is given, this dimension
  //!      is "dynamically" padded to next smallest multiple
  //!      of a warp size, i.e. 17 padded to 32, 33 padded to 64
  //!      based on the given input.
  void padToMultipleOfWarp(c10::optional<int64_t> maybe_to_size = {}) {
    // Currently only restricted to TIDx to generate warp reduce
    TORCH_CHECK(
        parallel_type_ == ParallelType::TIDx,
        "padToMultipleOfWarp : warp padding only supported on TIDx parallel dimension");
    is_padded_dimension_ = true;
    if (maybe_to_size.has_value()) {
      if (maybe_to_size.value() > 0) {
        padded_to_size_ = maybe_to_size.value();
      }
    }
  }

  //! Indicates if this iterdomain had padding
  //!  dynamical or statical
  bool hasPaddingToMultipleOfWarp() const {
    return is_padded_dimension_;
  }

  //! Returns a concrete value if this iterdomain
  //!  has been padded to a statical size.
  c10::optional<int64_t> getMaybeSizeAfterPadding() const {
    return padded_to_size_;
  }

  //! True if range of iteration domain isn't across the full extent
  bool maybePartial() const;

  //! Check if IterDomain is a broadcast axis with compile-time
  //! known extent. This is the case with all size-1 IterDomains on
  //! a TensorView's root domain when the TensorView is created.
  bool isImplicitBroadcast() const {
    return isBroadcast() && extent()->isOneInt();
  }

  //! Split for stride by a given factor. It effectively does an inner
  //! split by the factor and sets the inner domain as a Stride
  //! domain.
  std::pair<IterDomain*, IterDomain*> stridedSplit(int factor);

  // TODO: Remove
  bool isSimple() const {
    return definition() == nullptr;
  }

  //! Marks that this id represents a
  //!  instruction loop, mma use only.
  //!
  //! An instruction loop can be considered a generalization of
  //!  vectorization. It also represents a loop that's implemented
  //!  by an instruction and should not be realized by codegen and
  //!  cannot be inlined with.
  //! As an example, if a mma macro, call it mma_eg implements:
  //!  for m in M
  //!    for n in N
  //!      for k in K
  //!         C[m,n] += A[m,k]*B[k,n],
  //! But the generated code should simply be:
  //!  mma_eg(C,A,B)
  //! without the 3 level loopnest, i.e. they're instruction loops.
  //!
  //! In the actual mma macros, the loopnests it implements is a
  //!  transformed version of above to match the mma swizzle.
  //!  So it's different implicit loopnest for different macros.
  //!  WarpMmaSwizzler will label the instruction loops case-by-case.
  bool isMma() const {
    return parallel_type_ == ParallelType::Mma;
  }

  //! Applies 2D swizzle on a rectangular tile defined by
  //!  a pair of iterdomains.
  static std::pair<IterDomain*, IterDomain*> swizzle(
      Swizzle2DType swizzle_type,
      IterDomain* in_x,
      IterDomain* in_y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  bool isMmaSwizzled() const {
    return is_mma_swizzled_;
  }

  //! Used by WarpMmaSwizzler, this is an utility for WarpMmaSwizzler
  //!  to lock the thread swizzled iterdomains.
  //! Only true for the iterdomains produced by WarpMmaSwizzler.
  //! Mma ops require specific swizzle patterns
  //!  and this label utility is to prevent any further transform on the
  //!  iterdomains involved in the swizzle so that the pattern remain correct in
  //!  generated code.
  //!
  //! Note:
  //!    Used only through WarpMmaSwizzler only and mma validation relies on
  //!    this
  //!  flag being set on the correct iterdomains.
  void toMmaSwizzled() {
    is_mma_swizzled_ = true;
  }

 protected:
  friend TensorDomain;
  friend ReplayTransformations;
  friend IndexReferenceReplay;

 private:
  //! Valid range is defined as [start:-stop_offset]
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;

  // Broadcast dimensions are assumed to be size 1 for the sake of code
  // generation. If a user though calls `expand` on a tensor that dimension is
  // still considered a broadcast dimension. However if we ever output that
  // dimension it should be a size dictated by the `expand` operation, and have
  // a stride of zero. Since this extent is important to track, but not
  // necessarily generate code for (still want loops on broadcast to be of size
  // 0), we simply store it separately from extent_. Having an expanded_extent_
  // is only allowed with broadcasted dimsneions. Only in this instance does it
  // make sense to have an expanded_extent_, because it's used when users are
  // expecting return tensors to have a physical domain. If a user simply
  // "broadcasts" an operation
  Val* const expanded_extent_ = nullptr;

  //! Distance of stop from the end
  Val* const stop_offset_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;
  bool is_rfactor_domain_ = false;
  bool is_padded_dimension_ = false;
  c10::optional<int64_t> padded_to_size_ = c10::nullopt;

  // TODO: Remove only used in kernel IR because IterDomains don't maintain
  // definitions of split/merge.
  bool is_simple_ = true;

  //! Tracks if this id represents a thread swizzled loop or
  //!   models an implicit loop within instructions. Should not make
  //!   any changes once an id is warp mapped.
  bool is_mma_swizzled_ = false;
};

//! TensorDomain holds a vector of IterDomains. It holds an IterDomain for every
//! logical axis in its associated tensor. TensorDomain does not directly hold
//! the Tensor it is associated with, and in theory could be associated with
//! multiple tensors. TensorDomain's primary responsibility is to provide a
//! mechanism to access history of transformations that were used to generate
//! it. This is done through the normal interaction of Expr/Val in Fusion. i.e.
//! if we want to know the previous operation generating a particular
//! TensorDomain we can simply call:
//!
//!     FusionGuard::getCurFusion()->definition(a_tensor_domain)
//!
//! which should give us an operation in the list [split, merge] or similar
//! operations that take in a TensorDomain, applies a transformation and outputs
//! a tensor domain.
class TORCH_CUDA_CU_API TensorDomain : public Val {
 public:
  explicit TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<bool> contiguity = std::vector<bool>());

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> domain,
      std::vector<bool> contiguity = std::vector<bool>());

  TensorDomain(
      IrBuilderPasskey,
      std::vector<IterDomain*> root_domain,
      std::vector<IterDomain*> rfactor_domain,
      std::vector<IterDomain*> domain,
      std::vector<bool> contiguity = std::vector<bool>());

  TensorDomain(const TensorDomain* src, IrCloner* ir_cloner);

  bool operator==(const TensorDomain& other) const;
  bool operator!=(const TensorDomain& other) const {
    return !(*this == other);
  }

  std::vector<IterDomain*>::size_type nDims() const {
    return domain_.size();
  }

  bool sameAs(const Statement* other) const override;

  static bool sameAs(
      const std::vector<IterDomain*>& lhs,
      const std::vector<IterDomain*>& rhs);

  const std::vector<IterDomain*>& domain() const {
    return domain_;
  }

  const std::vector<bool>& contiguity() const {
    return contiguity_;
  }

  void setContiguity(const std::vector<bool>& contig);

  std::string getContiguityString() const {
    std::stringstream ss;
    for (auto b : contiguity()) {
      ss << (b ? "t" : "f");
    }
    return ss.str();
  }

  bool hasReduction() const;
  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBlockBroadcast() const;
  bool hasGridBroadcast() const;
  bool hasBroadcast() const;
  bool hasRFactor() const;

  // Returns if rfactor domain only consists of id's of iter type.
  bool hasViewLikeRFactor() const;

  bool hasVectorize() const;

  c10::optional<unsigned int> getReductionAxis() const;

  const std::vector<IterDomain*>& noReductions() const {
    return no_reduction_domain_;
  }

  const std::vector<IterDomain*>& noBroadcasts() const {
    return no_bcast_domain_;
  }

  const std::vector<IterDomain*>& getRootDomain() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& getRFactorDomain() const {
    return rfactor_domain_;
  };

  // If rfactor domain exists in domain() return it, otherwise return root
  // domain.
  const std::vector<IterDomain*>& getMaybeRFactorDomain() const {
    return hasRFactor() ? getRFactorDomain() : getRootDomain();
  }

  void resetDomains() {
    no_reduction_domain_ = noReductions(domain_);
    no_bcast_domain_ = noBroadcasts(domain_);
    has_reduction_ = hasReduction(domain_);
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  IterDomain* axis(int i) const;

  size_t posOf(IterDomain* id) const;

  //! Returns a position of a root domain
  size_t rootPosOf(IterDomain* id) const;

  // Split "axis" into 2 axes
  //! inner_split dictates if the factor section of the split should be inside
  //! the
  //! remainer or outside.
  //! e.g. split(0, 4, inner_split = true) will result in:
  //! tv[id{extent}] -> tv[id{ceilDiv(extent, factor)}, id{factor}]
  //! e.g. split(0, 4, inner_split = false) will result in:
  //! tv[id{extent}] -> tv[id{factor}, id{ceilDiv(extent, factor)}]
  void split(
      int axis_,
      Val* factor,
      bool inner_split,
      bool trim_out_of_bounds = false);

  // Merge axis_o and axis_i. axis_i is the fast changing dimension. Resulting
  // axis is by default placed at original position axis_o
  void merge(int axis_o, int axis_i);

  // Reorder axes according to map[old_pos] = new_pos
  void reorder(const std::unordered_map<int, int>& old2new);

  //! Applies 2D swizzle on a rectangular tile defined by
  //!  a pair of iterdomains contained in this domain.
  void swizzle(
      Swizzle2DType swizzle_type,
      int x,
      int y,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  // Transform TensorView according to merge and split transformations
  TensorDomain* view(const AnalyzeViewResult& view_analysis);

  TensorDomain* flatten(int64_t start_dim, int64_t end_dim);

  static std::vector<IterDomain*> orderedAs(
      const std::vector<IterDomain*>& td,
      const std::unordered_map<int, int>& old2new);

  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);

  static bool hasBroadcast(const std::vector<IterDomain*>&);
  static bool hasReduction(const std::vector<IterDomain*>&);

  // Get a vector of the same size as the given rfactor domain filled with true
  // except at the expanded dims and the dims right before expanded dims.
  //
  // Note: [Contiguity and expand]
  //
  // In the context of nvfuser, if a non-last dimension is contiguous, it means
  // that we can collapse its index into the next dimension during indexing. If
  // the last dimension is contiguous, it means that the last dimension has
  // stride 1. For example, if we have a tensor T1 of shape (4, 5, 6, 7) with
  // contiguity (true, false, true, false), then we can iterate the tensor with
  // the following for loops:
  //
  //   for i in range(4*5):
  //     for j in range(6*7):
  //       element = T1[0, i, 0, j]
  //
  // Note that in the above loop, we are doing out-of-bound access of T1's dim 1
  // and 3, but still getting the correct result.
  //
  // For expanded tensor, for example we expand from T2 = shape(4, 5, 1, 6) to
  // T3 = shape(4, 5, 10, 6), then before we materialize the expand, the stride
  // of the expanded dimension must remain 0 because by definition, different
  // indexes in the expanded dimension must map to the same underlying element.
  // This means that the contiguity of the expanded dimension can not be true.
  // That is, is we access T3 with the following loop:
  //
  //   for i in range(4):
  //     for j in range(5):
  //       for k in range(10*6):
  //         element = T3[i, j, 0, k]
  //
  // we will not get the correct result, because T3 is not materialized and
  // T3[0, 0, 0, 7] is actually accessing T2[0, 1, 0, 1] which is different from
  // the correct element T3[0, 0, 1, 1] == T2[0, 0, 0, 1]. Similarly, since the
  // stride of the expanded dimension is always zero, the index of the dimension
  // before it can not be coalapsed into the expanded dimension as well. For
  // example, T3[0, 0, 11, 0] points to T2[0, 0, 0, 0], which is different from
  // the correct element T3[0, 1, 1, 0] == T2[0, 1, 0, 0].
  static std::vector<bool> getContiguousContiguity(
      const std::vector<IterDomain*>& rfactor_domain);

  // pair is in order where second is the consumer of first
  std::pair<TensorDomain*, TensorDomain*> rFactor(const std::vector<int>& axes);

 private:
  const std::vector<IterDomain*> root_domain_;
  std::vector<IterDomain*> domain_;
  std::vector<IterDomain*> no_bcast_domain_;
  std::vector<IterDomain*> no_reduction_domain_;
  const std::vector<IterDomain*> rfactor_domain_;
  std::vector<bool> contiguity_;
  bool has_reduction_;
};

//! Representation a split on an IterDomain by "factor"
//! inner_split dictates if the factor section of the split should be inside the
//! remainer or outside.
class TORCH_CUDA_CU_API Split : public Expr {
 public:
  // start_offset and stop_offset are used to express partial
  // split. Only the partial domain from start_offset to stop_offset
  // is split and the outer sub-regions are ignored. Note that both
  // start_offset and stop_offset are distance from the left end and
  // right ends, respectively.
  Split(
      IrBuilderPasskey,
      IterDomain* outer,
      IterDomain* inner,
      IterDomain* in,
      Val* factor,
      bool inner_split = true,
      Val* start_offset = nullptr,
      Val* stop_offset = nullptr);

  Split(const Split* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  IterDomain* outer() const {
    return outer_;
  }
  IterDomain* inner() const {
    return inner_;
  }
  IterDomain* in() const {
    return in_;
  }
  Val* factor() const {
    return factor_;
  }

  bool innerSplit() const {
    return inner_split_;
  }

  Val* startOffset() const {
    TORCH_INTERNAL_ASSERT(start_offset_ != nullptr);
    return start_offset_;
  }

  Val* stopOffset() const {
    TORCH_INTERNAL_ASSERT(stop_offset_ != nullptr);
    return stop_offset_;
  }

  //! Utility function to compute the split extent.
  static Val* extent(Val* in_extent, Val* start_offset, Val* stop_offset);

  bool sameAs(const Statement* other) const override;

 private:
  IterDomain* const outer_ = nullptr;
  IterDomain* const inner_ = nullptr;
  IterDomain* const in_ = nullptr;
  Val* const factor_ = nullptr;
  bool inner_split_ = true;
  //! Start position of the input domain. Non-zero means partial
  //! split. Elements until this offset are ignored.
  Val* const start_offset_ = nullptr;
  //! Offset from extent of the input domain. Non-zero means partial
  //! split. Elements after this offset are ignored.
  Val* const stop_offset_ = nullptr;
};

//! Merge the IterDomains outer and inner into one domain, outer and inner
//! dictate which will be traversed first (inner). Both IterDomains must be of
//! the same iter or reduction type, as well as the same parallelization
//! strategy if there is one
class TORCH_CUDA_CU_API Merge : public Expr {
 public:
  Merge(
      IrBuilderPasskey,
      IterDomain* out,
      IterDomain* outer,
      IterDomain* inner);

  Merge(const Merge* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  IterDomain* out() const {
    return out_;
  }
  IterDomain* outer() const {
    return outer_;
  }
  IterDomain* inner() const {
    return inner_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  IterDomain* const out_ = nullptr;
  IterDomain* const outer_ = nullptr;
  IterDomain* const inner_ = nullptr;
};

//! Applies 2D swizzles on a rectangular tile defined by 2 iterdomains.
class TORCH_CUDA_CU_API Swizzle2D : public Expr {
 public:
  Swizzle2D(
      IrBuilderPasskey,
      IterDomain* out_x,
      IterDomain* out_y,
      IterDomain* in_x,
      IterDomain* in_y,
      Swizzle2DType swizzle_type = Swizzle2DType::NoSwizzle,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  Swizzle2D(const Swizzle2D* src, IrCloner* ir_cloner);

  Expr* shallowCopy() const override;

  IterDomain* outX() const {
    return out_x_;
  }

  IterDomain* outY() const {
    return out_y_;
  }

  IterDomain* inX() const {
    return in_x_;
  }

  IterDomain* inY() const {
    return in_y_;
  }

  auto swizzleType() const {
    return swizzle_type_;
  }

  auto swizzleMode() const {
    return swizzle_mode_;
  }

  bool sameAs(const Statement* other) const override;

 private:
  // Output iterdomain pair corresponding
  //  to the original input iterdomain pair.
  IterDomain* const out_x_ = nullptr;
  IterDomain* const out_y_ = nullptr;

  // Input iterdomain pair.
  IterDomain* const in_x_ = nullptr;
  IterDomain* const in_y_ = nullptr;

  // The type of predefined 1-to-1 functions
  //  used for swizzling math.
  Swizzle2DType swizzle_type_ = Swizzle2DType::NoSwizzle;

  // Swizzle mode of this swizzle instance.
  // [Note on swizzle mode]
  // On the current implementations we support two modes of
  //  swizzle math, namely, data mode and loop mode.
  // `Data` mode swizzling is a swizzle that will change the
  //  data layout in shared memory, likely in global memory buffers
  //  as well in the future. see also IndexSwizzle in index_compute.cpp.
  //
  //  Most important use cases are transpose bank conflict removal, and mma
  //  swizzled shared memory layout. Example illustrated in 1D case:
  //
  // for (int i = 0; i<I; i++){
  //   # This is a `Data` mode swizzle.
  //  Tshared [swizzled(i)] = Tin[i];
  // }
  // # Now Tshared holds swizzled data, i.e. the data layout of
  //    Tshared does not map to Tin with affine relationships.
  //
  // for(int i=0;i<I;i++){
  //   Tout = Tshared[swizzled(i)];
  // }
  //
  // `Loop` mode swizzling does not affect the data layout of any buffer
  //   but only permutes the iteration order of serial or parallel loop.
  // This is useful when we want to designate non-affine mapping of thread
  //   to data or we want to generate non-affine loops.
  // Exampe illustrated in 1D case:
  //   for (int i = 0; i<I; i++){
  //     # This is a `Loop` mode swizzle
  //    Tshared [swizzled(i)] = Tin[swizzled(i)];
  //   }
  // # Now Tshared holds normal data, i.e. it still has
  //   the same data layout as if the swizzle wasn't there.
  //
  // # Consumers of Tshared does not need to know about the
  //   loop swizzle at previous op if not inlined.
  // for(int i=0;i<I;i++){
  //   Tout = Tshared[i];
  // }
  //  TODO: Loop swizzles eventually will be piped through in all mappings
  //  and replay of the fusion IR infrastructure.
  SwizzleMode swizzle_mode_ = SwizzleMode::Data;
};

//! Integer value which has a special name
//!
//! These could be:
//! - threadIdx.x
//! - blockIdx.y
//! - blockDim.z
//! - T3.stride[2]
//!
class TORCH_CUDA_CU_API NamedScalar : public Val {
 public:
  NamedScalar(IrBuilderPasskey passkey, std::string name, DataType dtype);

  NamedScalar(const NamedScalar* src, IrCloner* ir_cloner);

  const std::string& name() const {
    return name_;
  }

  bool sameAs(const Statement* other) const override;

  //! Return the named scalar extent of a parallel dimension (e.g. blockDim.x)
  //! WARNING: Only works with Fusion container at the moment
  static NamedScalar* getParallelDim(ParallelType p_type);

  //! Return the named scalar index of a parallel dimension (e.g. threadIdx.x)
  //! WARNING: Only works with Fusion container at the moment
  static NamedScalar* getParallelIndex(ParallelType p_type);

  //! Return the parallel type of this NamedScalar if it is an extent of a
  //! parallel dimension
  c10::optional<ParallelType> getParallelDim() const;

  //! Return the parallel type of this NamedScalar if it is an index of a
  //! parallel dimension
  c10::optional<ParallelType> getParallelIndex() const;

 private:
  std::string name_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
