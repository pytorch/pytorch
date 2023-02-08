#pragma once

#include <c10/macros/Export.h>

#include <fusion.h>
#include <ir_base_nodes.h>
#include <mma_type.h>
#include <parallel_type_bitmap.h>

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
  using Expr::Expr;

  FullOp(IrBuilderPasskey, Val* out, Val* fill_value);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "FullOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* getFillValue() const {
    return inputs().back();
  }
};

class TORCH_CUDA_CU_API SelectOp : public Expr {
 public:
  using Expr::Expr;

  SelectOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      IterDomain* select_id,
      Val* index);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "SelectOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  IterDomain* getSelectAxis() const {
    return attribute(0)->as<IterDomain>();
  }

  std::unordered_map<IterDomain*, Val*> getIndexOverridingMap() const {
    return {{getSelectAxis(), input(1)}};
  }
};

class TORCH_CUDA_CU_API IndexSelectOp : public Expr {
 public:
  using Expr::Expr;

  IndexSelectOp(
      IrBuilderPasskey,
      Val* out,
      Val* in1,
      int dim,
      IterDomain* select_id,
      Val* in3);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "IndexSelectOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  IterDomain* getSelectAxis() const {
    return attribute(0)->as<IterDomain>();
  }
  int dim() const {
    return attribute(1)->as<Attribute<int>>()->value;
  }
};

class TORCH_CUDA_CU_API TorchGatherOp : public Expr {
 public:
  using Expr::Expr;
  TorchGatherOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      int dim,
      IterDomain* select_id,
      Val* index);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "TorchGatherOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* lookupTv() const {
    return input(0)->as<TensorView>();
  }

  TensorView* indexTv() const {
    return input(1)->as<TensorView>();
  }

  int dim() const {
    return attribute(1)->as<Attribute<int>>()->value;
  }

  IterDomain* getSelectAxis() const {
    return attribute(0)->as<IterDomain>();
  }
};

class TORCH_CUDA_CU_API ScatterOp : public Expr {
 public:
  using Expr::Expr;
  ScatterOp(
      IrBuilderPasskey,
      ScatterOpType type,
      Val* out,
      Val* self,
      int dim,
      Val* index,
      Val* src,
      IterDomain* select_out_id);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "ScatterOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* selfTv() const {
    return input(0)->as<TensorView>();
  }

  TensorView* indexTv() const {
    return input(1)->as<TensorView>();
  }

  TensorView* srcTv() const {
    return input(2)->as<TensorView>();
  }

  int dim() const {
    return attribute(1)->as<Attribute<int>>()->value;
  }

  IterDomain* getOutputSelectAxis() const {
    return attribute(0)->as<IterDomain>();
  }

  ScatterOpType getScatterOpType() const {
    return attribute(2)->as<Attribute<ScatterOpType>>()->value;
  }
};

class TORCH_CUDA_CU_API IotaOp : public Expr {
 public:
  using Expr::Expr;

  IotaOp(IrBuilderPasskey, Val* out, Val* length, Val* start, Val* step);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "IotaOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  DataType dtype() const {
    return *start()->getDataType();
  }

  Val* length() const {
    return input(0);
  }

  Val* start() const {
    return input(1);
  }

  Val* step() const {
    return input(2);
  }
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
  using Expr::Expr;

  EyeOp(IrBuilderPasskey, Val* out, DataType dtype);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "EyeOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  DataType dtype() const {
    return attribute(0)->as<Attribute<DataType>>()->value;
  }
};

//! A specialization for Unary operations. Unary operations take in a single
//! input and produce a single output. Examples include:
//!   1) Casting operation i.e. float(a_val)
//!   2) Negation i.e. val * -1
//!   3) Reduction across a dimension i.e. val.sum(axis=2)
//!   4) split/merge
class TORCH_CUDA_CU_API UnaryOp : public Expr {
 public:
  using Expr::Expr;

  UnaryOp(IrBuilderPasskey, UnaryOpType type, Val* out, Val* in);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "UnaryOp";
  }

  virtual std::vector<EvaluatorValue> evaluate(
      const std::vector<EvaluatorValue>& inputs) const override;

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }

  UnaryOpType getUnaryOpType() const {
    return attribute(0)->as<Attribute<UnaryOpType>>()->value;
  }

 private:
  void printHelper(std::stringstream& ss, std::string input) const;
};

//! A specialization for Binary operations. Binary operations take in two inputs
//! and produce a single output. Examples include:
//!  1) Add/mul/div/mod/sub (A * B)
//!  2) LT (A < B)
class TORCH_CUDA_CU_API BinaryOp : public Expr {
 public:
  using Expr::Expr;

  BinaryOp(IrBuilderPasskey, BinaryOpType type, Val* out, Val* lhs, Val* rhs);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "BinaryOp";
  }

  virtual std::vector<EvaluatorValue> evaluate(
      const std::vector<EvaluatorValue>& inputs) const override;

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* lhs() const {
    return input(0);
  }
  Val* rhs() const {
    return input(1);
  }

  BinaryOpType getBinaryOpType() const {
    return attribute(0)->as<Attribute<BinaryOpType>>()->value;
  }

 private:
  void printHelper(
      std::stringstream& ss,
      int indent_size,
      std::string lhs,
      std::string rhs) const;
};

class TORCH_CUDA_CU_API TernaryOp : public Expr {
 public:
  using Expr::Expr;

  TernaryOp(
      IrBuilderPasskey,
      TernaryOpType type,
      Val* out,
      Val* in1,
      Val* in2,
      Val* in3);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "TernaryOp";
  }

  virtual std::vector<EvaluatorValue> evaluate(
      const std::vector<EvaluatorValue>& inputs) const override;

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* in1() const {
    return input(0);
  }
  Val* in2() const {
    return input(1);
  }
  Val* in3() const {
    return input(2);
  }

  TernaryOpType getTernaryOpType() const {
    return attribute(0)->as<Attribute<TernaryOpType>>()->value;
  }

 private:
  void printHelper(
      std::stringstream& ss,
      int indent_size,
      std::string in1,
      std::string in2,
      std::string in3) const;
};

//! A specialization for random number generator (RNG) operations. RNG
//! operations take in no tensor input and produce a single output.
class TORCH_CUDA_CU_API RNGOp : public Expr {
  size_t getOutputDims() const;

 public:
  struct Attributes {
    RNGOpType rtype;
    DataType dtype;
    int rng_offset;

    // TODO: Enable the following in C++20:
    // bool operator==(const Attributes &other) const = default;
    bool operator==(const Attributes& other) const {
      return rtype == other.rtype && dtype == other.dtype &&
          rng_offset == other.rng_offset;
    }
  };

  using Expr::Expr;

  RNGOp(
      IrBuilderPasskey,
      RNGOpType type,
      Val* out,
      DataType dtype,
      std::vector<Val*> parameters = {},
      int rng_offset = 0,
      Val* philox_index = nullptr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "RNGOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  RNGOpType getRNGOpType() const {
    return attribute(0)->as<Attribute<Attributes>>()->value.rtype;
  }

  DataType dtype() const {
    return attribute(0)->as<Attribute<Attributes>>()->value.dtype;
  }

  int getRNGOffset() const {
    return attribute(0)->as<Attribute<Attributes>>()->value.rng_offset;
  }

  void setRNGOffset(int val) {
    attribute(0)->as<Attribute<Attributes>>()->value.rng_offset = val;
  }

  std::vector<Val*> getParameters() const {
    return {inputs().begin() + getOutputDims(), inputs().end()};
  }

  std::vector<Val*> getShape() const {
    return {inputs().begin(), inputs().begin() + getOutputDims()};
  }

  Val* getPhiloxIndex() const {
    return attributeVal(1);
  }

  int getPhiloxMultiple() const {
    return dtype() == DataType::Double ? 2 : 4;
  }
};

//! Broadcast in to match out. is_broadcast_dims are relative to out. Where
//! is_broadcast_dims.size() == out->nDims().
class TORCH_CUDA_CU_API BroadcastOp : public Expr {
 public:
  using Expr::Expr;

  //! \param out The output tensor
  //! \param in The input tensor
  //! \param is_broadcast_dims True when output dim is a new broadcast domain
  BroadcastOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<bool> is_broadcast_dims);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "BroadcastOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }

  bool isBroadcastDim(size_t dim) const {
    return getBroadcastDimFlags().at(dim);
  }

  //! The same list passed to the broadcast arithmetic op. Each
  //! element corresponds to an IterDomain of the output tensor and is
  //! true when the IterDomain is a new broadcast domain. Note
  //! that the output tensor may have other broadcast domains whose
  //! flags are false because the input tensor may already have
  //! broadcast domains.
  const std::vector<bool>& getBroadcastDimFlags() const {
    return attribute(0)->as<Attribute<std::vector<bool>>>()->value;
  }
};

//! Squeeze in to match out. is_squeeze_dims are relative to in. Where
//! is_squeeze_dims.size() == in->nDims(). Squeeze is the opposite of
//! broadcast.
class TORCH_CUDA_CU_API SqueezeOp : public Expr {
 public:
  using Expr::Expr;

  //! \param out The output tensor
  //! \param in The input tensor
  //! \param is_squeeze_dims True when input dim is a removed broadcast domain
  SqueezeOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<bool> is_broadcast_dims);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "SqueezeOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }

  bool isSqueezeDim(size_t dim) const {
    return getSqueezeDimFlags().at(dim);
  }

  //! The same list passed to the squeeze arithmetic op. Each
  //! element corresponds to an IterDomain of the input tensor and is
  //! true when the IterDomain is a broadcast domain that is removed in the
  //! output. Note that the output tensor may still contain broadcast domains
  //! because the input tensor may have broadcast domains that we don't want to
  //! remove (false flag).
  const std::vector<bool>& getSqueezeDimFlags() const {
    return attribute(0)->as<Attribute<std::vector<bool>>>()->value;
  }
};

//! Reduction operation. Out is first initialized to _init. Then
//! reduction_op_type is used to update out as out = reductionOp(out, in).
//! Output's axes marked as reduction will be reduced to produce an output
//! tensor. The output tensors size will be the size of all
//! non-reduction/non-broadcast dimensions.
class TORCH_CUDA_CU_API ReductionOp : public Expr {
 public:
  using Expr::Expr;

  ReductionOp(
      IrBuilderPasskey,
      BinaryOpType reduction_op_type,
      Val* init,
      Val* out,
      Val* in,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "ReductionOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }
  Val* init() const {
    return attributeVal(0);
  }

  BinaryOpType getReductionOpType() const {
    return attribute(1)->as<Attribute<BinaryOpType>>()->value;
  }

  bool isAllreduce() const {
    return attribute(2)->as<Attribute<bool>>()->value;
  }
};

//! Grouped reduction operation for horizontal fusions. It works like
//! batched GEMMs in the sense that multiple independent reductions are
//! performed together. The main benefit is when reducing tensors across thread
//! blocks, a single grid sync can be done for all individual
//! reductions. As grid sync is very expensive, this can be a
//! significant performance impact.
class TORCH_CUDA_CU_API GroupedReductionOp : public Expr {
 public:
  using Expr::Expr;

  GroupedReductionOp(
      IrBuilderPasskey,
      std::vector<BinaryOpType> reduction_op_types,
      std::vector<Val*> init,
      std::vector<Val*> out,
      std::vector<Val*> in,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "GroupedReductionOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! Number of expressions grouped horizontally. It does not reflect
  //! iteration grouping.
  size_t numHorizontallyGroupedExprs() const {
    return getReductionOpTypes().size();
  }

  std::vector<Val*> initVals() const {
    auto size = numHorizontallyGroupedExprs();
    std::vector<Val*> result;
    result.reserve(size);
    for (auto i : c10::irange(2, 2 + size)) {
      result.emplace_back(attribute(i)->as<Val>());
    }
    return result;
  }

  Val* initVal(size_t index) const {
    return attributeVal(2 + index);
  }

  const std::vector<BinaryOpType>& getReductionOpTypes() const {
    return attribute(0)->as<Attribute<std::vector<BinaryOpType>>>()->value;
  }

  BinaryOpType getReductionOpType(size_t index) const {
    return getReductionOpTypes().at(index);
  }

  bool isAllreduce() const {
    return attribute(1)->as<Attribute<bool>>()->value;
  }

  //! Return the index of the corresponding reduction expression for
  //! a given output val.
  int getExprIndexOfOutput(Val* output_val) const;
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
  using Expr::Expr;
  static constexpr int kNumAttrs = 4;

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

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "WelfordOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return outputTriplet().avg();
  }

  Val* in() const {
    return inputTriplet().avg();
  }

  WelfordTriplet outputTriplet() const {
    return WelfordTriplet(outAvg(), outVar(), outN());
  }

  Val* outAvg() const {
    return output(0);
  }

  Val* outVar() const {
    return output(1);
  }

  Val* outN() const {
    return output(2);
  }

  WelfordTriplet inputTriplet() const {
    return WelfordTriplet(inAvg(), inVar(), inN());
  }

  Val* inAvg() const {
    return input(0);
  }

  Val* inVar() const {
    return input(1);
  }

  Val* inN() const {
    return input(2);
  }

  WelfordTriplet initTriplet() const {
    return WelfordTriplet(initAvg(), initVar(), initN());
  }

  Val* initAvg() const {
    return attributeVal(0);
  }

  Val* initVar() const {
    return attributeVal(1);
  }

  Val* initN() const {
    return attributeVal(2);
  }

  bool singleValue() const {
    return inN()->isOneInt();
  }

  bool hasInit() const {
    return !initN()->isZeroInt();
  }

  //! True if using the fused reduction kernel (not implemented yet)
  bool isAllreduce() const {
    return attribute(3)->as<Attribute<bool>>()->value;
  }

  std::vector<Val*> getInitVals() const;

  //! Return the init val for an output val
  Val* getInitValOfOutput(Val* output_val) const;
};

class TORCH_CUDA_CU_API GroupedWelfordOp : public Expr {
 public:
  using Expr::Expr;

  GroupedWelfordOp(
      IrBuilderPasskey,
      std::vector<WelfordTriplet> output_vals,
      std::vector<WelfordTriplet> input_vals,
      std::vector<WelfordTriplet> init_vals,
      bool is_allreduce = false);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "GroupedWelfordOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  //! Number of expressions grouped horizontally. It does not reflect
  //! iteration grouping. As horizontal grouping is not supported,
  //! this always returns 1.
  size_t numHorizontallyGroupedExprs() const {
    return 1;
  }

  Val* out(size_t index) const {
    return outAvg(index);
  }

  Val* in(size_t index) const {
    return inAvg(index);
  }

  std::vector<WelfordTriplet> outputVals() const {
    std::vector<WelfordTriplet> result;
    auto size = outputs().size() / 3;
    result.reserve(size);
    for (auto i : c10::irange(size)) {
      result.emplace_back(outAvg(i), outVar(i), outN(i));
    }
    return result;
  }

  std::vector<WelfordTriplet> inputVals() const {
    std::vector<WelfordTriplet> result;
    auto size = inputs().size() / 3;
    result.reserve(size);
    for (auto i : c10::irange(size)) {
      result.emplace_back(inAvg(i), inVar(i), inN(i));
    }
    return result;
  }

  std::vector<WelfordTriplet> initVals() const {
    std::vector<WelfordTriplet> result;
    auto size = inputs().size() / 3;
    result.reserve(size);
    for (auto i : c10::irange(size)) {
      result.emplace_back(initAvg(i), initVar(i), initN(i));
    }
    return result;
  }

  Val* outAvg(size_t index) const {
    return output(index * 3);
  }

  Val* outVar(size_t index) const {
    return output(index * 3 + 1);
  }

  Val* outN(size_t index) const {
    return output(index * 3 + 2);
  }

  Val* inAvg(size_t index) const {
    return input(index * 3);
  }

  Val* inVar(size_t index) const {
    return input(index * 3 + 1);
  }

  Val* inN(size_t index) const {
    return input(index * 3 + 2);
  }

  Val* initAvg(size_t index) const {
    return attributeVal(1 + index * 3);
  }

  Val* initVar(size_t index) const {
    return attributeVal(2 + index * 3);
  }

  Val* initN(size_t index) const {
    return attributeVal(3 + index * 3);
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
    return attribute(0)->as<Attribute<bool>>()->value;
  }
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

  using Expr::Expr;

  MmaOp(IrBuilderPasskey, Val* out, Val* in_a, Val* in_b, Val* init);

  MmaOp(
      IrBuilderPasskey,
      Val* out,
      Val* in_a,
      Val* in_b,
      Val* init,
      OptionsInMma options);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "MmaOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* inA() const {
    return input(0);
  }

  Val* inB() const {
    return input(1);
  }

  Val* init() const {
    return attributeVal(0);
  }

  const auto& options() const {
    return attribute(1)->as<Attribute<OptionsInMma>>()->value;
  }

  auto accStride() const {
    return options().accumulator_stride;
  }

  void configureOptions(MmaOptions options);
};

class TORCH_CUDA_CU_API TransposeOp : public Expr {
 public:
  using Expr::Expr;

  TransposeOp(
      IrBuilderPasskey,
      TensorView* out,
      TensorView* in,
      std::vector<int64_t> new2old);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "TransposeOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  const std::vector<int64_t>& new2old() const {
    return attribute(0)->as<Attribute<std::vector<int64_t>>>()->value;
  }

  std::vector<int64_t> old2new() const;
};

class TORCH_CUDA_CU_API ExpandOp : public Expr {
 public:
  using Expr::Expr;

  ExpandOp(
      IrBuilderPasskey,
      TensorView* out,
      TensorView* in,
      std::vector<Val*> _expanded_extents);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "ExpandOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  TensorView* out() const {
    return output(0)->as<TensorView>();
  }

  TensorView* in() const {
    return input(0)->as<TensorView>();
  }

  std::vector<Val*> expanded_extents() const {
    return {inputs().begin() + 1, inputs().end()};
  }
};

//! Shift
class TORCH_CUDA_CU_API ShiftOp : public Expr {
 public:
  using Expr::Expr;

  //! \param out
  //! \param in
  //! \param offsets
  ShiftOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<int> offsets,
      std::vector<int> pad_width);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "ShiftOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }

  int offset(size_t dim) const {
    return offsets().at(dim);
  }

  //! Each of the root axes is shifted by the corresponding value of
  //! offsets. The sign of each value indicates the direction of shifting.
  const std::vector<int>& offsets() const {
    return attribute(0)->as<Attribute<std::vector<int>>>()->value;
  }

  const std::vector<int>& padWidth() const {
    return attribute(1)->as<Attribute<std::vector<int>>>()->value;
  }

  bool hasPadding() const {
    return std::any_of(padWidth().begin(), padWidth().end(), [](const auto p) {
      return p > 0;
    });
  }
};

//! Gather a window around each element.
class TORCH_CUDA_CU_API GatherOp : public Expr {
 public:
  using Expr::Expr;

  GatherOp(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      std::vector<int> window_shape,
      std::vector<std::vector<int>> pad_width);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "GatherOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }
  Val* in() const {
    return input(0);
  }

  //! Shape of a window gathered for each element.
  const auto& windowShape() const {
    return attribute(0)->as<Attribute<std::vector<int>>>()->value;
  }

  //! Returns the gather axis that corresponds to an input axis
  int gatherAxis(int axis) const;

  //! The size of zero-padding of each axis.
  const auto& padWidth() const {
    return attribute(1)->as<Attribute<std::vector<std::vector<int>>>>()->value;
  }

  bool hasPadding() const {
    return std::any_of(padWidth().begin(), padWidth().end(), [](const auto& p) {
      return p[0] > 0 || p[1] > 0;
    });
  }
};

class TORCH_CUDA_CU_API ViewAsScalar : public Expr {
 public:
  using Expr::Expr;

  ViewAsScalar(
      IrBuilderPasskey,
      Val* out,
      Val* in,
      IterDomain* vector_id,
      Val* index = nullptr);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "ViewAsScalar";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }

  // The IterDomain of type VectorComponent newly appended to the output
  IterDomain* vector_id() const {
    return attribute(0)->as<IterDomain>();
  }

  // The index that vector_id_ is lowered into
  Val* index() const {
    return attributeVal(1);
  }
};

class TORCH_CUDA_CU_API ViewOp : public Expr {
 public:
  using Expr::Expr;

  ViewOp(IrBuilderPasskey, Val* out, Val* in);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "ViewOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }
};

//! This operator explicitly models data movement between
//!   state spaces on GPU. Currently the modeled state spaces include
//!   global memory, shared memory and register.
//!
//! The main usage of this op is to facilitate generation of hardware
//!   accelerated memory ops, i.e. ldmatrix, cp.async and more to come.
class TORCH_CUDA_CU_API LoadStoreOp : public Expr {
 public:
  using Expr::Expr;

  LoadStoreOp(IrBuilderPasskey, LoadStoreOpType op_type, Val* out, Val* in);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "LoadStoreOp";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  Val* out() const {
    return output(0);
  }

  Val* in() const {
    return input(0);
  }

  LoadStoreOpType opType() const {
    return attribute(0)->as<Attribute<LoadStoreOpType>>()->value;
  }
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

  NVFUSER_DECLARE_CLONE

  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

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

  bool isGatherScatter() const {
    return getIterType() == IterType::GatherScatter;
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

  NVFUSER_DECLARE_CLONE

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

  std::string toString(int indent_size = 0) const override;

  std::string toInlineString(int indent_size = 0) const override;

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
  using Expr::Expr;

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

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "Split";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  IterDomain* outer() const {
    return output(0)->as<IterDomain>();
  }
  IterDomain* inner() const {
    return output(1)->as<IterDomain>();
  }
  IterDomain* in() const {
    return input(0)->as<IterDomain>();
  }
  Val* factor() const {
    return attributeVal(0);
  }

  bool innerSplit() const {
    return attribute(1)->as<Attribute<bool>>()->value;
  }

  //! Start position of the input domain. Non-zero means partial
  //! split. Elements until this offset are ignored.
  Val* startOffset() const {
    TORCH_INTERNAL_ASSERT(attributeVal(2) != nullptr);
    return attributeVal(2);
  }

  //! Offset from extent of the input domain. Non-zero means partial
  //! split. Elements after this offset are ignored.
  Val* stopOffset() const {
    TORCH_INTERNAL_ASSERT(attributeVal(3) != nullptr);
    return attributeVal(3);
  }

  //! Utility function to compute the split extent.
  static Val* extent(Val* in_extent, Val* start_offset, Val* stop_offset);
};

//! Merge the IterDomains outer and inner into one domain, outer and inner
//! dictate which will be traversed first (inner). Both IterDomains must be of
//! the same iter or reduction type, as well as the same parallelization
//! strategy if there is one
class TORCH_CUDA_CU_API Merge : public Expr {
 public:
  using Expr::Expr;

  Merge(
      IrBuilderPasskey,
      IterDomain* out,
      IterDomain* outer,
      IterDomain* inner);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "Merge";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  IterDomain* out() const {
    return output(0)->as<IterDomain>();
  }
  IterDomain* outer() const {
    return input(0)->as<IterDomain>();
  }
  IterDomain* inner() const {
    return input(1)->as<IterDomain>();
  }
};

//! Applies 2D swizzles on a rectangular tile defined by 2 iterdomains.
class TORCH_CUDA_CU_API Swizzle2D : public Expr {
 public:
  using Expr::Expr;

  Swizzle2D(
      IrBuilderPasskey,
      IterDomain* out_x,
      IterDomain* out_y,
      IterDomain* in_x,
      IterDomain* in_y,
      Swizzle2DType swizzle_type = Swizzle2DType::NoSwizzle,
      SwizzleMode swizzle_mode = SwizzleMode::Data);

  NVFUSER_DECLARE_CLONE_AND_CREATE

  virtual const char* getOpString() const override {
    return "Swizzle2D";
  }

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;

  // Output iterdomain pair corresponding
  //  to the original input iterdomain pair.
  IterDomain* outX() const {
    return output(0)->as<IterDomain>();
  }

  IterDomain* outY() const {
    return output(1)->as<IterDomain>();
  }

  // Input iterdomain pair.
  IterDomain* inX() const {
    return input(0)->as<IterDomain>();
  }

  IterDomain* inY() const {
    return input(1)->as<IterDomain>();
  }

  // The type of predefined 1-to-1 functions
  //  used for swizzling math.
  auto swizzleType() const {
    return attribute(0)->as<Attribute<Swizzle2DType>>()->value;
  }

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
  auto swizzleMode() const {
    return attribute(1)->as<Attribute<SwizzleMode>>()->value;
  }
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

  NVFUSER_DECLARE_CLONE

  const std::string& name() const {
    return name_;
  }

  bool sameAs(const Statement* other) const override;

  std::string toString(int indent_size = 0) const override {
    return name_;
  }

  std::string toInlineString(int indent_size = 0) const override {
    return name_;
  }

  //! Check if this is something like T0.size[1]
  bool isTensorSize() const;

  //! Check if this is something like T0.stride[1]
  bool isTensorStride() const;

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
