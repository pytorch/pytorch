#pragma once

#include <torch/csrc/jit/codegen/cuda/type.h>

// TODO(kir): remove these once the Kernel IR is separated from Fusion IR
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_internal_nodes.h>

#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

class IrBuilder;

//! Token used to restrict the access to Kernel IR constructors
//!
//! Granular "friendship" token, used to implement the "passkey" idiom:
//! https://www.spiria.com/en/blog/desktop-software/passkey-idiom-and-better-friendship-c
//! https://arne-mertz.de/2016/10/passkey-idiom
//!
class Passkey {
  friend class IrBuilder;
  Passkey() = default;
};

class TORCH_CUDA_CU_API NamedScalar : public Val {
 public:
  NamedScalar(Passkey, std::string name, DataType dtype)
      : Val(ValType::KirNamedScalar, dtype, true, true),
        name_(std::move(name)) {}

  explicit NamedScalar(Passkey, const fuser::cuda::NamedScalar* node)
      : Val(node), name_(node->name()) {}

  const std::string& name() const {
    return name_;
  }

  // Return the named scalar extent of a parallel dimension (e.g. blockDim.x)
  static NamedScalar* getParallelDim(ParallelType p_type);

  // Return the named scalar index of a parallel dimension (e.g. threadIdx.x)
  static NamedScalar* getParallelIndex(ParallelType p_type);

  // Return the parallel type of this NamedScalar if it is an extent of a
  // parallel dimension
  c10::optional<ParallelType> getParallelDim() const;

  // Return the parallel type of this NamedScalar if it is an index of a
  // parallel dimension
  c10::optional<ParallelType> getParallelIndex() const;

 private:
  std::string name_;
};

class TORCH_CUDA_CU_API Bool : public Val {
 public:
  explicit Bool(Passkey, const c10::optional<bool>& value)
      : Val(ValType::KirScalar, DataType::Bool, true, true),
        maybe_value_(value) {}

  explicit Bool(Passkey, const fuser::cuda::Bool* node)
      : Val(node), maybe_value_(node->value()) {}

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<bool> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<bool> maybe_value_;
};

class TORCH_CUDA_CU_API Float : public Val {
 public:
  using ScalarType = double;

  explicit Float(Passkey, const c10::optional<ScalarType>& value)
      : Val(ValType::KirScalar, DataType::Float, true, true),
        maybe_value_(value) {}

  explicit Float(Passkey, const fuser::cuda::Float* node)
      : Val(node), maybe_value_(node->value()) {}

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<ScalarType> maybe_value_;
};

class TORCH_CUDA_CU_API Half : public Val {
 public:
  explicit Half(Passkey, const c10::optional<float>& value)
      : Val(ValType::KirScalar, DataType::Half, true, true),
        maybe_value_(value) {}

  explicit Half(Passkey, const fuser::cuda::Half* node)
      : Val(node), maybe_value_(node->value()) {}

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<float> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<float> maybe_value_;
};

class TORCH_CUDA_CU_API Int : public Val {
 public:
  using ScalarType = int64_t;

  explicit Int(Passkey, const c10::optional<ScalarType>& value)
      : Val(ValType::KirScalar, DataType::Int, true, true),
        maybe_value_(value) {}

  explicit Int(
      Passkey,
      const fuser::cuda::Int* node,
      bool /*avoid_zero_ambiguity*/)
      : Val(node), maybe_value_(node->value()) {}

  bool isSymbolic() const {
    return !(maybe_value_.has_value());
  }
  bool isConst() const {
    return maybe_value_.has_value();
  }
  c10::optional<ScalarType> value() const {
    return maybe_value_;
  }

 private:
  const c10::optional<ScalarType> maybe_value_;
};

class TORCH_CUDA_CU_API IterDomain : public Val {
 public:
  IterDomain(Passkey, Val* start, Val* extent);

  explicit IterDomain(Passkey, const fuser::cuda::IterDomain* iter_domain);

  bool isReduction() const {
    return getIterType() == IterType::Reduction;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return getIterType() == IterType::BroadcastWithStride ||
        getIterType() == IterType::BroadcastWithoutStride;
  }

  bool isParallelized() const {
    return getParallelType() != ParallelType::Serial;
  }

  // Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return (
        getParallelType() == ParallelType::BIDz ||
        getParallelType() == ParallelType::BIDy ||
        getParallelType() == ParallelType::BIDx);
  }

  // Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return (
        getParallelType() == ParallelType::TIDz ||
        getParallelType() == ParallelType::TIDy ||
        getParallelType() == ParallelType::TIDx);
  }

  // Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  ParallelType getParallelType() const {
    return parallel_type_;
  }

  IterType getIterType() const {
    return iter_type_;
  }

  Val* start() const {
    return start_;
  }

  Val* extent() const;

  Val* rawExtent() const {
    return extent_;
  }

 private:
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;
  bool is_rfactor_domain_ = false;
};

class TORCH_CUDA_CU_API TensorDomain : public Val {
 public:
  explicit TensorDomain(Passkey, std::vector<IterDomain*> domain);

  explicit TensorDomain(
      Passkey,
      const fuser::cuda::TensorDomain* tensor_domain);

  std::vector<IterDomain*>::size_type nDims() const {
    return domain_.size();
  }

  const std::vector<IterDomain*>& domain() const {
    return domain_;
  }

  const std::vector<bool>& contiguity() const {
    return contiguity_;
  }

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
  bool hasBroadcast() const;
  bool hasRFactor() const;

  const std::vector<IterDomain*>& noReductions() const {
    return no_reduction_domain_;
  }

  const std::vector<IterDomain*>& noBroadcasts() const {
    return no_bcast_domain_;
  }

  const std::vector<IterDomain*>& rootDomain() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& rfactorDomain() const {
    return rfactor_domain_;
  };

  void resetDomains() {
    no_reduction_domain_ = noReductions(domain_);
    no_bcast_domain_ = noBroadcasts(domain_);
  }

  IterDomain* axis(int i) const;

  // TODO(kir): overloading non-static and static methods is not a good idea
  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);

 private:
  std::vector<IterDomain*> root_domain_;
  std::vector<IterDomain*> domain_;
  std::vector<IterDomain*> no_bcast_domain_;
  std::vector<IterDomain*> no_reduction_domain_;
  std::vector<IterDomain*> rfactor_domain_;
  const std::vector<bool> contiguity_;
};

class TORCH_CUDA_CU_API TensorView : public Val {
 public:
  explicit TensorView(Passkey, const fuser::cuda::TensorView* tv);

  TensorDomain* domain() const {
    return domain_;
  }

  MemoryType memoryType() const {
    return memory_type_;
  }

  const fuser::cuda::TensorView* fuserTv() const {
    TORCH_INTERNAL_ASSERT(fuser_tv_ != nullptr);
    return fuser_tv_;
  }

 private:
  TensorDomain* domain_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;

  // TODO(kir): remove temporary hack
  const fuser::cuda::TensorView* fuser_tv_ = nullptr;
};

class TORCH_CUDA_CU_API UnaryOp : public Expr {
 public:
  UnaryOp(Passkey, UnaryOpType type, Val* out, Val* in);

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

  UnaryOpType getUnaryOpType() const {
    return unary_op_type_;
  }

 private:
  const UnaryOpType unary_op_type_;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

class TORCH_CUDA_CU_API BinaryOp : public Expr {
 public:
  BinaryOp(Passkey, BinaryOpType type, Val* out, Val* lhs, Val* rhs);

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

 private:
  const BinaryOpType binary_op_type_;
  Val* const out_ = nullptr;
  Val* const lhs_ = nullptr;
  Val* const rhs_ = nullptr;
};

class TORCH_CUDA_CU_API TernaryOp : public Expr {
 public:
  TernaryOp(
      Passkey,
      TernaryOpType type,
      Val* out,
      Val* in1,
      Val* in2,
      Val* in3);

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

 private:
  const TernaryOpType ternary_op_type_;
  Val* const out_ = nullptr;
  Val* const in1_ = nullptr;
  Val* const in2_ = nullptr;
  Val* const in3_ = nullptr;
};

class TORCH_CUDA_CU_API ReductionOp : public Expr {
 public:
  ReductionOp(
      Passkey,
      BinaryOpType reduction_op_type,
      Val* init,
      Val* out,
      Val* in,
      Bool* pred = nullptr);

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

  Val* init() const {
    return init_;
  }

  Bool* pred() const {
    return pred_;
  }

  BinaryOpType getReductionOpType() const {
    return reduction_op_type_;
  }

  std::unordered_map<ParallelType, IterDomain*, TypeHash>
  getParallelReductionDomains() const;

 private:
  std::vector<IterDomain*> getReductionDomains() const;

 private:
  const BinaryOpType reduction_op_type_;
  Val* const init_ = nullptr;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
  Bool* const pred_ = nullptr;
};

class TORCH_CUDA_CU_API TensorIndex : public Val {
 public:
  TensorIndex(
      Passkey,
      const fuser::cuda::TensorView* view,
      std::vector<Val*> indices);

  std::vector<Val*>::size_type nDims() const {
    return indices_.size();
  }

  Val* index(int i) const;

  const std::vector<Val*>& indices() const {
    return indices_;
  }

  const TensorView* view() const {
    return view_;
  }

 private:
  const TensorView* view_ = nullptr;
  std::vector<Val*> indices_;
};

class TORCH_CUDA_CU_API BroadcastOp : public Expr {
 public:
  BroadcastOp(Passkey, Val* out, Val* in);

  Val* out() const {
    return out_;
  }

  Val* in() const {
    return in_;
  }

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

// Allocate is a lower level Node that describes a buffer of memory that
// is required as an intermediate within a kernel.  The extent is the expression
// of the size of the buffer that is generated from the TensorView that
// describes the output of an operation.
//
// TODO: The components of Allocate like Type and Name could be separated from
// the the assocated TensorView.  Perhaps that is more appropriate?
class TORCH_CUDA_CU_API Allocate : public Expr {
 public:
  explicit Allocate(
      Passkey,
      Val* buffer,
      MemoryType memory_type = MemoryType::Local,
      Val* size = nullptr,
      bool zero_init = false);

  Val* buffer() const {
    return buffer_;
  }

  MemoryType getMemoryType() const {
    return memory_type_;
  }

  Val* size() const {
    return size_;
  }

  bool zeroInit() const {
    return zero_init_;
  }

  DataType buffer_type() const {
    return buffer_->getDataType().value();
  }

  Allocate* alias() const {
    return alias_;
  }

  void setAlias(Allocate* alias) {
    TORCH_INTERNAL_ASSERT(alias->getMemoryType() == memory_type_);
    alias_ = alias;
  }

 private:
  Val* buffer_ = nullptr;
  MemoryType memory_type_ = MemoryType::Local;
  Val* size_ = nullptr;
  bool zero_init_ = false;

  // This alias tracks the next Allocate node in a linked chain of aliases
  // If the alias is nullptr, then the Allocate node uses memory in the kernel
  Allocate* alias_ = nullptr;
};

// Sync represents __syncthreads barrier for block level coordination.
class TORCH_CUDA_CU_API Sync : public Expr {
 public:
  explicit Sync(Passkey, bool war_sync = false);

  bool isWarHazardSync() const {
    return war_sync_;
  }

 private:
  // TODO: war_sync_ is only used for testing/validation purposes.
  bool war_sync_ = false;
};

// TODO(kir): promote to IR node
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API Scope {
 public:
  Scope() = default;

  const std::vector<Expr*>& exprs() const {
    return exprs_;
  }

  void push_back(Expr* e) {
    exprs_.push_back(e);
  }

  void insert(size_t pos, Expr* expr) {
    exprs_.insert(exprs_.begin() + pos, expr);
  }

  void erase(size_t pos) {
    exprs_.erase(exprs_.begin() + pos);
  }

  bool empty() const {
    return exprs_.empty();
  }

  auto size() const {
    return exprs_.size();
  }

  auto& operator[](size_t i) {
    return exprs_[i];
  }

  auto& operator[](size_t i) const {
    return exprs_[i];
  }

  // Insert expr before ref
  void insert_before(Expr* ref, Expr* expr);

  // Insert expr after ref
  void insert_after(Expr* ref, Expr* expr);

  bool contains(Expr* expr) const;

  void erase(Expr* ref);

  void clear();

 private:
  std::vector<Expr*> exprs_;
};

// ForLoop provides scoping around an int iterator from 0 to range. Exprs placed
// in its body are considered inside the scope of the for loop. In the future
// the implementation should look quite different so that we can do proper
// dependency annalysis like in Fusion.
//
// TODO(kir): this is not a real expression
//
class TORCH_CUDA_CU_API ForLoop : public Expr {
 public:
  ForLoop(Passkey, Val* index, IterDomain* iter_domain, Expr* parent_scope);

  Val* index() const {
    return index_;
  }

  IterDomain* iter_domain() const {
    return iter_domain_;
  }

  Scope& body() {
    return body_;
  }

  const Scope& body() const {
    return body_;
  }

  Expr* parentScope() const {
    return parent_scope_;
  }

  void setParentScope(Expr* scope);

 private:
  Val* const index_ = nullptr;
  IterDomain* const iter_domain_;
  Scope body_;
  Expr* parent_scope_ = nullptr;
};

// IfThenElse provides scoping for an boolean operator. Exprs placed in its body
// are considered inside the scope of the if statement. In the future the
// implementation should look quite different so that we can do proper
// dependency annalysis like in Fusion.
//
// TODO(kir): this is not a real expression
//
class TORCH_CUDA_CU_API IfThenElse : public Expr {
 public:
  explicit IfThenElse(Passkey, Bool* cond, Expr* parent_scope);

  Bool* cond() const {
    return cond_;
  }

  Scope& thenBody() {
    return then_body_;
  }
  const Scope& thenBody() const {
    return then_body_;
  }

  Scope& elseBody() {
    return else_body_;
  }

  const Scope& elseBody() const {
    return else_body_;
  }

  bool hasElse() const {
    return !else_body_.empty();
  }

  Expr* parentScope() const {
    return parent_scope_;
  }

  void setParentScope(Expr* scope);

 private:
  Bool* const cond_ = nullptr;
  Scope then_body_;
  Scope else_body_;
  Expr* parent_scope_ = nullptr;
};

// Grid reduction operation, this node is used only after lowering a fusion to
// explicitly mark a grid reduction and the buffer allocation needed to do it.
// This node provides FusionExecutor the information it needs to allocate the
// reduction and sync buffers.
class TORCH_CUDA_CU_API GridReduction : public Expr {
 public:
  explicit GridReduction(Passkey, ReductionOp* reduction_op);

  GridReduction(
      Passkey,
      ReductionOp* reduction_op,
      Allocate* reduction_buffer,
      Allocate* sync_buffer,
      Bool* pred = nullptr);

  ReductionOp* reduction_op() const {
    return reduction_op_;
  }

  Allocate* reduction_buffer() const {
    return reduction_buffer_;
  }

  Allocate* sync_buffer() const {
    return sync_buffer_;
  }

  Bool* pred() const {
    return pred_;
  }

  static std::string getPredicateFlagName(const TensorView* val);
  static std::string getPredicateFlagName(const fuser::cuda::TensorView* val);

 private:
  ReductionOp* reduction_op_ = nullptr;
  Allocate* reduction_buffer_ = nullptr;
  Allocate* sync_buffer_ = nullptr;
  Bool* pred_ = nullptr;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
