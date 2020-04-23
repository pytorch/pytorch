#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>

/*
 * Nodes in here should generally not be used by users. They should be behind
 * the scenes and users shouldn't have to be aware of what they do to use the
 * code generator.
 */

namespace torch {
namespace jit {
namespace fuser {

/*
 * TODO: improve implementation bool IterDomain::sameAs(const IterDomain*) const
 * TODO: Add testing of sameAs functions for these nodes
 */

/*
 * A specialization for Unary operations. Unary operations take in a single
 * input and produce a single output. Examples include:
 *   1) Casting operation i.e. float(a_val)
 *   2) Negation i.e. val * -1
 *   3) Reduction across a dimension i.e. val.sum(axis=2)
 *   4) split/merge/reorder
 */
struct TORCH_CUDA_API UnaryOp : public Expr {
  ~UnaryOp() = default;
  UnaryOp(UnaryOpType _type, Val* _out, Val* _in);

  UnaryOp(const UnaryOp& other) = delete;
  UnaryOp& operator=(const UnaryOp& other) = delete;

  UnaryOp(UnaryOp&& other) = delete;
  UnaryOp& operator=(UnaryOp&& other) = delete;

  Val* out() const noexcept {
    return out_;
  }
  Val* in() const noexcept {
    return in_;
  }

  UnaryOpType getUnaryOpType() const noexcept {
    return unary_op_type_;
  }

  bool sameAs(const UnaryOp* const other) const;

 private:
  const UnaryOpType unary_op_type_;
  Val* const out_;
  Val* const in_;
};

/*
 * A specialization for Binary operations. Binary operations take in two inputs
 * and produce a single output. Examples include:
 *  1) Add/mul/div/mod/sub (A * B)
 *  2) LT (A < B)
 */
struct TORCH_CUDA_API BinaryOp : public Expr {
  ~BinaryOp() = default;
  BinaryOp(BinaryOpType _type, Val* _out, Val* _lhs, Val* _rhs);

  BinaryOp(const BinaryOp& other) = delete;
  BinaryOp& operator=(const BinaryOp& other) = delete;

  BinaryOp(BinaryOp&& other) = delete;
  BinaryOp& operator=(BinaryOp&& other) = delete;

  Val* out() const noexcept {
    return out_;
  }
  Val* lhs() const noexcept {
    return lhs_;
  }
  Val* rhs() const noexcept {
    return rhs_;
  }

  BinaryOpType getBinaryOpType() const noexcept {
    return binary_op_type_;
  }

  bool sameAs(const BinaryOp* other) const;

 private:
  const BinaryOpType binary_op_type_;
  Val* const out_;
  Val* const lhs_;
  Val* const rhs_;
};

/*
 * Simply a representation of an annotated 1D iterable from start to extent.
 * TensorDomains which represent how to iterate over a tensor is made up of
 * IterDomains to form an ND iterable. We directly set parallization strategies
 * on IterDomains.
 */
struct TORCH_CUDA_API IterDomain : public Val {
  ~IterDomain() = default;

  IterDomain() = delete;

  IterDomain(
      Val* _start,
      Val* _extent,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false);

  bool sameAs(const IterDomain* const other) const;

  bool isReduction() const noexcept {
    return is_reduction_domain_;
  }

  bool isParallelized() const {
    return parallel_method_ != ParallelType::Serial;
  }

  // Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return (
        parallel_method_ == ParallelType::BIDz ||
        parallel_method_ == ParallelType::BIDy ||
        parallel_method_ == ParallelType::BIDx);
  }

  // Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return (
        parallel_method_ == ParallelType::TIDz ||
        parallel_method_ == ParallelType::TIDy ||
        parallel_method_ == ParallelType::TIDx);
  }

  // Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  void parallelize(ParallelType t) {
    parallel_method_ = t;
    if (isBlockDim()) {
      TORCH_CHECK(
          !isReduction(),
          "Cannot parallelize reductions across a block dimension.");
      if (isThreadDim())
        TORCH_CHECK(
            !isReduction(),
            "Thread parallelized reductions not yet supported.");
      TORCH_CHECK(
          t != ParallelType::Vectorize, "Vectorization not yet supported.");
      if (t == ParallelType::Unroll)
        TORCH_CHECK(
            start()->isZeroInt() && extent()->isConstScalar(),
            "Unrolling only supported with start = 0 and extent as a const int, but got ",
            "a start of ",
            start(),
            " and extent ",
            extent(),
            " .");
    }
  }

  ParallelType parallel_method() const noexcept {
    return parallel_method_;
  }

  Val* start() const noexcept {
    return start_;
  }
  Val* extent() const;

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

 private:
  Val* const start_;
  Val* const extent_;
  ParallelType parallel_method_ = ParallelType::Serial;
  bool is_reduction_domain_;
};
/*
 * TensorDomain holds a vector of IterDomains. It holds an IterDomain for every
 * logical axis in its associated tensor. TensorDomain does not directly hold
 * the Tensor it is associated with, and in theory could be associated with
 * multiple tensors. TensorDomain's primary responsibility is to provide a
 * mechanism to access history of transformations that were used to generate it.
 * This is done through the normal interaction of Expr/Val in Fusion. i.e. if we
 * want to know the previous operation generating a particular TensorDomain we
 * can simply call FusionGuard::getCurFusion()->origin(a_tensor_domain) which
 * should give us an operation in the list [split, merge, reorder] or similar
 * operations that take in a TensorDomain, applies a transformation and outputs
 * a tensor domain.
 */
struct TORCH_CUDA_API TensorDomain : public Val {
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  TensorDomain(std::vector<IterDomain*> _domain)
      : Val(ValType::TensorDomain), domain_(_domain) {}

  std::vector<IterDomain*>::size_type nDims() const {
    return domain_.size();
  }

  bool sameAs(const TensorDomain* const other) const;

  const std::vector<IterDomain*>& domain() const noexcept {
    return domain_;
  }

  TensorDomain* noReductions() const;

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  IterDomain* axis(int i) const;

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor
  TensorDomain* split(int axis, int factor);

  // Merge "axis" and "axis+1" into 1 dimension
  TensorDomain* merge(int axis);

  // Reorder axes according to map[old_pos] = new_pos
  TensorDomain* reorder(const std::unordered_map<int, int>& axis2pos);

  TensorDomain* rootDomain();

 private:
  std::vector<IterDomain*> domain_;
};

/*
 * Representation for a split on IterDomain = axis in a TensorDomain, by factor
 * = factor
 * TODO: Implement split by nparts
 */
struct TORCH_CUDA_API Split : public Expr {
  ~Split() = default;

  Split(const Split& other) = delete;
  Split& operator=(const Split& other) = delete;

  Split(Split&& other) = delete;
  Split& operator=(Split&& other) = delete;

  Split(TensorDomain* _out, TensorDomain* _in, int _axis, Int* _factor);

  TensorDomain* out() const noexcept {
    return out_;
  }
  TensorDomain* in() const noexcept {
    return in_;
  }

  int axis() const noexcept {
    return axis_;
  }
  Int* factor() const noexcept {
    return factor_;
  }
  bool sameAs(const Split* const other) const;

 private:
  TensorDomain* const out_;
  TensorDomain* const in_;
  const int axis_;
  Int* const factor_;
};

/*
 * Merge Iterdomain _axis in TensorDomain with the following IterDomain. Both
 * IterDomains must be of the same iter or reduction type, as well as the same
 * parallelization strategy if there is one.
 * TODO: Should this be a unary op type?
 */
struct TORCH_CUDA_API Merge : public Expr {
  ~Merge() = default;
  Merge(TensorDomain* _out, TensorDomain* _in, int _axis);

  Merge(const Merge& other) = delete;
  Merge& operator=(const Merge& other) = delete;

  Merge(Merge&& other) = delete;
  Merge& operator=(Merge&& other) = delete;

  TensorDomain* out() const noexcept {
    return out_;
  }
  TensorDomain* in() const noexcept {
    return in_;
  }
  int axis() const noexcept {
    return axis_;
  }

  bool sameAs(const Merge* const other) const;

 private:
  TensorDomain* const out_;
  TensorDomain* const in_;
  int axis_;
};

/*
 * Reorder the IterDomains of a tensor domain with the map
 * pos2axis[new_position] = old_position
 */
struct TORCH_CUDA_API Reorder : public Expr {
  ~Reorder() = default;
  Reorder(TensorDomain* _out, TensorDomain* _in, std::vector<int> _pos2axis);

  Reorder(const Reorder& other) = delete;
  Reorder& operator=(const Reorder& other) = delete;

  Reorder(Reorder&& other) = delete;
  Reorder& operator=(Reorder&& other) = delete;

  TensorDomain* out() const noexcept {
    return out_;
  }
  TensorDomain* in() const noexcept {
    return in_;
  }
  const std::vector<int>& pos2axis() const noexcept {
    return pos2axis_;
  }

  bool sameAs(const Reorder* const other) const;

 private:
  TensorDomain* const out_;
  TensorDomain* const in_;
  const std::vector<int> pos2axis_;
};

/*
 * ForLoop provides scoping around an int iterator from 0 to range. Exprs placed
 * in its body are considered inside the scope of the for loop. In the future
 * the implementation should look quite different so that we can do proper
 * dependency annalysis like in Fusion.
 *
 * TODO: Change implmentation of Exprs contained in the scope to be more similar
 * to Fusion where we can do proper dependency analysis.
 */
struct TORCH_CUDA_API ForLoop : public Expr {
  ~ForLoop() = default;
  ForLoop(
      Val* _index,
      IterDomain* _iter_domain,
      const std::vector<Expr*>& _body = {},
      Expr* parent_scope = nullptr);

  ForLoop(const ForLoop& other) = delete;
  ForLoop& operator=(const ForLoop& other) = delete;

  ForLoop(ForLoop&& other) = delete;
  ForLoop& operator=(ForLoop&& other) = delete;

  Val* index() const noexcept {
    return index_;
  }

  IterDomain* iter_domain() const noexcept {
    return iter_domain_;
  }

  Scope& body() noexcept {
    return body_;
  }

  const Scope& constBody() const noexcept {
    return body_;
  }

  bool sameAs(const ForLoop* other) const;
  Expr* parentScope() const noexcept {
    return parent_scope_;
  }
  bool hasParentScope() const noexcept {
    return parent_scope_ == nullptr;
  }

 private:
  Val* const index_;
  IterDomain* const iter_domain_;
  Scope body_;
  Expr* parent_scope_;
};

/*
 * IfThenElse provides scoping for an boolean operator. Exprs placed in its body
 * are considered inside the scope of the if statement. In the future the
 * implementation should look quite different so that we can do proper
 * dependency annalysis like in Fusion.
 *
 * TODO: Change implmentation of Exprs contained in the scope to be more similar
 * to Fusion where we can do proper dependency analysis.
 */
struct TORCH_CUDA_API IfThenElse : public Expr {
  ~IfThenElse() = default;
  IfThenElse(
      Int* _cond,
      const std::vector<Expr*>& _if_body = {},
      const std::vector<Expr*>& _else_body = {},
      Expr* _parent_scope = nullptr);

  IfThenElse(const IfThenElse& other) = delete;
  IfThenElse& operator=(const IfThenElse& other) = delete;

  IfThenElse(IfThenElse&& other) = delete;
  IfThenElse& operator=(IfThenElse&& other) = delete;

  Int* cond() const noexcept {
    return cond_;
  }

  const Scope& constBody() const noexcept {
    return body_;
  }

  const Scope& constElseBody() const noexcept {
    return else_body_;
  }

  Scope& body() noexcept {
    return body_;
  }

  Scope& elseBody() noexcept {
    return else_body_;
  }

  bool hasElse() const noexcept {
    return !else_body_.empty();
  }

  bool sameAs(const IfThenElse* other) const;

  Expr* parentScope() const noexcept {
    return parent_scope_;
  }

 private:
  Int* const cond_;
  Scope body_;
  Scope else_body_;
  Expr* parent_scope_;
};

/*
 * TODO: Fill out TensorIndex, which is a list of Ints used to directly index a
 * TensorView. It is not the flattened index, which needs to be computed using
 * stride information.
 */
struct TORCH_CUDA_API TensorIndex : public Val {
  ~TensorIndex() = default;

  TensorIndex(const TensorIndex& other) = delete;
  TensorIndex& operator=(const TensorIndex& other) = delete;

  TensorIndex(TensorIndex&& other) = delete;
  TensorIndex& operator=(TensorIndex&& other) = delete;

  TensorIndex(const TensorView* const _view, std::vector<Val*> _indices)
      : Val(ValType::TensorIndex), view_(_view), indices_(_indices) {
    TORCH_INTERNAL_ASSERT(
        std::all_of(
            _indices.begin(),
            _indices.end(),
            [](Val* v) {
              return (v->getValType() == ValType::Scalar ||
                      v->getValType() == ValType::NamedScalar) &&
                  v->getDataType() == DataType::Int;
            }),
        "Cannot index with a value other than an int.");
  }

  std::vector<Val*>::size_type nDims() const {
    return indices_.size();
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  Val* index(int i) const;

  const std::vector<Val*>& indices() const noexcept {
    return indices_;
  }

  const TensorView* view() const noexcept {
    return view_;
  }

  bool sameAs(const TensorIndex* const other) const;

 private:
  const TensorView* view_;
  std::vector<Val*> indices_;
};

/*
 * Allocate is a lower level Node that describes a buffer of memory that
 * is required as an intermediate within a kernel.  The extent is the expression
 * of the size of the buffer that is generated from the TensorView that
 * describes the output of an operation.
 *
 * TODO: The components of Allocate like Type and Name could be separated from
 * the the assocated TensorView.  Perhaps that is more appropriate?
 */
struct TORCH_CUDA_API Allocate : public Expr {
  ~Allocate() = default;

  Allocate(const Allocate& other) = delete;
  Allocate& operator=(const Allocate& other) = delete;

  Allocate(Allocate&& other) = delete;
  Allocate& operator=(Allocate&& other) = delete;

  Allocate(TensorView* _tv, Val* size);

  DataType buf_type() const;
  Val* extent() const noexcept {
    return extent_;
  }
  TensorView* buffer() const noexcept {
    return buffer_;
  }

  bool sameAs(const Allocate* other) const;

 private:
  TensorView* buffer_;
  Val* extent_;
};

/*
 * Integer value which has a special name. These could be:
 * - threadIdx.x
 * - blockIdx.y
 * - blockDim.z
 * - T3.stride[2]
 */
struct TORCH_CUDA_API NamedScalar : public Val {
  ~NamedScalar() = default;
  NamedScalar() = delete;

  NamedScalar(std::string _name, DataType dtype)
      : Val(ValType::NamedScalar, dtype), name_(_name) {}

  NamedScalar(const NamedScalar& other) = delete;
  NamedScalar& operator=(const NamedScalar& other) = delete;

  NamedScalar(NamedScalar&& other) = delete;
  NamedScalar& operator=(NamedScalar&& other) = delete;

  const std::string& name() const noexcept {
    return name_;
  }

  bool sameAs(const NamedScalar* const other) const {
    return other->name().compare(name()) == 0;
  }

 private:
  std::string name_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
