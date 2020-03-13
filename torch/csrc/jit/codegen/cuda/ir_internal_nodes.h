#pragma once


#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
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
struct TORCH_API UnaryOp : public Expr {
  ~UnaryOp() = default;
  UnaryOp(UnaryOpType _type, Val* _out, Val* _in);

  UnaryOp(const UnaryOp& other) = delete;
  UnaryOp& operator=(const UnaryOp& other) = delete;

  UnaryOp(UnaryOp&& other) = delete;
  UnaryOp& operator=(UnaryOp&& other) = delete;

  Val* out() const noexcept { return out_; }
  Val* in() const noexcept { return in_; }

  UnaryOpType getUnaryOpType() const noexcept { return unary_op_type_; }

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
struct TORCH_API BinaryOp : public Expr {
  ~BinaryOp() = default;
  BinaryOp(BinaryOpType _type, Val* _out, Val* _lhs, Val* _rhs);

  BinaryOp(const BinaryOp& other) = delete;
  BinaryOp& operator=(const BinaryOp& other) = delete;

  BinaryOp(BinaryOp&& other) = delete;
  BinaryOp& operator=(BinaryOp&& other) = delete;

  Val* out() const noexcept { return out_; }
  Val* lhs() const noexcept { return lhs_; }
  Val* rhs() const noexcept { return rhs_; }

  BinaryOpType getBinaryOpType() const noexcept { return binary_op_type_; }

  bool sameAs(const BinaryOp* other) const;

 private:
  const BinaryOpType binary_op_type_;
  Val* const out_;
  Val* const lhs_;
  Val* const rhs_;
};

/*
 * Simply a representation of an iterable from 0 to size. TensorDomains which
 * represent how to iterate over a tensor is made up of IterDomains. We directly
 * set parallization strategies on IterDomains.
 */
struct TORCH_API IterDomain : public Val {
  ~IterDomain() = default;

  IterDomain() = delete;

  IterDomain(
      Int* _size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false);

  IterDomain(
      Val* int_size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false);

  bool sameAs(const IterDomain* const other) const;

  bool isReduction() const noexcept { return is_reduction_domain_; }
  
  bool isParallelized() const { return parallel_method_ != ParallelType::Serial;}

  bool isBlockDim() const {
    return ( parallel_method_ == ParallelType::BIDz
          || parallel_method_ == ParallelType::BIDy
          ||parallel_method_ == ParallelType::BIDx);
  }

  bool isThreadDim() const {
    return ( parallel_method_ == ParallelType::TIDz
          || parallel_method_ == ParallelType::TIDy
          || parallel_method_ == ParallelType::TIDx);
  }

  bool isThread() const {
    return ( isBlockDim() || isThreadDim() );
  }

  void parallelize(ParallelType t){parallel_method_ = t;}

  ParallelType parallel_method() const noexcept {
    return parallel_method_;
  }

  Int* size() const noexcept { return size_; }

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

 private:
  Int* const size_;
  ParallelType parallel_method_ = ParallelType::Serial;
  bool is_reduction_domain_;
};

// A list of IterDomains representing how to iterate across a given Tensor.
struct TORCH_API TensorDomain : public Val {
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  TensorDomain(std::vector<IterDomain*> domain_)
      : Val(ValType::TensorDomain), domain(domain_) {}

  std::vector<IterDomain*>::size_type size() const {
    return domain.size();
  }

  bool sameAs(const TensorDomain* const other) const;

  TensorDomain* noReductions() const;

  //i here is int, as we want to accept negative value and ::size_type can be a uint.
  IterDomain* axis(int i) const;

 private:
  std::vector<IterDomain*> domain;
  
};

/*
 * Representation for a split on IterDomain = axis in a TensorDomain, by factor
 * = factor
 * TODO: Implement split by nparts
 */
struct TORCH_API Split : public Expr {
  ~Split() = default;

  Split(const Split& other) = delete;
  Split& operator=(const Split& other) = delete;

  Split(Split&& other) = delete;
  Split& operator=(Split&& other) = delete;

  Split(
      TensorDomain* _out,
      TensorDomain* _in,
      int _axis,
      Int* _factor);

  TensorDomain* out() const noexcept { return out_; }
  TensorDomain* in() const noexcept { return in_; }

  int axis() const noexcept { return axis_; }
  Int* factor() const noexcept { return factor_; }
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
struct TORCH_API Merge : public Expr {
  ~Merge() = default;
  Merge(TensorDomain* _out, TensorDomain* _in, int _axis);

  Merge(const Merge& other) = delete;
  Merge& operator=(const Merge& other) = delete;

  Merge(Merge&& other) = delete;
  Merge& operator=(Merge&& other) = delete;

  TensorDomain* out() const noexcept { return out_; }
  TensorDomain* in() const noexcept { return in_; }
  int axis() const noexcept { return axis_; }

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
struct TORCH_API Reorder : public Expr {
  ~Reorder() = default;
  Reorder(
      TensorDomain* _out,
      TensorDomain* _in,
      std::vector<int> _pos2axis);

  Reorder(const Reorder& other) = delete;
  Reorder& operator=(const Reorder& other) = delete;

  Reorder(Reorder&& other) = delete;
  Reorder& operator=(Reorder&& other) = delete;

  TensorDomain* out() const noexcept { return out_; }
  TensorDomain* in() const noexcept { return in_; }
  const std::vector<int>& pos2axis() const noexcept { return pos2axis_; }

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
struct TORCH_API ForLoop : public Expr {
  ~ForLoop() = default;
  ForLoop(
      Int* _index,
      IterDomain* _range,
      const std::vector<const Expr*>& _body);

  ForLoop(const ForLoop& other) = delete;
  ForLoop& operator=(const ForLoop& other) = delete;

  ForLoop(ForLoop&& other) = delete;
  ForLoop& operator=(ForLoop&& other) = delete;

  Int* index() const noexcept {
    return index_;
  }
  IterDomain* range() const noexcept {
    return range_;
  }

  const std::vector<const Expr*>& body() const noexcept {
    return body_;
  }

  void add_expr(const Expr* e) {
    body_.push_back(e);
  }

  void remove_expr(const Expr* e);
  bool sameAs(const ForLoop* other) const;

 private:
  Int* const index_;
  IterDomain* const range_;
  std::vector<const Expr*> body_;
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
struct TORCH_API IfThenElse : public Expr {
  ~IfThenElse() = default;
  IfThenElse(
      Val* _cond,
      const std::vector<const Expr*>& _if_body,
      const std::vector<const Expr*>& _else_body = {});

  IfThenElse(const IfThenElse& other) = delete;
  IfThenElse& operator=(const IfThenElse& other) = delete;

  IfThenElse(IfThenElse&& other) = delete;
  IfThenElse& operator=(IfThenElse&& other) = delete;

  Val* cond() const noexcept {
    return cond_;
  }
  const std::vector<const Expr*>& if_body() const noexcept {
    return if_body_;
  }
  const std::vector<const Expr*>& else_body() const noexcept {
    return else_body_;
  }

  void add_if_expr(const Expr* e) {
    if_body_.push_back(e);
  }
  void add_else_expr(const Expr* e) {
    else_body_.push_back(e);
  }

  bool hasElse() const noexcept {
    return !else_body_.empty();
  }

  bool sameAs(const IfThenElse* other) const;
  
 private:
  // TODO: Why is the pointer const and not what's in the object?
  Val* const cond_;
  std::vector<const Expr*> if_body_;
  std::vector<const Expr*> else_body_;
};


/*
 * TODO: Fill out TensorIndex, which is a list of Ints used to directly index a
 * TensorView. It is not the flattened index, which needs to be computed using
 * stride information.
 */
struct TORCH_API TensorIndex : public Val {
  ~TensorIndex() = default;

  TensorIndex(const TensorIndex& other) = delete;
  TensorIndex& operator=(const TensorIndex& other) = delete;

  TensorIndex(TensorIndex&& other) = delete;
  TensorIndex& operator=(TensorIndex&& other) = delete;

  TensorIndex(std::vector<Int*> _indices)
      : Val(ValType::TensorIndex), indices_(_indices) {}

  std::vector<Int*>::size_type size() const {
    return indices_.size();
  }

  bool sameAs(const TensorIndex* const other) const;
  //i here is int, as we want to accept negative value and ::size_type can be a uint.
  Int* axis(int i) const;

 private:
  std::vector<Int*> indices_;
};

}}}

