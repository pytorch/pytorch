#pragma once


#include <torch/csrc/jit/codegen/cuda/ir_base.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API Float : public Val {
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

  virtual bool same_as(const Float* const other) const {
    if (isConst() && other->isConst())
      return *value() == *(other->value());
    return this == other;
  }

 private:
  const c10::optional<float> maybe_value_;
};

struct TORCH_API Int : public Val {
  ~Int() = default;

  Int() : Val(ValType::Scalar, DataType::Int), maybe_value_{c10::nullopt} {}

  Int(int _value) : Val(ValType::Scalar, DataType::Int), maybe_value_{_value} {}

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
  c10::optional<int> value() const noexcept {
    return maybe_value_;
  }

  virtual bool same_as(const Int* const other) const {
    if (isConst() && other->isConst())
      return *value() == *(other->value());
    return this == other;
  }

 private:
  const c10::optional<int> maybe_value_;
};


// TODO: comment
struct TORCH_API UnaryOp : public Expr {
  ~UnaryOp() = default;
  UnaryOp(UnaryOpType _type, Val* _out, Val* _in);

  UnaryOp(const UnaryOp& other) = delete;
  UnaryOp& operator=(const UnaryOp& other) = delete;

  UnaryOp(UnaryOp&& other) = delete;
  UnaryOp& operator=(UnaryOp&& other) = delete;

  Val* out() const noexcept { return out_; }
  Val* in() const noexcept { return in_; }

  UnaryOpType type() const noexcept { return unary_op_type_; }

  bool same_as(const UnaryOp* const other) const {
    if (this->type() != other->type())
      return false;
    return static_cast<const Expr*>(this)->same_as(other);
  }

 private:
  const UnaryOpType unary_op_type_;
  Val* const out_;
  Val* const in_;
};

// TODO: comment
struct TORCH_API BinaryOp : public Expr {
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

  BinaryOpType type() const noexcept {
    return binary_op_type_;
  }

  bool same_as(const BinaryOp* other) const {
    if (type() != other->type())
      return false;
    return static_cast<const Expr*>(this)->same_as(other);
  }

 private:
  const BinaryOpType binary_op_type_;
  Val* const out_;
  Val* const lhs_;
  Val* const rhs_;
};

// TODO: Not sure if array of expressions is appropriate composition
// TODO: I named it ForLoop instead of For because it is easer to search for.
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

  void remove_expr(const Expr* e) {
    auto it = body_.begin();
    for (; it != body_.end(); ++it)
      if (*it == e)
        break;
    if (it != body_.end())
      body_.erase(it);
  }

  // TODO: This should probably be more sophisiticated.
  bool same_as(const ForLoop* other) const {
    return static_cast<const Expr*>(this)->same_as(other);
  }

 private:
  // TODO: Why is the pointer const and not what's in the object?
  Int* const index_;
  IterDomain* const range_;
  std::vector<const Expr*> body_;
};

// TODO: Not sure if array of expressions is appropriate composition
// TODO: I named it IfThenElse instead of For because it is easer to search for.
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

  // TODO: This should probably be more sophisiticated.
  bool same_as(const IfThenElse* other) const {
    return static_cast<const Expr*>(this)->same_as(other);
  }

 private:
  // TODO: Why is the pointer const and not what's in the object?
  Val* const cond_;
  std::vector<const Expr*> if_body_;
  std::vector<const Expr*> else_body_;
};


struct TORCH_API TensorIndex : public Val {
  TensorIndex(Int* _size)
      : Val(ValType::TensorIndex, DataType::Int),
        size_(_size) { }

  bool same_as(const TensorIndex* const other) const {
    return size()->same_as(other->size());
  }

  Int* size() const noexcept {
    return size_;
  }

  TensorIndex() = delete;
  ~TensorIndex() = default;

  TensorIndex(const TensorIndex& other) = delete;
  TensorIndex& operator=(const TensorIndex& other) = delete;

  TensorIndex(TensorIndex&& other) = delete;
  TensorIndex& operator=(TensorIndex&& other) = delete;

 private:
  Int* const size_;
};

struct TORCH_API IterDomain : public Val {
  ~IterDomain() = default;

  IterDomain() = delete;

  IterDomain(
      Int* _size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false)
      : Val(ValType::IterDomain, DataType::Int),
        size_(_size),
        parallel_method_(_parallel_method),
        is_reduction_domain_(_reduction_domain) {}

  IterDomain(
      Val* int_size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false)
      : Val(ValType::IterDomain, DataType::Int),
        size_(static_cast<Int*>(int_size)),
        parallel_method_(_parallel_method),
        is_reduction_domain_(_reduction_domain) {
    assert(int_size->isVal());
    assert(int_size->getDataType() == DataType::Int);
  }

  bool same_as(const IterDomain* const other) const {
    return(
         isReduction() == other->isReduction()
      && parallel_method() == other->parallel_method()
      && size()->same_as(other->size())
    );
  }

  bool isReduction() const noexcept {
    return is_reduction_domain_;
  }
  
  bool isParallelized(){ return parallel_method_ != ParallelType::Serial;}

  bool isBlockDim(){
    if( parallel_method_ == ParallelType::BIDz
      ||parallel_method_ == ParallelType::BIDy
      ||parallel_method_ == ParallelType::BIDx)
      return true;
    return false;
  }

  bool isThreadDim(){
    if( parallel_method_ == ParallelType::TIDz
      ||parallel_method_ == ParallelType::TIDy
      ||parallel_method_ == ParallelType::TIDx)
      return true;
    return false;
  }

  bool isThread(){
    return ( isBlockDim() || isThreadDim() );
  }

  void parallelize(ParallelType t){parallel_method_ = t;}

  ParallelType parallel_method() const noexcept {
    return parallel_method_;
  }
  Int* size() const noexcept {
    return size_;
  }

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

 private:
  Int* const size_;
  ParallelType parallel_method_ = ParallelType::Serial;
  bool is_reduction_domain_;
};

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

  bool same_as(const TensorDomain* const other) const {
    if(size() != other->size())
      return false;

    for(decltype(size()) i = 0; i<size(); i++)
      if( !(axis(i)->same_as(other->axis(i))) )
        return false;

    return true;
      
  }

  TensorDomain* noReductions() const {
    std::vector<IterDomain*> noReductionDomain;
    for(IterDomain* id : domain)
      if(!id->isReduction())
        noReductionDomain.push_back(id);
    return new TensorDomain(noReductionDomain);
  }

  //i here is int, as we want to accept negative value and ::size_type can be a uint.
  IterDomain* axis(int i) const {
    if(i < 0)
      i+=size();
    assert(i >= 0 && i < size());
    return domain[i];
  }


 private:
  std::vector<IterDomain*> domain;
};

//Going to friend TransformReplay so it can reset TensorView domain before replaying.
//We could narrow friend down to a single function but it would require including the entire header.
struct TransformReplay;

struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor() = delete;

  Tensor(DataType dt, TensorDomain* _td = nullptr)
      : Val(ValType::Tensor, dt), contiguity_(c10::nullopt), domain_(_td) {}

  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;

  Tensor(Tensor&& other) = delete;
  Tensor& operator=(Tensor&& other) = delete;

  Tensor(const std::shared_ptr<c10::TensorType>& tensor_type);

  Tensor(const std::shared_ptr<Value>& jit_value);
  

  //TODO: implement   bool same_as(const Tensor* other) const
  bool hasContiguityInfo() const;

  const c10::optional<TensorContiguity>& getContiguityInfo() const;

  static Tensor* MakeDummyTensor(int ndims) {
    std::vector<IterDomain*> sizes;
    for (int i = 0; i < ndims; i++) {
      sizes.push_back(new IterDomain(new Int()));
    }
    TensorDomain* td = new TensorDomain(sizes);

    return new Tensor(DataType::Float, td);
  }

  TensorDomain* domain() const noexcept { return domain_; }

  private:

  // Implementation details:
  const c10::optional<TensorContiguity> contiguity_;
  TensorDomain* domain_;
};

/*
 * Split an axis, by factor factor
 * TODO: Implement split by nparts
 */
struct TORCH_API Split : public Expr {
  ~Split() = default;
  Split(
      TensorDomain* _out,
      TensorDomain* _in,
      int _axis,
      Int* _factor);

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

  bool same_as(const Split* const other) const{
    return(
         out()->same_as(other->out())
      && in()->same_as(other->in())
      && axis() == other->axis()
      && factor()->same_as(other->factor())
    );
  }

  Split(const Split& other) = delete;
  Split& operator=(const Split& other) = delete;

  Split(Split&& other) = delete;
  Split& operator=(Split&& other) = delete;

 private:
  TensorDomain* const out_;
  TensorDomain* const in_;
  const int axis_;
  Int* const factor_;
};

/*
 * Merge axis _axis with the following axis. Both axis must be of the same
 * iter or reduction axis, as well as the same parallelization strategy if
 * there is one.
 * TODO: Should this be a unary op type?
 */
struct TORCH_API Merge : public Expr {
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

  bool same_as(const Merge* const other) const{
    return(
         out()->same_as(other->out())
      && in()->same_as(other->in())
      && axis() == other->axis()
    );
  }

 private:
  TensorDomain* const out_;
  TensorDomain* const in_;
  int axis_;
};

/*
 * Reorder axis of a tensor domain with the map
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

  TensorDomain* out() const noexcept {
    return out_;
  }
  TensorDomain* in() const noexcept {
    return in_;
  }
  //Returns map pos2axis[new_position] = old_position
  const std::vector<int>& pos2axis() const noexcept {
    return pos2axis_;
  }

  bool same_as(const Merge* const other) const{
    //Implicitly in and out matching means pos2axis matches
    return(
         out()->same_as(other->out())
      && in()->same_as(other->in())
    );
  }

 private:
  TensorDomain* const out_;
  TensorDomain* const in_;
  const std::vector<int> pos2axis_;
};



}}}

