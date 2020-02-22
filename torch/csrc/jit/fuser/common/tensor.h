#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>

/*
 * TODO: improve implementation bool IterDomain::same_as(const IterDomain*) const 
 * TODO: Add testing of same_as functions for these nodes
 * 
 * This file currently contains items associated with tensors, tensor domains, tensor views
 * and transforms associated with them (split, merge, reorder, compute_at).
 * 
 * Tensor is our link to the tensors described and used in the JIT. We create our own wrapper
 * version as a stepping stone into our IR structure, this allows us to link our concept of
 * tensors with that of the JIT.
 * 
 * IterDomain for now is an annotated size. The size is a range for us to iterate over (number of 
 * elements, not including stride). The annotations are associated with if there's a parallelization
 * mechanism associated with the iter domain, and if we need to reduce over it.
 * 
 * TensorDomain holds a vector (could be changed to an array) of IterDomains. It holds an IterDomain
 * for every logical axis in its associated tensor. TensorDomain does not directly hold the Tensor it
 * is associated. TensorDomain's primary responsibility is to hold the history of transformations 
 * that were used to generate it. This is done through the normal interaction of Expr/Val in Fusion.
 * i.e. if we want to know the previous operation generating a particular TensorDomain we can simply
 * call FusionGuard::getCurFusion()->origin(a_tensor_domain) which should give us an operation in the
 * list [split, merge, reorder] or similar operations that take in a TensorDomain, applies a
 * transformation and outputs a tensor domain.
 * 
 * TensorView is the glue between TensorDomain and Tensor. TensorView is intended to be used directly
 * in mathematical operations. TensorView is directly used in the "what" is being computed. TensorView
 * holds a reference to the Tensor it's a view of, as well as the TensorDomain of that particular view.
 * TensorView provides the history of the what is being computed and that history can be accessed,
 * similar to the mechanism TensorDomain uses, through normal Expr/Val interactions in Fusion. i.e.
 * FusionGuard::getCurFusion()->origin(a_tensor_view) which should give us an operation that takes in 
 * a TensorView, other inputs (other TensorViews, or Scalars) applies a mathematical operation and
 * outputs a TensorView (and other outputs?).
 * 
 * The reason we need TensorView and TensorDomain is that we need to have a record of both what is being
 * computed and how it is being computed. For Example we may have the operation:
 * TV3[I, J, K] = TV2[I, J, K] + TV1[I, J, K]
 * The mathematical operationss here are on the tensor views TV1, TV2, and TV3. This operation is a 
 * pointwise operation. To compute this pointwise operation we iterate over the 3D TensorDomain [I, J, K],
 * where K is the fastest changing dimension.
 * 
 * For now the functions split, merge, reorder, and compute_at are also in this file and its associated .cpp
 * file. However, they may be moved later.
 * 
 */ 

namespace torch {
namespace jit {
namespace fuser {

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
  ParallelType parallel_method_;
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

struct TensorView;
TORCH_API TensorView* ComputeAt_impl(TensorView* consumer, TensorView* producer, int axis);

struct TORCH_API TensorView : public Val {
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView(Tensor* _tensor)
      : Val(ValType::TensorView)
      , tensor_(_tensor)
      , compute_at_view_(nullptr)
      , compute_at_axis_(-1) {
        copyDomain(_tensor->domain());
      }

  TensorView(Tensor* _tensor, TensorDomain* _domain)
      : Val(ValType::TensorView)
      , tensor_(_tensor)
      , compute_at_view_(nullptr)
      , compute_at_axis_(-1) {
        copyDomain(_domain);
      }

  Tensor* tensor() const noexcept { return tensor_; }
  TensorDomain* domain() const noexcept { return domain_; }

  bool same_as(const TensorView* const other) const{
    return(
         tensor()->same_as(other->tensor())
      && domain()->same_as(other->domain())
    );
  }

  const TensorView* getComputeAtView() const noexcept { return compute_at_view_; }

  int getComputeAtAxis() const noexcept { return compute_at_axis_; }

friend TensorView* split(TensorView*, int axis, int factor);
friend TensorView* reorder(TensorView*, std::unordered_map<int, int>);
friend TensorView* merge(TensorView*, int axis);
friend TensorView* ComputeAt(const TensorView* consumer, TensorView* producer, int axis);

protected:
  void setDomain(TensorDomain* td){domain_ = td;}
  
private:
  Tensor* const tensor_;
  TensorDomain* domain_;
  TensorView* compute_at_view_;
  int compute_at_axis_;

  void copyDomain(const TensorDomain* td){
    std::vector<IterDomain*> idv;
    for(decltype(td->size()) i = 0; i <td->size(); i++)
      idv.push_back(td->axis(i));
    setDomain(new TensorDomain(idv));
  }

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

TORCH_API TensorView* split(TensorView*, int axis, int factor);
TORCH_API TensorView* merge(TensorView*, int axis);
TORCH_API TensorView* reorder(TensorView*, std::unordered_map<int, int>);
TORCH_API TensorView* computeAt(TensorView* consumer, TensorView* producer, int axis);


} // namespace fuser
} // namespace jit
} // namespace torch
