#pragma once

#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor_meta.h>

/*
 * TODO: improve implementation bool IterDomain::same_as(const IterDomain*) const 
 * TODO: Add testing of same_as functions for these nodes
 */ 

namespace torch {
namespace jit {
namespace fuser {

struct TORCH_API IterDomain : public Val {
  ~IterDomain() = default;

  IterDomain() = delete;

  IterDomain(
      const Int* _size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false)
      : Val(ValType::IterDomain, DataType::Int),
        size_(_size),
        parallel_method_(_parallel_method),
        is_reduction_domain_(_reduction_domain) {}

  IterDomain(
      const Val* int_size,
      ParallelType _parallel_method = ParallelType::Serial,
      bool _reduction_domain = false)
      : Val(ValType::IterDomain, DataType::Int),
        size_(static_cast<const Int*>(int_size)),
        parallel_method_(_parallel_method),
        is_reduction_domain_(_reduction_domain) {
    assert(int_size->isVal());
    assert(int_size->getDataType() == DataType::Int);
  }

  bool same_as(const IterDomain* other) const {
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
  const Int* size() const noexcept {
    return size_;
  }

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

 private:
  const Int* size_;
  const ParallelType parallel_method_;
  const bool is_reduction_domain_;
};

struct TORCH_API TensorDomain : public Val {
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  TensorDomain(std::vector<const IterDomain*> domain_)
      : Val(ValType::TensorDomain), domain(domain_) {}

  std::vector<const IterDomain*>::size_type size() const {
    return domain.size();
  }

  bool same_as(const TensorDomain* other) const {
    if(size() != other->size())
      return false;

    for(decltype(size()) i = 0; i<size(); i++)
      if( !(axis(i)->same_as(other->axis(i))) )
        return false;

    return true;
      
  }

  //i here is int, as we want to accept negative value and ::size_type can be a uint.
  const IterDomain* axis(int i) const {
    if(i < 0)
      i+=size();
    assert(i >= 0 && i < size());
    return domain[i];
  }


 private:
  const std::vector<const IterDomain*> domain;
};

struct TORCH_API Tensor : public Val {
  ~Tensor() = default;

  Tensor() = delete;

  Tensor(DataType dt, const TensorDomain* _td = nullptr)
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

  static const Tensor* MakeDummyTensor(int ndims) {
    std::vector<const IterDomain*> sizes;
    for (int i = 0; i < ndims; i++) {
      sizes.push_back(new IterDomain(new Int()));
    }
    TensorDomain* td = new TensorDomain(sizes);

    return new Tensor(DataType::Float, td);
  }

  const TensorDomain* domain() const noexcept { return domain_; }

  private:

  // Implementation details:
  const c10::optional<TensorContiguity> contiguity_;
  const TensorDomain* domain_;
};

//void ComputeAt_impl(const TensorView* consumer, const TensorView* producer, int axis){
  /*
   * TODO:
   * Recursive compute_at:
   * Recurse backward from consumer, to producer, make sure there's a dependency chain there.
   * After recursing, recurse again, and call ComputeAt for all tensors between producer and consumer.
   * 
   * Assert direct consumer/producer relationship.
   * Compute at modifies the consumer, not the producer.
   * 
   */
//}

struct TORCH_API TensorView : public Val {
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView(const Tensor* _tensor, const TensorDomain* _domain)
      : Val(ValType::TensorView)
      , tensor_(_tensor)
      , domain_(_domain)
      , compute_at_view_(nullptr)
      , compute_at_axis_(-1) {
      }

  const Tensor* tensor() const noexcept { return tensor_; }
  const TensorDomain* domain() const noexcept { return domain_; }

  bool same_as(const TensorView* other) const{
    return(
         tensor()->same_as(other->tensor())
      && domain()->same_as(other->domain())
    );
  }

  const TensorView* getComputeAtView() const noexcept { return compute_at_view_; }
  int getComputeAtAxis() const noexcept { return compute_at_axis_; }
  void computeAt(const TensorView* tv, int axis) {
    compute_at_view_ = tv;
    compute_at_axis_ = axis;
    //ComputeAt_impl(tv, this, axis);
  }

private:
  const Tensor* tensor_;
  const TensorDomain* domain_;
  const TensorView* compute_at_view_;
  int compute_at_axis_;

};

/*
 * Split an axis, by factor factor
 * TODO: Implement split by nparts
 */
struct TORCH_API Split : public Expr {
  ~Split() = default;
  Split(
      const TensorView* _out,
      const TensorView* _in,
      int _axis,
      const Int* _factor);

  const TensorView* out() const noexcept {
    return out_;
  }
  const TensorView* in() const noexcept {
    return in_;
  }
  int axis() const noexcept {
    return axis_;
  }
  const Int* factor() const noexcept {
    return factor_;
  }

  bool same_as(const Split* other) const{
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
  const TensorView* out_;
  const TensorView* in_;
  const int axis_;
  const Int* factor_;
};

/*
 * Merge axis _axis with the following axis. Both axis must be of the same
 * iter or reduction axis, as well as the same parallelization strategy if
 * there is one.
 * TODO: Should this be a unary op type?
 */
struct TORCH_API Merge : public Expr {
  ~Merge() = default;
  Merge(const TensorView* _out, const TensorView* _in, int _axis);

  Merge(const Merge& other) = delete;
  Merge& operator=(const Merge& other) = delete;

  Merge(Merge&& other) = delete;
  Merge& operator=(Merge&& other) = delete;

  const TensorView* out() const noexcept {
    return out_;
  }
  const TensorView* in() const noexcept {
    return in_;
  }
  int axis() const noexcept {
    return axis_;
  }

  bool same_as(const Merge* other) const{
    return(
         out()->same_as(other->out())
      && in()->same_as(other->in())
      && axis() == other->axis()
    );
  }

 private:
  const TensorView* out_;
  const TensorView* in_;
  const int axis_;
};

/*
 * Reorder axis of a tensor domain with the map
 * pos2axis[new_position] = old_position
 */
struct TORCH_API Reorder : public Expr {
  ~Reorder() = default;
  Reorder(
      const TensorView* _out,
      const TensorView* _in,
      std::vector<int> _pos2axis);

  Reorder(const Reorder& other) = delete;
  Reorder& operator=(const Reorder& other) = delete;

  Reorder(Reorder&& other) = delete;
  Reorder& operator=(Reorder&& other) = delete;

  const TensorView* out() const noexcept {
    return out_;
  }
  const TensorView* in() const noexcept {
    return in_;
  }
  const std::vector<int> pos2axis() const noexcept {
    return pos2axis_;
  }

  bool same_as(const Merge* other) const{
    //Implicitly in and out matching means pos2axis matches
    return(
         out()->same_as(other->out())
      && in()->same_as(other->in())
    );
  }

 private:
  const TensorView* out_;
  const TensorView* in_;
  const std::vector<int> pos2axis_;
};

TORCH_API const TensorView* split(const TensorView*, int axis, int factor);
TORCH_API const TensorView* split(const Tensor*, int axis, int factor);

TORCH_API const TensorView* merge(const TensorView*, int axis);
TORCH_API const TensorView* merge(const Tensor*, int axis);

TORCH_API const TensorView* reorder(const TensorView*, std::unordered_map<int, int>);
TORCH_API const TensorView* reorder(const Tensor*, std::unordered_map<int, int>);

} // namespace fuser
} // namespace jit
} // namespace torch
