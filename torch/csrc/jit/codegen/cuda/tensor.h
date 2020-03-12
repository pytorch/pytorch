#pragma once

#include <torch/csrc/jit/codegen/cuda/ir_nodes.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>

/* 
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


struct TransformReplay;

struct TORCH_API TensorView : public Val {
  ~TensorView() = default;

  TensorView(const TensorView& other) = delete;
  TensorView& operator=(const TensorView& other) = delete;

  TensorView(TensorView&& other) = delete;
  TensorView& operator=(TensorView&& other) = delete;

  TensorView(Tensor* _tensor, TensorDomain* _domain = nullptr)
      : Val(ValType::TensorView, _tensor->getDataType().value())
      , tensor_(_tensor)
      , domain_(_domain) {
        if(_domain == nullptr)
          copyDomain(_tensor->domain());
      }

  TensorView(TensorDomain* _domain, DataType dtype)
      : Val(ValType::TensorView, dtype)
      , tensor_(new Tensor(dtype, _domain))
      , domain_(_domain) {}

  //We'll copy the TensorView with some minor modifications
  //Reduction axes won't get copied over.
  TensorView* newForOutput(DataType dtype) const;
  TensorView* clone() const;

  Tensor* tensor() const noexcept { return tensor_; }
  TensorDomain* domain() const noexcept { return domain_; }

  bool sameAs(const TensorView* const other) const{
    bool same_tensor = tensor() == nullptr
                    || other->tensor() == nullptr 
                    ? tensor()==nullptr && other->tensor()==nullptr : tensor()->sameAs(other->tensor());
    return(
         same_tensor
      && domain()->sameAs(other->domain())
      && getDataType().value() == other->getDataType().value()
    );
  }

  bool hasComputeAt() const { return compute_at_view_ != nullptr; }

  const TensorView* getComputeAtView() const noexcept { return compute_at_view_; }
  
  IterDomain* getAxis(int axis){
    if(!hasComputeAt()
      || getComputeAtAxis() <= axis)
      return domain()->axis(axis);
    return compute_at_view_->getAxis(axis);
  }

  int getComputeAtAxis() const noexcept { return compute_at_axis_; }
  
  TensorView* computeAt(TensorView* consumer, int axis);

  void resetView();

  friend TensorView* split(TensorView*, int axis, int factor);
  friend TensorView* reorder(TensorView*, std::unordered_map<int, int>);
  friend TensorView* merge(TensorView*, int axis);
  friend TransformReplay;

protected:
  void setDomain(TensorDomain* td){domain_ = td;}
  
private:
  Tensor* const tensor_;
  TensorDomain* domain_;
  TensorView* compute_at_view_ = nullptr;
  int compute_at_axis_ = 0;

  void copyDomain(const TensorDomain* td){
    std::vector<IterDomain*> idv;
    for(decltype(td->size()) i = 0; i <td->size(); i++)
      idv.push_back(td->axis(i));
    setDomain(new TensorDomain(idv));
  }

};


TORCH_API TensorView* split(TensorView*, int axis, int factor);
TORCH_API TensorView* merge(TensorView*, int axis);
TORCH_API TensorView* reorder(TensorView*, std::unordered_map<int, int>);

} // namespace fuser
} // namespace jit
} // namespace torch
