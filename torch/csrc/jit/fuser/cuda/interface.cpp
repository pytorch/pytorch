#include <torch/csrc/jit/fuser/cuda/interface.h>


namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

std::vector<bool> canCollapseDimsDown(const std::shared_ptr<c10::TensorType> tensor){
  int64_t ndims = *(tensor->dim());

  //Flags to see if the current dim can be fused with the one after
  //Goes left to right, furthest right doesn't need a flag
  std::vector<bool> canCollapseDown(ndims, true);

  for (int64_t d = 0; d < ndims - 1; d++) {
    int64_t stride = *(tensor->strides()[d]);
    int64_t stride_p_1 = *(tensor->strides()[d+1]);
    int64_t size_p_1 = *(tensor->sizes()[d+1]);

    if( (stride_p_1 * size_p_1 != stride)
	&& !(stride_p_1 == 0 && stride == 0) )
      canCollapseDown[d] = false;

  }

  canCollapseDown[ndims-1] = true;

  return canCollapseDown;
}

// Returns true if the node is added to the fusion group, false o.w.
int tryCreateFusion(const Node* const node) {
  int64_t ndims = *(node->inputs()[0]->type()->expect<TensorType>()->dim());
  std::vector< std::vector<bool> > collapse_vecs;


  //Check how we could dimensionally reduce each input
  for(const auto& value : node->inputs())
    if(value->isCompleteTensor()){
      assert(*(value->type()->expect<TensorType>()->dim()) == ndims);
      collapse_vecs.push_back(canCollapseDimsDown(value->type()->expect<TensorType>()));
    }

  //Check how we could dimennsionally reduce each output
  for(const auto& value : node->outputs())
    if(value->isCompleteTensor()){
      assert(*(value->type()->expect<TensorType>()->dim()) == ndims);
      collapse_vecs.push_back(canCollapseDimsDown(value->type()->expect<TensorType>()));
    }

  std::vector<bool> dim_collapse = collapse_vecs[0];

  for(auto it = collapse_vecs.begin() + 1; it!=collapse_vecs.end(); ++it){
    for(int64_t d = 0; d<ndims; d++){
      dim_collapse[d] = dim_collapse[d] && (*it)[d];
    }
  }

  //Contig not the right word here because the tensor:
  //Size(4, 4, 2) stride(16, 4, 2) will be fully
  //collapsable but not contiguous
  bool contig = true;
  for(const auto iscontig : dim_collapse)
    contig = contig && iscontig;

  if(contig)
    std::cout<<"All tensors are contiguous"<<std::endl;

  bool first = true;
  for (auto i = decltype(dim_collapse.size()){0}; i < dim_collapse.size() - 1 ; ++i) {
    if(dim_collapse[i]){
      if(first){
	std::cout<<"Tensors could be collapsed on Dims = ("<<i;
	first = false;
      }else{
	std::cout<<", "<<i;
      }
    }
  }
  if(!first) std::cout<<")"<<std::endl;


  if(node->kind() ==  aten::add){
    std::cout<<"Can fuse node!"<<std::endl;
    return -1;
  }

  return -1;
}

}}}}
