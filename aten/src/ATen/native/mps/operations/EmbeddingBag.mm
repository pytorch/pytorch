 #define TORCH_ASSERT_ONLY_METHOD_OPERATORS                                           
 #include <ATen/core/Tensor.h>                                                        
 #include <ATen/AccumulateType.h>                                                     
 #include <ATen/ceil_div.h>                                                           
 #include <ATen/Dispatch.h>                                                           
 #include <ATen/native/EmbeddingBag.h>                                                
 #include <ATen/TensorUtils.h>  
 #include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS                                                                                                                                                                                                                                             
#include <ATen/Functions.h>                                                          
#include <ATen/NativeFunctions.h>                                                    
#else                                                                                
#include <ATen/ops/arange.h>                                                         
#include <ATen/ops/empty.h>                                                          
#include <ATen/ops/empty_like.h>                                                     
#include <ATen/ops/zeros.h>                                                          
#include <ATen/ops/embedding_bag_native.h>                                          
#include <ATen/ops/_embedding_bag_native.h>                                          
#include <ATen/ops/_embedding_bag_forward_only_native.h>                             
#include <ATen/ops/_embedding_bag_dense_backward_native.h>                           
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_native.h>              
#endif  

namespace at::native { 
// Assumes all input tensors are contiguous.
// See NOTE [ embedding_bag Native Functions ] in native_functions.yaml for details
std::tuple<Tensor, Tensor, Tensor, Tensor>
_embedding_bag_forward_only_mps(const Tensor &weight, const Tensor &indices,
                   const Tensor &offsets, const bool scale_grad_by_freq,
                   const int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights_opt,
                   bool include_last_offset, int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  return _embedding_bag_mps(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx);
}

std::tuple<Tensor, Tensor, Tensor, Tensor>                                           
 _embedding_bag_mps(const Tensor &weight, const Tensor &indices_,                    
                    const Tensor &offsets_, const bool scale_grad_by_freq,            
                    const int64_t mode, bool sparse, const std::optional<Tensor>& per_sample_weights_opt,
                    bool include_last_offset, int64_t padding_idx) {                  
   TORCH_CHECK(indices_.dim() == 1 || indices_.dim() == 2,                            
       "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",             
       indices_.dim());                                                               
   if (indices_.dim() == 1) {                                                         
     TORCH_CHECK(offsets_.dim() == 1,                                                 
         "offsets has to be a 1D Tensor, but got Tensor of dimension ",               
         offsets_.dim());                                                             
   }                                                                                  
   TORCH_CHECK(weight.dim() == 2,                                                     
       "weight has to be a 2D Tensor, but got Tensor of dimension ",                  
       weight.dim());                                                                 
   // See [Note: hacky wrapper removal for optional tensor]                           
   c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
   const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;                
                                                                                      
   Tensor indices, offsets;                                                           
//   std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);         
//   auto indices_arg = TensorArg(indices, "indices", 1);                               
//   checkScalarTypes("embedding_bag_mps", indices_arg, {kLong, kInt});                
//   auto offsets_arg = TensorArg(offsets, "offsets", 1);                               
//   checkScalarTypes("embedding_bag_mps", offsets_arg, {kLong, kInt});                
//   checkSameType("embedding_bag_mps", indices_arg, offsets_arg);                     
//   auto weight_arg = TensorArg(weight, "weight", 1);                                  
//   checkSameGPU("embedding_bag_mps", weight_arg, indices_arg);                       
//   checkSameGPU("embedding_bag_mps", weight_arg, offsets_arg);                       
                                                                                      
   int64_t numIndices = indices.size(0);                                              
   int64_t numBags = offsets.size(0);                                                 
//   if (include_last_offset) {                                                         
//     // Check https://github.com/pytorch/pytorch/issues/29019                                                                                                                                                                                                                
//     // We plan to add one more element in offsets, which is equal to the size of     
//     // indices. Currently for cuda devices, we still use the legacy                  
//     // implementation even this flag is enabled.                                     
//     TORCH_CHECK(                                                                     
//         numBags >= 1, "include_last_offset: numBags should be at least 1");          
//     numBags -= 1;                                                                    
//   }                                                                                  
//   int64_t featureSize = weight.size(1);                                              
//                                                                                      
//   auto bag_size = at::empty(offsets.sizes(), indices.options());                     
//   auto offset2bag =                                                                  
//       at::empty({indices.size(0)}, indices.options()); // offset2bag = [0 0 0 0 0]   
//                                                                                      
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();                            
//                                                                                      
//   auto output = at::empty({numBags, featureSize}, weight.options());                 
//                                                                                      
//   Tensor max_indices;                                                                
//                                                                                      
//   if (mode == EmbeddingBagMode::MAX) {                                               
//     max_indices = at::empty({numBags, featureSize}, indices.options());              
//   } else {                                                                           
//     // No need to allocate if we aren't doing a backwards pass                       
//     max_indices = at::empty({0}, indices.options());                                 
//   }                                      
//#if defined(USE_ROCM)                                                                
//  dim3 block = dim3(64, 4);                                                          
//#else                                                                                
//  dim3 block = dim3(32, 8);                                                          
//#endif                                                                               
//  int grid = 1024;                                                                   
//  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, weight.scalar_type(), "embedding_bag_cuda", [&] {
//    AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cuda", [&] () {    
//      if (mode == EmbeddingBagMode::MAX) {                                           
//        EmbeddingBag_updateOutputKernel_max<scalar_t, index_t><<<grid, block, 0, stream>>>(
//            indices.const_data_ptr<index_t>(), offsets.const_data_ptr<index_t>(),    
//            weight.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(),  
//            offset2bag.mutable_data_ptr<index_t>(), numIndices, numBags, featureSize,
//            weight.stride(0), weight.stride(1), bag_size.mutable_data_ptr<index_t>(),
//            max_indices.mutable_data_ptr<index_t>(),                                 
//            padding_idx, weight.size(0));                                            
//        C10_CUDA_KERNEL_LAUNCH_CHECK();                                              
//      } else {                                                                       
//        EmbeddingBag_updateOutputKernel_sum_mean<scalar_t, index_t><<<grid, block, 0, stream>>>(
//            indices.const_data_ptr<index_t>(), offsets.const_data_ptr<index_t>(),    
//            weight.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(),  
//            offset2bag.mutable_data_ptr<index_t>(), numIndices, numBags, featureSize,
//            weight.stride(0), weight.stride(1), mode, bag_size.mutable_data_ptr<index_t>(),
//            per_sample_weights.defined() ? per_sample_weights.const_data_ptr<scalar_t>() : NULL,
//            per_sample_weights.defined() ? per_sample_weights.stride(0) : 0,         
//            padding_idx, weight.size(0));                                            
//        C10_CUDA_KERNEL_LAUNCH_CHECK();                                              
//      }                                                                              
//    });                                                                              
//  });                                                                                
//                                                                                     
//  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, offset2bag, bag_size, max_indices);
    return std::tuple<Tensor, Tensor, Tensor, Tensor>();
}   

Tensor _embedding_bag_dense_backward_mps(const Tensor &grad_, const Tensor &indices,
                                   const Tensor &offset2bag,                         
                                   const Tensor &bag_size_,                          
                                   const Tensor &max_indices,                        
                                   int64_t num_weights,                              
                                   bool scale_grad_by_freq, int64_t mode, const std::optional<Tensor>& per_sample_weights_opt,
                                   int64_t padding_idx) {                            
  // See [Note: hacky wrapper removal for optional tensor]                           
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned = at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;                
                                                                                     
  // indices, offsets and offset2bag are assumed having correct dtypes and           
  // contiguous here due to the checks in _embedding_bag_backward in                 
  // EmbeddingBag.cpp.                                                               
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml       
  // for more details.                                                               
                                                                                     
  Tensor grad = grad_.contiguous();                                                  
  auto indices_arg = TensorArg(indices, "indices", 1);                               
  auto grad_arg = TensorArg(grad, "grad", 1);                                        
//  checkSameGPU("embedding_bag_cuda", grad_arg, indices_arg);                         
                                                                                     
                                                                                     
//  switch (static_cast<EmbeddingBagMode>(mode)) {                                     
//    case EmbeddingBagMode::SUM:                                                      
//    case EmbeddingBagMode::MEAN:                                                     
//      if (mode == EmbeddingBagMode::MEAN)                                            
//        AT_ASSERT(!per_sample_weights.defined());                                    
//      return embedding_bag_backward_cuda_sum_avg(grad, indices, offset2bag,          
//              bag_size_, num_weights, scale_grad_by_freq, mode,                      
//              per_sample_weights, padding_idx);                                      
//                                                                                     
//    case EmbeddingBagMode::MAX:                                                      
//      AT_ASSERT(!per_sample_weights.defined());                                      
//      return embedding_bag_backward_cuda_max(grad, max_indices, num_weights,         
//              padding_idx);                                                          
//                                                                                     
//    default:                                                                                                                                                                                                                                                                
//      AT_ERROR(                                                                      
//          "Unknown mode for embedding_bag_backward_cuda ", mode);                    
//  }                                                                                  
    return grad_;
}

Tensor _embedding_bag_per_sample_weights_backward_mps(
    const Tensor& grad,
    const Tensor& weight,  // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
//  TORCH_CHECK(
//      mode == EmbeddingBagMode::SUM,
//      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");
//
  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

 Tensor indices, offsets;
// std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
 AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);
//
//  AT_ASSERT(weight.dim() == 2);
//  AT_ASSERT(weight.size(1) == embedding_features);
//
//  const int threads_per_block = 512;
//  const int warps_per_block = threads_per_block / at::cuda::warp_size();
//
//  dim3 block(threads_per_block);
//  dim3 grid((num_samples + warps_per_block - 1) / warps_per_block);
//
  auto output = at::empty({num_samples}, grad.options());
//
//  // Early return when there is no samples in the batch. This saves unnecessary kernel
//  // launch, but also prevents cudaGetLastError() to complain about invalid launch args
//  if (num_samples == 0) {
//    return output;
//  }
//
//  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//    grad.scalar_type(), "_embedding_bag_per_sample_weights_backward_cuda", [&]() {
//      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "_embedding_bag_per_sample_weights_backward_cuda", [&]() {
//        _embedding_bag_per_sample_weights_backward_kernel<scalar_t, index_t>
//          <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
//            grad.const_data_ptr<scalar_t>(), grad.stride(0), grad.stride(1),
//            weight.const_data_ptr<scalar_t>(), weight.stride(0), weight.stride(1),
//            indices.const_data_ptr<index_t>(),
//            offset2bag.const_data_ptr<index_t>(),
//            num_samples,
//            embedding_features,
//            output.mutable_data_ptr<scalar_t>(),
//            padding_idx);
//        C10_CUDA_KERNEL_LAUNCH_CHECK();
//      });
//    }
//  );
  return output;
}

} // at::native
