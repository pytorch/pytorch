#include "ATen/ATen.h"
#include "ATen/TensorUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/AccumulateType.h"

#ifdef _OPENMP
#include <omp.h>
#endif


#ifdef _MSC_VER
  #define LOG_SOFTMAX_SIZE_TYPE int64_t
  #define LOG_SOFTMAX_CAST_TYPE (int64_t)
#else
  #define LOG_SOFTMAX_SIZE_TYPE uint64_t
  #define LOG_SOFTMAX_CAST_TYPE
#endif

namespace at {
namespace native {

namespace{

template<typename scalar_t, bool LogSoftMax>//, template<typename> class Epilogue>
void host_softmax(Tensor output, const Tensor & input_, const int64_t dim_){
  auto input = input_.contiguous();
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  if (input.dim() == 0) input = input.view(1);
  AT_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  uint64_t outer_size = 1;
  uint64_t dim_size = input.size(dim);
  uint64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input.size(i);
  for (int64_t i = dim + 1; i < input.dim(); ++i)
    inner_size *= input.size(i);
  uint64_t dim_stride = inner_size;
  uint64_t outer_stride = dim_size * dim_stride; 
  scalar_t *input_data_base  = input.data<scalar_t>();
  scalar_t *output_data_base = output.data<scalar_t>();
  LOG_SOFTMAX_SIZE_TYPE i, d;
#pragma omp parallel for private(i, d)
  for (i = 0; i < LOG_SOFTMAX_CAST_TYPE (outer_size * inner_size); i++)
  {
    uint64_t outer_idx = i / inner_size;
    uint64_t inner_idx = i % inner_size;
    scalar_t *input_data  = input_data_base  + outer_idx * outer_stride + inner_idx;
    scalar_t *output_data = output_data_base + outer_idx * outer_stride + inner_idx;

    scalar_t max_input = -std::numeric_limits<scalar_t>::max();
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      max_input = std::max(max_input, input_data[d * dim_stride]);

    using accscalar_t =  acc_type<scalar_t, false>; 
    accscalar_t tmpsum = 0;
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++){
      scalar_t z = std::exp(input_data[d * dim_stride] - max_input);
      tmpsum += z;
      if (!LogSoftMax){
        output_data[d*dim_stride] = z;
      }
    }
  
    if (LogSoftMax)
       tmpsum = max_input + std::log(tmpsum);
    else
       tmpsum = 1 / tmpsum;

    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      if (LogSoftMax)
         output_data[d * dim_stride] = input_data[d * dim_stride] - tmpsum;
      else
         output_data[d * dim_stride] *= tmpsum; 
  }

}

template<typename scalar_t, bool LogSoftMax>
void host_softmax_backward(Tensor gI, const Tensor &grad_, const Tensor &output_, int64_t dim_){   
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  auto grad = grad_.contiguous();
  auto output = output_.contiguous();
  if (grad.dim() == 0) grad = grad.view(1);
  if (output.dim() == 0) output = output.view(1);
  AT_CHECK(dim >=0 && dim < grad.dim(), "dim must be non-negative and less than input dimensions");

  uint64_t outer_size = 1;
  uint64_t dim_size = grad.size(dim);
  uint64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= grad.size(i);
  for (int64_t i = dim + 1; i < grad.dim(); ++i)
    inner_size *= grad.size(i);
  uint64_t dim_stride = inner_size;
  uint64_t outer_stride = dim_size * dim_stride; 
  scalar_t *gradInput_data_base  = gI.data<scalar_t>();
  scalar_t *output_data_base = output.data<scalar_t>();
  scalar_t *gradOutput_data_base  = grad.data<scalar_t>();
  LOG_SOFTMAX_SIZE_TYPE i, d;

#pragma omp parallel for private(i, d)
  for (i = 0; i < LOG_SOFTMAX_CAST_TYPE (outer_size * inner_size); i++)
  {
    uint64_t outer_idx = i / inner_size;
    uint64_t inner_idx = i % inner_size;
    scalar_t *gradInput_data  = gradInput_data_base  + outer_idx * outer_stride + inner_idx;
    scalar_t *output_data     = output_data_base     + outer_idx * outer_stride + inner_idx;
    scalar_t *gradOutput_data = gradOutput_data_base + outer_idx * outer_stride + inner_idx;

    using accscalar_t =  acc_type<scalar_t, false>; 
    accscalar_t sum = 0;//TODO was accreal here
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      if (LogSoftMax) 
          sum += gradOutput_data[d * dim_stride];
      else 
          sum += gradOutput_data[d * dim_stride] * output_data[d * dim_stride]; 

    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      if (LogSoftMax)
          gradInput_data[d * dim_stride] = gradOutput_data[d * dim_stride] - std::exp(output_data[d * dim_stride]) * sum;
      else 
          gradInput_data[d * dim_stride] = output_data[d * dim_stride] * (gradOutput_data[d * dim_stride] - sum);
  }

}
}



Tensor log_softmax_cpu(const Tensor &input, const int64_t dim){
  Tensor output = at::native::empty_like(input);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax_forward", [&]{
     host_softmax<scalar_t, true>(output, input, dim);
  });
  return output;
}

Tensor log_softmax_backward_cpu(const Tensor &grad, const Tensor &output, int64_t dim, const Tensor &input){
  TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
  checkSameSize("log_softmax_backward", grad_arg, output_arg);
  Tensor gI = at::native::empty_like(grad);
  AT_DISPATCH_FLOATING_TYPES(grad.type(), "log_softmax_backward", [&]{
     host_softmax_backward<scalar_t, true>(gI, grad, output, dim);
  });
  return gI;
}

Tensor softmax_cpu(const Tensor &input, const int64_t dim){  
  Tensor output = at::native::empty_like(input);
  AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax_forward", [&]{
     host_softmax<scalar_t, false>(output, input, dim);
  });
  return output;
}

Tensor softmax_backward_cpu(const Tensor &grad, const Tensor &output, int64_t dim, const Tensor &input){
  TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
  checkSameSize("log_softmax_backward", grad_arg, output_arg);
  Tensor gI = at::native::empty_like(grad);
  AT_DISPATCH_FLOATING_TYPES(grad.type(), "log_softmax_backward", [&]{
     host_softmax_backward<scalar_t, false>(gI, grad, output, dim);
  });
  return gI;
}



}
}
