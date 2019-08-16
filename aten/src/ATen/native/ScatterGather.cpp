#include <ATen/ATen.h>


void THTensor_(gather)(THTensor *tensor, THTensor *src, int dim, THLongTensor *index)
{
}


namespace at { namespace native {

Tensor & gather_out_cpu(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  TORCH_CHECK(std::max(index.dim(), 1) == std::max(self.dim(), 1), "Index tensor must have same dimensions as input tensor");
  TORCH_CHECK(dim >= 0 && dim < std::max(self.dim(), 1), "Index dimension is out of bounds");
  TORCH_CHECK(std::max(result.dim(), 1) == std::max(self.dim(), 1)), "Input tensor must have same dimensions as output tensor");

  int64_t elems_per_row = (index->dim() == 0 ? 1 : index->size(dim));
  CPU_tensor_apply3;  // @ pytorch/aten/src/ATen/CPUApplyUtils.h
    TH_TENSOR_DIM_APPLY3(scalar_t, tensor, scalar_t, src, int64_t, index, dim,
                        TH_TENSOR_DIM_APPLY3_SIZE_EQ_EXCEPT_DIM,
                        for (int64_t i = 0; i < elems_per_row; ++i)
                        {
                            int64_t idx = *(index_data + i*index_stride);
                            if (idx < 0 || idx >= src_size)
                            {
                            THFree(TH_TENSOR_DIM_APPLY_counter);
                            THError("Invalid index in gather");
                            }
                            *(tensor_data + i*tensor_stride) = src_data[idx * src_stride];
                        })
  return result;
}

Tensor gather_cpu(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather(self, dim, index);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar source) {
  return self.clone().scatter_(dim, index, source);
}

Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone().scatter_add_(dim, index, source);
}

}}  // namespace at::native
