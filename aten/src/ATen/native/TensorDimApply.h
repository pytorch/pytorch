#include <ATen/ATen.h>

namespace at {
  namespace native {
    //input tensors are non-zero dim and non-empty
    template<typename T1, typename T2, typename Function>
    void tensor_dim_apply3(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim, Function func) {
      int ndims = self.dim();
      int tensor_dim_apply_has_finished = 0;
      std::vector<int64_t> counter(ndims, 0);
      T1* self_data = self.data_ptr<T1>();
      T1* values_data = values.data_ptr<T1>();
      T2* indices_data = indices.data_ptr<T2>();
      int64_t self_stride = self.stride(dim);
      int64_t values_stride = values.stride(dim);
      int64_t indices_stride = indices.stride(dim);
      int self_dim_size = self.size(dim);

      while(!tensor_dim_apply_has_finished) {
        func(self_data, values_data, indices_data, self_dim_size, self_stride, values_stride, indices_stride);
        if(ndims == 1)
           break;
        for(int dim_i = 0; dim_i < ndims; dim_i++) {
          if(dim_i == dim) {
            if(dim_i == (ndims - 1)) {
              tensor_dim_apply_has_finished = 1;
              break;
            }
            continue;
          }
          counter[dim_i]++;
          self_data += self.stride(dim_i);
          values_data += values.stride(dim_i);
          indices_data += indices.stride(dim_i);

          if(counter[dim_i] == self.size(dim_i)) {
            if(dim_i == ndims-1) {
              tensor_dim_apply_has_finished = 1;
              break;
            } else {
              self_data -= counter[dim_i]*self.stride(dim_i);
              values_data -= counter[dim_i]*values.stride(dim_i);
              indices_data -= counter[dim_i]*indices.stride(dim_i);
              counter[dim_i] = 0;
            }
          } else {
            break;
         }
        }
      }
    }
}}
