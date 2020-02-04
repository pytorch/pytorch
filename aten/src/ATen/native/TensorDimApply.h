#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace at {
  namespace native {
    template<typename T1, typename T2, typename Operation>
    void tensor_dim_apply3(const Tensor& self, Tensor& values, Tensor& indices, int64_t dim, Operation op) {
      int ndims = self.dim();
      if(ndims == 0) {
        values.fill_(self);
        indices.fill_(0);
      } else if(self.numel() != 0) {
        int tensor_dim_apply_has_finished = 0;
        int tensor_dim_apply_i;
        int64_t tensor_dim_apply_counter[ndims];
        for(tensor_dim_apply_i = 0; tensor_dim_apply_i < ndims; tensor_dim_apply_i++)
          tensor_dim_apply_counter[tensor_dim_apply_i] = 0;
        T1* self_data = self.data_ptr<T1>();
        T2* values_data = values.data_ptr<T2>();
        int64_t* indices_data = indices.data_ptr<int64_t>();
        int64_t self_stride = self.stride(dim);
        int64_t values_stride = values.stride(dim);
        int64_t indices_stride = indices.stride(dim);
        int self_dim_size = self.size(dim);

        while(!tensor_dim_apply_has_finished) {
          op(self_data, values_data, indices_data, self_dim_size, self_stride, values_stride, indices_stride);
          if(ndims == 1)
             break;
          for(tensor_dim_apply_i = 0; tensor_dim_apply_i < ndims; tensor_dim_apply_i++) {
            if(tensor_dim_apply_i == dim) {
              if(tensor_dim_apply_i == (ndims - 1)) {
                tensor_dim_apply_has_finished = 1;
                break;
              }
              continue;
            }
            tensor_dim_apply_counter[tensor_dim_apply_i]++;
            self_data += self.stride(tensor_dim_apply_i);
            values_data += values.stride(tensor_dim_apply_i);
            indices_data += indices.stride(tensor_dim_apply_i);

            if(tensor_dim_apply_counter[tensor_dim_apply_i] == self.size(tensor_dim_apply_i)) {
              if(tensor_dim_apply_i == ndims-1) {
                tensor_dim_apply_has_finished = 1;
                break;
              } else {
                self_data -= tensor_dim_apply_counter[tensor_dim_apply_i]*self.stride(tensor_dim_apply_i);
                values_data -= tensor_dim_apply_counter[tensor_dim_apply_i]*values.stride(tensor_dim_apply_i);
                indices_data -= tensor_dim_apply_counter[tensor_dim_apply_i]*indices.stride(tensor_dim_apply_i);
                tensor_dim_apply_counter[tensor_dim_apply_i] = 0;
              }
            } else {
              break;
           }
          }
        }
      }
    }
}}
