#pragma once

#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

/* This is a function which doing dim apply using tensor iterator.
 * It tries to provide a generic way for doing dim apply (scan) operation.
 * iter - a tensor iterator which created by dim_apply_op() or similar apis.
 * func - func(char** data, const int64_t *dim_strides, const int64_t dim_size)
 *   data - is a char pointer array that contains the start byte address
 *     of each tensor at current iteration along the dim apply dimension.
 *   dim_strides - an int64 array that contains the original stride of
 *     each tensor at dim apply dimension
 *   dim_size - the size of all tensors in tensor iterator at dim apply dimension,
 *     note that all tensors in the tensor iterator should
 *     have same shape.
 *
 *   data and dim_strides array have the same order, this order is the same as
 *   the order that input/ouput tensors being added into tensor iterator.
 *
 * eg. 2 tensors, an input [[1,2,3],[4,5,6]] and a result that has same shape
 *     suppose tensor iterator added result first, then input, and we are
 *     doing dim apply from dimension 0.
 *
 *   func(char** data, const int64_t *dim_strides, const int64_t dim_size) {
 *     data[0] -> start byte address of result
 *     data[1] -> start byte address of input
 *     dim_strides[0] -> origional stride of result at dimension 0
 *     dim_strides[1] -> origional stride of input at dimension 0
 *     dim_size -> size at dimension 0 for all tensors
 *
 *     // to scan each element from current start byte
 *     for (int64_t i = 0; i <= dim_size; ++i) {
 *       (scalar_t*)(data[0])[i * dim_strides[0]) -> current item in result tensor
 *       (scalar_t*)(data[1])[i * dim_strides[1]) -> current item in input tensor
 *     }
 *   }
 *
 *   dim_size == 2 here,
 *   data[1] will be 1 in 1st iteration, then 2, then 3.
 *   data[0] will be the corresponding byte start in result tensor.
 *   So in 1st iteration you can scan [1,4], 2nd iteration [2,5] and then [3,6],
 *   doing calculation and store result in data[0], you might need to cast
 *   the char* to your current scalar type.
 *
 * If additional value is needed in the func, you can make it as a lambda and
 * pass whatever parameter into the function.
 *
 * You can find more examples in native/ReduceOps.cpp
 *
 */
template <typename func_t>
void dim_apply_cpu(TensorIterator& iter, func_t&& func) {
  const DimApplyInfo& dim_apply_info = iter.get_dim_apply_info();
  TORCH_INTERNAL_ASSERT(dim_apply_info.is_enabled());

  int ntensors = iter.ntensors();
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    std::vector<char*> data_bytes(ntensors);
    for (int64_t i = 0; i < ntensors; ++i) {
      data_bytes[i] = data[i];
    }
    for (int64_t i = 0; i < n; ++i) {
      func(
        data_bytes.data(),
        dim_apply_info.dim_original_strides.data(),
        dim_apply_info.dimension_size);
      for (int64_t j = 0; j < ntensors; ++j) {
        data_bytes[j] += strides[j];
      }
    }
  };
  iter.for_each(loop);
  iter.cast_outputs();
}

inline TensorIterator dim_apply_op(Tensor& output, const Tensor& input, int64_t dim) {
  auto iter = TensorIterator();
  iter.add_output(output);
  iter.add_input(input);
  iter.set_dim_apply_dimension(dim);
  iter.promote_common_dtype();
  iter.dont_resize_outputs();
  iter.build();
  return iter;
}

inline TensorIterator dim_apply_op(Tensor& output, Tensor& input1, Tensor& input2, int64_t dim) {
  auto iter = TensorIterator();
  iter.add_output(output);
  iter.add_input(input1);
  iter.add_input(input2);
  iter.set_dim_apply_dimension(dim);
  iter.promote_common_dtype();
  iter.dont_resize_outputs();
  iter.build();
  return iter;
}

inline TensorIterator dim_apply_op(Tensor& output, Tensor& input1, Tensor& input2, Tensor& input3, int64_t dim) {
  auto iter = TensorIterator();
  iter.add_output(output);
  iter.add_input(input1);
  iter.add_input(input2);
  iter.add_input(input3);
  iter.set_dim_apply_dimension(dim);
  iter.promote_common_dtype();
  iter.dont_resize_outputs();
  iter.build();
  return iter;
}


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
