#include <ATen/ATen.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/VectorizedCopy.h>

namespace at {
namespace native {

namespace {

template <typename T>
using VecType = at::vec256::Vec256<T>;
// Copies num_elements elements from src to dest for total of
// repeat_factor times.
template <typename DataType>
inline void single_element_broadcast(DataType* dest_ptr,
    const DataType src_element, const uint64_t repeat_factor) {
  auto src_broadcasted = at::vec256::Vec256<DataType>(src_element);
  auto size = at::vec256::Vec256<DataType>::size();
  for (uint64_t j = 0; j < repeat_factor / size; ++j) {
    src_broadcasted.store(dest_ptr);
    dest_ptr += size;
  }
  for (uint64_t j = 0; j < repeat_factor % size; ++j) {
    *dest_ptr = src_element;
    ++dest_ptr;
  }
}

template <typename DataType>
void sub_tensor_broadcast(DataType* dest_ptr, const DataType* src_ptr,
    const uint64_t num_elements, const uint64_t repeat_factor) {
  for (uint64_t j = 0; j < repeat_factor; ++j) {
    memcpy(dest_ptr, src_ptr, num_elements * sizeof(DataType));
    dest_ptr += num_elements;
  }
}

// Copies num_elements elements from src to dest for total of
// repeat_factor times.
template <typename DataType>
void replicate_last_dim(DataType* dest_ptr, const DataType* src_ptr,
    const uint64_t num_elements, const uint64_t repeat_factor) {
  // Case of stretching last dim.
  for (uint64_t i = 0; i < num_elements; ++i) {
    single_element_broadcast(dest_ptr, *src_ptr, repeat_factor);
    dest_ptr += repeat_factor;
    src_ptr++;
  }
}

template <typename DataType>
void vectorized_repeat(Tensor& output_tensor, const Tensor& input_tensor,
    IntArrayRef repeats) {
  int64_t num_dims = repeats.size();
  AT_CHECK(num_dims > 0, "Number of dims in the input tensor must be > 0");
  auto input_tensor_sizes = input_tensor.sizes();
  std::vector<int64_t> indices(num_dims, 0);
  const DataType* src_data_pointer = input_tensor.data<DataType>();
  DataType* dest_data_pointer = output_tensor.data<DataType>();
  int64_t inner_most_dim = num_dims - 1;
  // Make sure num dims is at least 1.
  while (indices[0] < input_tensor_sizes[0]) {
    uint64_t i = inner_most_dim;
    for (; i >= 1; --i ) {
      if (i == inner_most_dim) {
        if (input_tensor_sizes[i] == 1) {
          replicate_last_dim(dest_data_pointer, src_data_pointer,
              input_tensor_sizes[i], repeats[i]);
        }
        else {
          sub_tensor_broadcast(dest_data_pointer, src_data_pointer,
              input_tensor_sizes[i], repeats[i]);
        }
        indices[i - 1]++;
        src_data_pointer += input_tensor.stride(i - 1);
        dest_data_pointer += output_tensor.stride(i - 1);
      }
      else if (indices[i] == input_tensor_sizes[i]) {
        if (repeats[i] > 1) {
          int64_t output_tensor_stride = output_tensor.stride(i);
          int64_t input_dim_size = indices[i];
          DataType* orig_dest_ptr = dest_data_pointer -
            (output_tensor_stride * input_dim_size);
          uint64_t num_elements_to_replicate =
            output_tensor_stride * input_dim_size;
          sub_tensor_broadcast(dest_data_pointer, orig_dest_ptr,
              num_elements_to_replicate, repeats[i] - 1);
          dest_data_pointer = orig_dest_ptr +
            output_tensor.stride(i) * output_tensor.size(i);
        }
        indices[i] = 0;
        indices[i - 1]++;
      }
    }
  }

  if (repeats[0] > 1) {
    int64_t output_tensor_stride = output_tensor.stride(0);
    int64_t input_dim_size = indices[0];
    DataType* orig_dest_ptr = dest_data_pointer -
      (output_tensor_stride * input_dim_size);
    uint64_t num_elements_to_replicate = output_tensor_stride * input_dim_size;
    sub_tensor_broadcast(dest_data_pointer, orig_dest_ptr,
        num_elements_to_replicate, repeats[0] - 1);
  }
}

template <typename DataType>
void vectorized_repeat_single_dim(Tensor& output_tensor,
    const Tensor& input_tensor, IntArrayRef repeats) {
  DataType* input_data_ptr = input_tensor.data<DataType>();
  DataType* output_data_ptr = output_tensor.data<DataType>();
  uint64_t num_iterations = 1, i;
  uint64_t num_elements = 1;
  for (i = 0; i < repeats.size(); ++i) {
    if (repeats[i] > 1) {
      break;
    }
    num_iterations *= input_tensor.sizes()[i];
  }
  uint64_t repeat_dim_idx = i;
  for (; i < repeats.size(); ++i) {
    num_elements *= input_tensor.sizes()[i];
  }
  for (i = 0; i < num_iterations; ++i) {
    if ((repeat_dim_idx == repeats.size() - 1) &&
        (input_tensor.sizes()[repeat_dim_idx] == 1)) {
      replicate_last_dim(output_data_ptr, input_data_ptr,
          num_elements, repeats[repeat_dim_idx]);
    }
    else {
      sub_tensor_broadcast(output_data_ptr, input_data_ptr,
          num_elements, repeats[repeat_dim_idx]);
    }
    output_data_ptr += num_elements * repeats[repeat_dim_idx];
    input_data_ptr += num_elements;
  }
}

} // namespace

#define VEC_REPEAT_DISPATCH_REPLICATE_FOR_SCALAR_TYPES(T, name, _)    \
  if (xtensor.scalar_type() == ScalarType::name)                      \
  {                                                                   \
    if (num_dims_to_repeat == 1) {                                    \
      vectorized_repeat_single_dim<T>(result, xtensor, repeats);    \
    }                                                                 \
    else {                                                            \
      vectorized_repeat<T>(result, xtensor, repeats);               \
    }                                                                 \
  }

Tensor vectorized_contig_tensor_repeat(const Tensor& self,
    IntArrayRef repeats) {
  AT_CHECK(repeats.size() >= (size_t)self.dim(),
           "Number of dimensions of repeat dims can not be smaller than"
           "number of dimensions of tensor");

  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(padded_size.end(), self.sizes().begin(),
      self.sizes().end());
  std::vector<int64_t> target_size(repeats.size());

  uint64_t num_dims_to_repeat{0};
  for (size_t idx = 0; idx < repeats.size(); ++idx) {
    target_size[idx] = padded_size[idx] * repeats[idx];
    if (repeats[idx] > 1) {
      num_dims_to_repeat++;
    }
  }

  Tensor result = at::empty(target_size, self.options());
  // We do not need to create alias on result because we do not rely
  // on unfold like the original repeat does in order to create strides
  // that help with copy op.

  if (num_new_dimensions > 0) {
    Tensor xtensor = self.expand(padded_size);
    AT_FORALL_SCALAR_TYPES_EXCEPT_QINT(VEC_REPEAT_DISPATCH_REPLICATE_FOR_SCALAR_TYPES)
  }
  else {
    const Tensor& xtensor = self;
    AT_FORALL_SCALAR_TYPES_EXCEPT_QINT(VEC_REPEAT_DISPATCH_REPLICATE_FOR_SCALAR_TYPES)
  }

  return result;
}

} // native
} // at
