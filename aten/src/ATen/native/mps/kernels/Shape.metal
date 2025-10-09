#include <ATen/native/mps/kernels/Shape.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T_in, typename T_out>
kernel void cat_large(
    constant T_in* input [[buffer(0)]],
    device T_out* output [[buffer(1)]],
    constant CatLargeSharedParams<>& shared_params [[buffer(2)]],
    constant CatLargeInputParams<>& input_params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto ndim = shared_params.ndim;
  auto cat_dim = shared_params.cat_dim;
  constant auto& output_strides = shared_params.output_strides;
  constant auto& output_sizes = shared_params.output_sizes;

  auto cat_dim_offset = input_params.cat_dim_offset;
  auto input_element_offset = input_params.input_element_offset;
  constant auto& input_strides = input_params.input_strides;
  constant auto& input_sizes = input_params.input_sizes;

  auto input_element_idx = static_cast<int64_t>(tid) + input_element_offset;
  int64_t input_offset = 0;
  int64_t output_offset = 0;

  for (auto dim = ndim - 1; dim >= 0; dim--) {
    auto dim_size = input_sizes[dim];
    auto input_dim_idx = input_element_idx % dim_size;
    auto output_dim_idx =
        input_dim_idx + ((dim == cat_dim) ? cat_dim_offset : 0);

    input_offset += input_strides[dim] * input_dim_idx;
    output_offset += output_strides[dim] * output_dim_idx;

    input_element_idx = input_element_idx / dim_size;
  }

  output[output_offset] = static_cast<T_out>(input[input_offset]);
}

#define REGISTER_CAT_LARGE_OP(T_in, T_out)                           \
  template [[host_name("cat_large_" #T_in "_" #T_out)]]              \
  kernel void cat_large<T_in, T_out>(                                \
      constant T_in * input [[buffer(0)]],                           \
      device T_out * output [[buffer(1)]],                           \
      constant CatLargeSharedParams<> & shared_params [[buffer(2)]], \
      constant CatLargeInputParams<> & input_params [[buffer(3)]],   \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(T_out) \
  REGISTER_CAT_LARGE_OP(float, T_out);               \
  REGISTER_CAT_LARGE_OP(half, T_out);                \
  REGISTER_CAT_LARGE_OP(bfloat, T_out);              \
  REGISTER_CAT_LARGE_OP(int, T_out);                 \
  REGISTER_CAT_LARGE_OP(uint, T_out);                \
  REGISTER_CAT_LARGE_OP(long, T_out);                \
  REGISTER_CAT_LARGE_OP(ulong, T_out);               \
  REGISTER_CAT_LARGE_OP(short, T_out);               \
  REGISTER_CAT_LARGE_OP(ushort, T_out);              \
  REGISTER_CAT_LARGE_OP(char, T_out);                \
  REGISTER_CAT_LARGE_OP(uchar, T_out);               \
  REGISTER_CAT_LARGE_OP(bool, T_out);

REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(float);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(half);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(bfloat);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(int);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(uint);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(long);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(ulong);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(short);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(ushort);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(char);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(uchar);
REGISTER_CAT_LARGE_OP_ALL_INPUT_TYPES(bool);

REGISTER_CAT_LARGE_OP(float2, float2);
REGISTER_CAT_LARGE_OP(half2, half2);
