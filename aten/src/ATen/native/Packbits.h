#pragma once

// define constants like M_PI and C keywords for MSVC
#include "Functions.h"
#include "torch/types.h"
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Utils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TracerMode.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Deprecated.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/TensorOptions.h>
#include <TH/THAllocator.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/util/Exception.h>
#include <ATen/NamedTensorUtils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>

namespace at {
namespace native {

static inline Tensor pack1d(const Tensor& bit_array, bool big_endian) {
  auto window = big_endian ? native::arange(7, -1, -1) : at::arange(8);
  auto pow_window = native::pow(2, window);

  auto opt = TensorOptions(torch::kUInt8);

  int64_t num_elements = bit_array.numel();

  auto result = at::empty(std::ceil(double(num_elements) / 8), opt);

  for (int i = 0; i < num_elements; i += 8) {
    if ((i + 8) > num_elements) {
      auto t_zeros = at::zeros(8);
      t_zeros.slice(0, 0, num_elements - i).copy_(bit_array.slice(0, i));
      result[i/8] = at::sum(pow_window * t_zeros);
    }
    else 
      result[i/8] = at::sum(pow_window * bit_array.slice(0, i, i+8));
  }

  return result;
}

static inline Tensor packnd(const Tensor& bit_array, int64_t dim, bool big_endian) {

  auto out_dims_vec = bit_array.sizes().vec();
  int64_t last_dim = out_dims_vec.size() - 1;

  out_dims_vec[dim] = std::ceil(double(out_dims_vec[dim]) / 8);
  
  auto out_dims = at::IntArrayRef(out_dims_vec);
  auto opt = TensorOptions(torch::kUInt8);
  auto result = at::empty(out_dims, opt);

  auto bit_array_tc = bit_array.transpose(dim, last_dim).contiguous();

  int64_t ldim_size = bit_array.size(dim);
  auto source_dptr = bit_array_tc.data_ptr<bool>();
  auto dest_dptr = result.data_ptr<uint8_t>();
  int64_t win_start, win_end, win_size, source_idx = 0, dest_idx = 0;
  
  while (source_idx < bit_array_tc.numel()) {
    int dim_end = source_idx + ldim_size;
    uint8_t cur_pack_val = 0;

    for (win_start = source_idx; win_start < dim_end; win_start += 8) {
      win_end = (win_start + 8 < dim_end) ? win_start + 8 : dim_end;
      win_size = win_end - win_start;
      if (big_endian) {
        for (int64_t i = win_start; i < win_end; i++)
          cur_pack_val = (cur_pack_val << 1) + source_dptr[i];

        cur_pack_val <<= (8 - win_size);
      } else {
        for (int i = win_end - 1; i >= win_start; i--)
          cur_pack_val = (cur_pack_val << 1) + source_dptr[i];
      }

      dest_dptr[dest_idx] = cur_pack_val;      
      dest_idx++;
    }
    source_idx = dim_end;
  }

return result;
}

Tensor packbits_helper(const Tensor& bit_array, c10::optional<int64_t> dim, bool big_endian) {
TORCH_CHECK(true, "packbits works with boolean tensors only");

  if (dim) {
    int64_t dim_non_optional = dim.value();
    auto out_dims_vec = bit_array.sizes().vec();
    int64_t last_dim = out_dims_vec.size() - 1;
    dim_non_optional = (dim_non_optional == -1) ? last_dim : dim_non_optional;
    out_dims_vec[dim_non_optional] = std::ceil(double(out_dims_vec[dim_non_optional]) / 8);

    return packnd(bit_array, dim_non_optional, big_endian);
  } else
    return packnd(bit_array.flatten(), 0, big_endian);
}

}
}
