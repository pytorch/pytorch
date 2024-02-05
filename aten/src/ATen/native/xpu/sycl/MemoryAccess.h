#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/core/DynamicCast.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <c10/macros/Macros.h>
#include <ATen/core/Array.h>
#include <ATen/detail/OffsetCalculator.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/detail/MemoryAccessUtils.h>

namespace at { namespace native { namespace memory {
namespace policies {

// Assumption:
// all tensors are contiguous, that is: stride == sizeof(type) for all tensors
template <
    int item_work_size,
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t,
    int num_outputs = 1>
struct unroll {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  loader_t loader;
  storer_t storer;
  int item_idx;
  int group_idx;
  int num_items_per_group;
  int group_work_size;

  unroll(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s,
      int item_idx,
      int group_idx,
      int num_items_per_group)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        loader(l),
        storer(s),
        item_idx(item_idx),
        group_idx(group_idx),
        num_items_per_group(num_items_per_group),
        group_work_size(item_work_size * num_items_per_group) {}

  inline bool check_inbounds(int item_work_elem) const {
    return (item_idx + item_work_elem * num_items_per_group < remaining);
  }

  template <typename args_t>
  inline void load(args_t* args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int item_idx_ = item_idx;
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      if (item_idx_ >= remaining) {
        return;
      }
      int linear_idx = item_idx_ + group_work_size * group_idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(
          *this, args, offset, loader, i, num_outputs);
      item_idx_ += num_items_per_group;
    }
  }

  template <typename scalar_t>
  inline void store(scalar_t* from) {
    int item_idx_ = item_idx;
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      if (item_idx_ >= remaining) {
        return;
      }
      int linear_idx = item_idx_ + group_work_size * group_idx;
      int offset = output_offset_calculator.get(linear_idx)[0];
      storer.store(from[i], data[0], offset);
      item_idx_ += num_items_per_group;
    }
  }
};

// Assumption:
// 1. tensors could be contiguous, that is: stride == sizeof(type).
// 2. tensors could be broadcasted.
template <
    int vec_size,
    typename data_t,
    typename inp_calc_t>
struct vectorized {
  data_t data;
  inp_calc_t input_offset_calculator;
  int item_idx;
  int group_idx;
  int num_items_per_group;
  int group_work_size;

  vectorized(
      data_t data,
      inp_calc_t ic,
      int item_idx,
      int group_idx,
      int num_items_per_group)
      : data(data),
        input_offset_calculator(ic),
        item_idx(item_idx),
        group_idx(group_idx),
        num_items_per_group(num_items_per_group),
        group_work_size(vec_size * num_items_per_group) {}

  inline constexpr bool check_inbounds(int item_work_elem) const {
    return true;
  }

  template <typename accessor_t, typename scalar_t>
  inline void load_single_arg(accessor_t to, scalar_t* from) {
    int index = item_idx + num_items_per_group;
    auto v = load_vector<vec_size>(from, index);
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      to(j) = v.val[j];
    }
  }

  inline int get_offset(typename inp_calc_t::offset_type offset, int arg_index) {
    return offset[arg_index];
  }

  template <typename args_t>
  inline void load(args_t* args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int group_offset = group_work_size * group_idx;
    // `Unroll` policy cannot feed memory bandwidth well on Intel GPU,
    // 1. Small loop size cannot provide enough payloads, specially for small
    //    size data type s8/u8/f16/bf16.
    // 2. Large loop size leads to register pressure and register spill.
    // Apply `Vectorized` policy for some 'non-contiguous' cases,
    // like broadcast. The broadcasted operands are dense and could be
    // optimized by the policy if satisfying vectorization conditions.
    auto offset = input_offset_calculator.get(group_offset);
    detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(*this, args, offset);
  }

  template <typename scalar_t>
  inline void store(scalar_t* from) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    scalar_t* to =
        reinterpret_cast<scalar_t*>(data[0]) + group_work_size * group_idx;
    vec_t* to_ = reinterpret_cast<vec_t*>(to);

    int index = item_idx + num_items_per_group;
    vec_t v;
#pragma unroll
    for (int j = 0; j < vec_size; j++) {
      v.val[j] = from[j];
    }
    to_[index] = v;
  }
};

template <
    int item_work_size,
    typename data_t,
    typename inp_calc_t,
    typename out_calc_t,
    int num_outputs>
struct multi_outputs_unroll {
  data_t data;
  int remaining;
  inp_calc_t input_offset_calculator;
  out_calc_t output_offset_calculator;
  LoadWithoutCast loader;
  StoreWithoutCast storer;
  int item_idx;
  int group_idx;
  int num_items_per_group;
  int group_work_size;

  multi_outputs_unroll(
      data_t data,
      int remaining,
      inp_calc_t ic,
      out_calc_t oc,
      int item_idx,
      int group_idx,
      int num_items_per_group)
      : data(data),
        remaining(remaining),
        input_offset_calculator(ic),
        output_offset_calculator(oc),
        item_idx(item_idx),
        group_idx(group_idx),
        num_items_per_group(num_items_per_group),
        group_work_size(item_work_size * num_items_per_group) {}

  inline bool check_inbounds(int item_work_elem) const {
    return (item_idx + item_work_elem * num_items_per_group < remaining);
  }

  template <typename args_t>
  inline void load(args_t* args) {
    constexpr int arity = std::tuple_size<args_t>::value;
    int item_idx_ = item_idx;
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      if (item_idx_ >= remaining) {
        return;
      }
      int linear_idx = item_idx_ + group_work_size * group_idx;
      auto offset = input_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::unroll_load_helper, arity>::with_args(
          *this, args, offset, loader, i, num_outputs);
      item_idx_ += num_items_per_group;
    }
  }

  template <typename return_t>
  inline void store(return_t* from) {
    int item_idx_ = item_idx;
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      if (item_idx_ >= this->remaining) {
        return;
      }
      int linear_idx = item_idx_ + group_work_size * group_idx;
      auto offsets = this->output_offset_calculator.get(linear_idx);
      detail::static_unroll<detail::multi_outputs_store_helper, num_outputs>::
          with_args(this->data, offsets, from[i]);
      item_idx_ += num_items_per_group;
    }
  }
};

} // namespace policies

// Query vector size for specific `scalar_t` on current device.
// Enlarge payloads on each work item with a definite data type
// specific vector size preference queried from SYCL runtime to feed
// memory bandwidth well on the specified hardward platform.
// In `Vectorized` policy, using definite vector size for some data type could
// get enough payloads without outer loop. Outer loop may bring additional
// instructions and potential registers usage.
template <typename scalar_t>
inline int can_vectorize_up_to(char* pointer) {
  int elem_size = sizeof(scalar_t);
  at::DeviceIndex dev_id = 0; // xpu::current_device();
  // int preferred_width = preferred_vector_width(dev_id, elem_size);
  int preferred_width = 4;
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec2_alignment =
      std::alignment_of<aligned_vector<scalar_t, 2>>::value;
  constexpr int vec4_alignment =
      std::alignment_of<aligned_vector<scalar_t, 4>>::value;
  constexpr int vec8_alignment =
      std::alignment_of<aligned_vector<scalar_t, 8>>::value;
  constexpr int vec16_alignment =
      std::alignment_of<aligned_vector<scalar_t, 16>>::value;
  if (address % vec16_alignment == 0) {
    return std::min<int>(preferred_width, 16);
  } else if (address % vec8_alignment == 0) {
    return std::min<int>(preferred_width, 8);
  } else if (address % vec4_alignment == 0) {
    return std::min<int>(preferred_width, 4);
  } else if (address % vec2_alignment == 0) {
    return std::min<int>(preferred_width, 2);
  }
  return 1;
}

template <int i>
struct can_vectorize_up_to_helper {
  template <typename array_t, typename traits>
  static void apply(int& result, array_t pointers, traits _) {
    using arg_t = typename traits::template arg<i>::type;
    result = std::min<int>(
        result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
  }
};

template <typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = can_vectorize_up_to<return_t>(pointers[0]);
  detail::static_unroll<can_vectorize_up_to_helper, arity>::with_args(
      result, pointers, traits());
  return result;
}

}}} // namespace at::native::memory
