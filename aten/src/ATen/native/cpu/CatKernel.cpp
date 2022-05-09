#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/utils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

struct InputMeta {
  void* data_ptr;
  int64_t slice_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner_size)
    : data_ptr(t.data_ptr())
    , slice_size(t.sizes()[dim] * inner_size) {}

  InputMeta(void* data_ptr, int64_t slice_size)
    : data_ptr(data_ptr)
    , slice_size(slice_size) {}
};

template <typename scalar_t>
static inline void copy_stub(scalar_t* result, scalar_t* self, int64_t size) {
  using Vec = vec::Vectorized<scalar_t>;
  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    Vec data_vec = Vec::loadu(self + d);
    data_vec.store(result + d);
  }
  #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
  # pragma unroll
  #endif
  for (; d < size; ++d) {
    result[d] = self[d];
  }
}

template <typename scalar_t>
void cat_contig_firstdim_impl(
    const Tensor& result,
    const MaterializedITensorListRef& tensors,
    int64_t dim,
    int64_t dim_size,
    int64_t inner_size,
    bool all_same_sizes_and_stride) {
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = static_cast<int64_t>(tensors.size());

  if (all_same_sizes_and_stride) {
    // input tensors have the same shapes and strides
    if (ninputs < 64) {
      std::vector<InputMeta> inputs;
      inputs.reserve(ninputs);
      for (const Tensor& tensor : tensors) {
        inputs.emplace_back(tensor, dim, inner_size);
      }
      int64_t input_dim_size = dim_size / ninputs;

      // short input tensor list: parallel on dim_size (dim_size == ninputs * input_dim_size).
      //
      // note that prallel on ninputs may not have enough parallelism (e.g. inputs == 2), also
      // parallel on input_dim_size would trigger multiple omp sessions, which has additional overhead.
      //
      at::parallel_for(0, dim_size, internal::GRAIN_SIZE / inner_size, [&](int64_t begin, int64_t end) {
        int64_t n{0}, i{0};
        data_index_init(begin, n, ninputs, i, input_dim_size);
        for (const auto ii : c10::irange(begin, end)) {
          scalar_t* result_ptr = result_data + ii * inner_size;
          scalar_t* input_ptr = (scalar_t*)(inputs[n].data_ptr) + i * inner_size;
          copy_stub(result_ptr, input_ptr, inner_size);
          data_index_step(n, ninputs, i, input_dim_size);
        }
      });
    } else {
      // long input tensor list: directly parallel on ninputs
      //
      // no need to create InputMeta array. In case the tensor list is very long and
      // each tensor is small, creating InputMeta array itself would take too much time.
      //
      int64_t result_slice_size = dim_size * inner_size;
      int64_t input_slice_size = result_slice_size / ninputs;
      at::parallel_for(0, ninputs, internal::GRAIN_SIZE / input_slice_size, [&](int64_t begin, int64_t end) {
        for (const auto n : c10::irange(begin, end)) {
          scalar_t* result_ptr = result_data + n * input_slice_size;
          const Tensor& input = tensors[n];
          scalar_t* input_data = input.data_ptr<scalar_t>();
          copy_stub(result_ptr, input_data, input_slice_size);
        }
      });
    }
  } else {
    // input tensors have different shapes and strides
    bool use_serial_kernel = dim_size * inner_size < at::internal::GRAIN_SIZE || at::get_num_threads() == 1;
    if (use_serial_kernel) {
      // for sequential case, just copy input tensor one by one
      scalar_t* result_ptr = result_data;
      for (const Tensor& tensor : tensors) {
        scalar_t* input_data = tensor.data_ptr<scalar_t>();
        int64_t input_slice_size = tensor.numel();
        copy_stub(result_ptr, input_data, input_slice_size);
        result_ptr += input_slice_size;
      }
    } else if (ninputs < 64) {
      // for parallel case, for short input list, calculate input offset first
      std::vector<InputMeta> inputs;
      inputs.reserve(dim_size);
      for (const Tensor& tensor : tensors) {
        scalar_t* input_data = tensor.data_ptr<scalar_t>();
        int64_t input_dim_size = tensor.sizes()[dim];
        for (const auto i : c10::irange(input_dim_size)) {
          scalar_t* input_ptr = input_data + i * inner_size;
          inputs.emplace_back((void*)input_ptr, inner_size);
        }
      }
      at::parallel_for(0, dim_size, internal::GRAIN_SIZE / inner_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          scalar_t* result_ptr = result_data + i * inner_size;
          scalar_t* input_ptr = (scalar_t*)(inputs[i].data_ptr);
          copy_stub(result_ptr, input_ptr, inner_size);
        }
      });
    } else {
      // for parallel case, for long input list, parallel on ninputs.
      // prefix sum the offsets.
      std::vector<std::pair<int64_t, int64_t>> inputs_offset;
      inputs_offset.reserve(ninputs);
      int64_t sum = 0;
      for (const Tensor& tensor : tensors) {
        int64_t input_slice_size = tensor.numel();
        inputs_offset.emplace_back(sum, input_slice_size);
        sum += input_slice_size;
      }
      int64_t average_input_slice_size = dim_size * inner_size / ninputs;
      at::parallel_for(0, ninputs, internal::GRAIN_SIZE / average_input_slice_size, [&](int64_t begin, int64_t end) {
        for (const auto n : c10::irange(begin, end)) {
          int64_t input_offset = std::get<0>(inputs_offset[n]);
          int64_t input_slice_size = std::get<1>(inputs_offset[n]);
          scalar_t* result_ptr = result_data + input_offset;
          const Tensor& input = tensors[n];
          scalar_t* input_data = input.data_ptr<scalar_t>();
          copy_stub(result_ptr, input_data, input_slice_size);
        }
      });
    }
  }
}

template <typename scalar_t>
void cat_interleave2_impl(scalar_t* result, scalar_t* input0, scalar_t* input1, int64_t outer_size) {
  using Vec = vec::Vectorized<scalar_t>;
  at::parallel_for(0, outer_size, internal::GRAIN_SIZE / 2, [&](int64_t begin, int64_t end) {
    int64_t d = begin;
    for (; d <= end - Vec::size(); d += Vec::size()) {
      // interleave2 pattern:
      //   data0_vec = {a0, a1, a2, a3, a4, a5, a6, a7}
      //   data1_vec = {b0, b1, b2, b3, b4, b5, b6, b7}
      //   out0_vec  = {a0, b0, a1, b1, a2, b2, a3, b3}
      //   out1_vec  = {a4, b4, a5, b5, a6, b6, a7, b7}
      //
      Vec data0_vec = Vec::loadu(input0 + d);
      Vec data1_vec = Vec::loadu(input1 + d);
      Vec out0_vec, out1_vec;
      std::tie(out0_vec, out1_vec) = vec::interleave2(data0_vec, data1_vec);
      out0_vec.store(result + d * 2);
      out1_vec.store(result + d * 2 + Vec::size());
    }
    for (; d < end; ++d) {
      result[d * 2 + 0] = input0[d];
      result[d * 2 + 1] = input1[d];
    }
  });
}

template <typename scalar_t>
void cat_interleave4_impl(scalar_t* result, scalar_t* input0, scalar_t* input1, int64_t outer_size) {
  at::parallel_for(0, outer_size, internal::GRAIN_SIZE / 4, [&](int64_t begin, int64_t end) {
    for (const auto d : c10::irange(begin, end)) {
      result[d * 4] = input0[d * 2];
      result[d * 4 + 1] = input0[d * 2 + 1];
      result[d * 4 + 2] = input1[d * 2];
      result[d * 4 + 3] = input1[d * 2 + 1];
    }
  });
}

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER) && !defined(C10_MOBILE)
template <>
void cat_interleave4_impl<float>(float* result, float* input0, float* input1, int64_t outer_size) {
  using Vec = vec::Vectorized<float>;
  constexpr int64_t K = Vec::size() / 2;
  at::parallel_for(0, outer_size, internal::GRAIN_SIZE / 4, [&](int64_t begin, int64_t end) {
    int64_t d = begin;
    for (; d <= end - K; d += K) {
      // interleave4 pattern:
      //   data0_vec = {a0, a1, a2, a3, a4, a5, a6, a7}
      //   data1_vec = {b0, b1, b2, b3, b4, b5, b6, b7}
      //   out0_vec  = {a0, a1, b0, b1, a2, a3, b2, b3}
      //   out1_vec  = {a4, a5, b4, b5, a6, a7, b6, b7}
      //
      Vec data0_vec = Vec::loadu(input0 + d * 2);
      Vec data1_vec = Vec::loadu(input1 + d * 2);
      auto t0 = _mm256_permute2f128_ps(data0_vec, data1_vec, 0b0100000);
      auto t1 = _mm256_permute2f128_ps(data0_vec, data1_vec, 0b0110001);
      const __m256i group_ctrl = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
      Vec out0_vec = _mm256_permutevar8x32_ps(t0, group_ctrl);
      Vec out1_vec = _mm256_permutevar8x32_ps(t1, group_ctrl);
      out0_vec.store(result + d * 4);
      out1_vec.store(result + d * 4 + Vec::size());
    }
    for (; d < end; ++d) {
      result[d * 4] = input0[d * 2];
      result[d * 4 + 1] = input0[d * 2 + 1];
      result[d * 4 + 2] = input1[d * 2];
      result[d * 4 + 3] = input1[d * 2 + 1];
    }
  });
}
#endif

template <typename scalar_t>
void cat_contig_non_firstdim_impl(
    const Tensor& result,
    const MaterializedITensorListRef& tensors,
    int64_t dim,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    bool all_same_sizes_and_stride) {
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = static_cast<int64_t>(tensors.size());
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (const Tensor& tensor : tensors) {
    inputs.emplace_back(tensor, dim, inner_size);
  }

  bool use_interleave_copy = result.scalar_type() == kFloat &&
      all_same_sizes_and_stride && ninputs == 2 && inner_size == 1;
  if (use_interleave_copy && dim_size == 2) {
    cat_interleave2_impl<scalar_t>(result_data, (scalar_t*)(inputs[0].data_ptr), (scalar_t*)(inputs[1].data_ptr), outer_size);
  } else if (use_interleave_copy && dim_size == 4) {
    cat_interleave4_impl<scalar_t>(result_data, (scalar_t*)(inputs[0].data_ptr), (scalar_t*)(inputs[1].data_ptr), outer_size);
  } else {
    int64_t result_slice_size = dim_size * inner_size;
    int64_t grain_size = internal::GRAIN_SIZE / result_slice_size;
    at::parallel_for(0, outer_size, grain_size, [&](int64_t begin, int64_t end) {
      scalar_t* result_ptr = result_data + begin * result_slice_size;
      for (const auto i : c10::irange(begin, end)) {
        for (const auto j : c10::irange(ninputs)) {
          int64_t input_slice_size = inputs[j].slice_size;
          scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * input_slice_size;
          copy_stub(result_ptr, input_ptr, input_slice_size);
          result_ptr += input_slice_size;
        }
      }
    });
  }
}

template <typename scalar_t>
void cpu_cat_contig_dispatch(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim, bool all_same_sizes_and_stride) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim < result.dim(), "dim out of range in cat_serial_kernel_impl");

  // normalize self and result shape as:
  //   input: [outer_size, input_dim_size, inner_size]
  //   result: [outer_size, dim_size, inner_size]
  int64_t inner_size = result.strides()[dim];
  int64_t dim_size = result.sizes()[dim];
  int64_t outer_size = result.numel() / (dim_size * inner_size);

  // Note on cat implementation choosen:
  //
  // In order to minimize overhead of meta info creation, pass down `all_same_sizes_and_stride`
  // to the kernel. `True` indicates all the input tensors all have the same shape and stride.
  //
  // All kernels have a single omp loop (the non-contiguous path may have mutiple omp loops).
  // All kernels trim grain_size in the parallel loop w.r.t. `at::internal::GRAIN_SIZE`.
  //
  // 1. `cat_contig_firstdim_impl`: used when outer_size == 1 (dim is the first dimension)
  //   a. all_same_sizes_and_stride is true:
  //     for short input tensor list (e.g. ninputs == 2), parallel on dim_size (ninputs * input_dim_size);
  //     for long input tensor list (e.g. ninputs = 1000), directly parallel on ninputs (no need to create
  //   input meta info array).
  //
  //   b. all_same_sizes_and stride is false:
  //     for single thread case, directly copy the input tensor list one by one (no need to create input meta
  //   info array).
  //     for multi thread case, pre-calculate input offset in order to evenly parallel on dim_size.
  //
  //  2. `cat_contig_non_firstdim_impl`: used when outer_size != 1 (dim is not the first dimension)
  //   a. specialize cases when dim is the last dimension and input.size(dim) is 1 or 2 with manual vectorization.
  //
  //   b. for generic cases, simply parallel on outer_size and copy the input slice one by one.
  //
  if (outer_size == 1) {
    cat_contig_firstdim_impl<scalar_t>(result, tensors, dim, dim_size, inner_size, all_same_sizes_and_stride);
  } else {
    cat_contig_non_firstdim_impl<scalar_t>(result, tensors, dim, outer_size, dim_size, inner_size, all_same_sizes_and_stride);
  }
}

void cat_contig_kernel(const Tensor& result, const MaterializedITensorListRef& tensors, int64_t dim, bool all_same_sizes_and_stride) {
  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, result.scalar_type(), "cat_contig_kernel", [&]() {
    cpu_cat_contig_dispatch<scalar_t>(result, tensors, dim, all_same_sizes_and_stride);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cat_contig_stub, &cat_contig_kernel);

}} // at::native
