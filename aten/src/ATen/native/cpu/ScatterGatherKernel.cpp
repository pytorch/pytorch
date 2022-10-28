#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/radix_sort.h>
#include <c10/util/irange.h>

namespace at { namespace native {

namespace {

// Implement as functors since lambdas don't get optimized.
class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data *= *src_data;
  }

  constexpr void operator() (bool * self_data, bool * src_data) const {
    *self_data = *self_data && *src_data;
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data += *src_data;
  }
};
static ReduceAdd reduce_add;

class ReduceMean {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data += *src_data;
  }
};
static ReduceMean reduce_mean;

class ReduceMaximum {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data = at::_isnan<scalar_t>(*src_data) ? *src_data : std::max(*self_data, *src_data);
  }
};
static ReduceMaximum reduce_maximum;

class ReduceMinimum {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data = at::_isnan<scalar_t>(*src_data) ? *src_data : std::min(*self_data, *src_data);
  }
};
static ReduceMinimum reduce_minimum;

class TensorAssign {
public:
  template <typename scalar_t>
  constexpr void operator() (scalar_t * self_data, scalar_t * src_data) const {
    *self_data = *src_data;
  }
};
static TensorAssign tensor_assign;

template <bool is_scatter_like = true>
struct _cpu_scatter_gather_dim_loop {
  template <typename scalar_t, typename func_t>
  void operator()(
    scalar_t* self_data, int64_t self_dim_stride,
    int64_t* index_data, int64_t index_dim_stride,
    scalar_t* src_data, int64_t src_dim_stride,
    int64_t dim, int64_t index_dim_size,
    int64_t index_upper_bound,
    func_t& f
  ) {

    for (const auto i : c10::irange(index_dim_size)) {
      int64_t idx_dim = index_data[i * index_dim_stride];
      // we are not putting idx_dim in the error message because it disables
      // loop optimization in clang-7
      TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
        "index ", index_data[i * index_dim_stride],
        " is out of bounds for dimension ", dim,
        " with size ", index_upper_bound
      );

      f(
        self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
        src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
      );
    }
  }

  template <typename scalar_t, typename func_t>
  void operator()(
    scalar_t* self_data, int64_t self_dim_stride,
    int64_t* index_data, int64_t index_dim_stride,
    Scalar value,
    int64_t dim, int64_t index_dim_size,
    int64_t index_upper_bound,
    func_t& f
  ) {

    for (const auto i : c10::irange(index_dim_size)) {
      int64_t idx_dim = index_data[i * index_dim_stride];
      // we are not putting idx_dim in the error message because it disables
      // loop optimization in clang-7
      TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
        "index ", index_data[i * index_dim_stride],
        " is out of bounds for dimension ", dim,
        " with size ", index_upper_bound
      );
      auto temp = value.to<scalar_t>();
      f(
        self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp
      );
    }
  }
};


template <bool is_scatter_like = true>
struct cpu_scatter_gather_base_kernel {
  template <typename func_t>
  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Scalar& value,
    const std::string& method_name, func_t& kernel_func) {

    auto index_sizes = ensure_nonempty_vec(index.sizes().vec());
    auto index_strides = ensure_nonempty_vec(index.strides().vec());

    // `dim` is traversed in the kernel,
    // that is why index.stride(dim) = 0 and index.size(dim) = 1.
    // Also, index.size(dim) = 1 makes sure that TensorIterator.DimCounter
    // has the following form : (i_1,..., i_{dim-1}, 0, i_{dim+1},...,i_n).
    index_sizes[dim] = 1;
    index_strides[dim] = 0;

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(self)
      .add_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto index_upper_bound = self_dim_size;

    // since the index dimension is squashed, need to alter the grain size according
    // to keep equal granularity in parallelism.
    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
      "scatter_gather_scalar_cpu", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 1;

        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== self.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                (scalar_t*)self_data_bytes, self_dim_stride,
                (int64_t*)index_data_bytes, index_dim_stride,
                value, dim, index_dim_size, index_upper_bound,
                kernel_func);

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                auto temp = value.to<scalar_t>();
                kernel_func((scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop, grain_size);
      }
    );
  }

  template <typename func_t>
  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, func_t& kernel_func) {

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(self)
      .add_input(src)
      .add_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
      "scatter_gather_tensor_cpu", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== self.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                 (scalar_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop, grain_size);
      }
    );
  }

  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMean& kernel_func) {

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(self)
      .add_input(src)
      .add_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
      "scatter_gather_tensor_cpu_reduce_mean", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== self.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                 (scalar_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop, grain_size);
      }
    );
  }

  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMaximum& kernel_func) {

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(self)
      .add_input(src)
      .add_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
      "scatter_gather_tensor_cpu_reduce_amax", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== self.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                 (scalar_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop, grain_size);
      }
    );
  }

  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMinimum& kernel_func) {

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(self)
      .add_input(src)
      .add_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(self, dim);
    auto self_dim_size = ensure_nonempty_size(self, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(),
      "scatter_gather_tensor_cpu_reduce_amin", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== self.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                 (scalar_t*)self_data_bytes, self_dim_stride,
                 (int64_t*)index_data_bytes, index_dim_stride,
                 (scalar_t*)src_data_bytes, src_dim_stride,
                 dim, index_dim_size, index_upper_bound,
                 kernel_func
               );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;
              for (const auto nelem C10_UNUSED : c10::irange(n)) {
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop, grain_size);
      }
    );
  }
};

// Note [scatter reduce optimization]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// 1. initiative: optimize `scatter_reduce` on classic PyG use-case:
//   `scatter_reduce` is extensively used on 'message passing' when
//   aggregating info.
//
//   Typically, `self` will 2D tensor and `index` is a 1D extended/broadcasted
//   tensor, which means that the aggregation is on rowwise and we can vectorize
//   on the inner dimensions.
//
// 2. implementation: map `scatter_reduce` to `spmm` reduce
//   in the shape of `[M, N]` * `[N, K]`, where:
//
//   M: self_dim_size
//   nnz: index_dim_size
//   K: index.numel() / index_dim_size;
//
//   step 1: convert input index to CSR format (use radix_sort to
//     solve write addr conflicts on `self` tensor)
//
//   step 2: spmm reduce, parallel on M and vectorize on K
//
template <typename scalar_t>
void cpu_scatter_add_expanded_index_kernel(const Tensor& self, const Tensor& index, const Tensor& src) {
  int64_t* index_data = index.data_ptr<int64_t>();
  scalar_t* self_data = self.data_ptr<scalar_t>();
  scalar_t* src_data = src.data_ptr<scalar_t>();

  const int64_t M = ensure_nonempty_size(self, 0);
  const int64_t nnz = ensure_nonempty_size(index, 0);
  const int64_t K = index.numel() / nnz;

  const int64_t index_upper_bound = M;

  std::unique_ptr<int64_t []> keys(new int64_t[nnz]);
  std::unique_ptr<int64_t []> values(new int64_t[nnz]);
  std::unique_ptr<int64_t []> keys_tmp(new int64_t[nnz]);
  std::unique_ptr<int64_t []> values_tmp(new int64_t[nnz]);
  at::parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t index = index_data[i];
      TORCH_CHECK(index >= 0 && index < index_upper_bound,
                "index ", index,
                " is out of bounds for dimension ", 0,
                " with size ", index_upper_bound);
      keys[i] = index;
      values[i] = i;
    }
  });

  int64_t* sorted_col_index_keys = nullptr;
  int64_t* sorted_col_index_values = nullptr;
  std::tie(sorted_col_index_keys, sorted_col_index_values) = radix_sort_parallel(
      keys.get(),
      values.get(),
      keys_tmp.get(),
      values_tmp.get(),
      nnz,
      M);

  int num_threads = at::get_num_threads();
  std::vector<int64_t> num_uniq(num_threads, 0);
  at::parallel_for(1, nnz, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    for(const auto i : c10::irange(begin, end)) {
      if (sorted_col_index_keys[i] != sorted_col_index_keys[i - 1]) {
        num_uniq[tid]++;
      }
    }
  });
  num_uniq[0]++;
  for (const auto n : c10::irange(1, num_threads)) {
    num_uniq[n] += num_uniq[n - 1];
  }

  // in case some rows are not written into, num_nonzero_rows will be smaller than M
  int64_t num_nonzero_rows = num_uniq[num_threads - 1];
  std::unique_ptr<int64_t []> row_index_tmp(new int64_t[num_nonzero_rows]);
  std::unique_ptr<int64_t []> row_index_offset_tmp(new int64_t[num_nonzero_rows + 1]);
  int64_t* row_index = row_index_tmp.get();
  int64_t* row_index_offset = row_index_offset_tmp.get();
  row_index[0] = sorted_col_index_keys[0];
  row_index_offset[0] = 0;
  row_index_offset[num_nonzero_rows] = nnz;

  at::parallel_for(1, nnz, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    int64_t* t_index = row_index + ((tid == 0) ? 1 : num_uniq[tid - 1]);
    int64_t* t_index_offset = row_index_offset + ((tid == 0) ? 1 : num_uniq[tid - 1]);
    for (const auto i : c10::irange(begin, end)) {
      if (sorted_col_index_keys[i] != sorted_col_index_keys[i - 1]) {
        *t_index = sorted_col_index_keys[i];
        *t_index_offset = i;
        t_index++;
        t_index_offset++;
      }
    }
  });

  // TODO: do blocking on col dimension to reduce WR bandwidth
  using Vec = vec::Vectorized<vec::vec_scalar_t<scalar_t>>;
  at::parallel_for(0, num_nonzero_rows, 1, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      int64_t row = row_index[m];
      int64_t off_start = row_index_offset[m];
      int64_t off_end = row_index_offset[m + 1];
      scalar_t* self_ptr = self_data + row * K;
      for (const auto n : c10::irange(off_start, off_end)) {
        int64_t col = sorted_col_index_values[n];
        scalar_t* src_ptr = src_data + col * K;
        vec::map2<scalar_t>(
            [](Vec x, Vec y) { return x + y; },
            self_ptr,
            self_ptr,
            src_ptr,
            K);
      }
    }
  });
}

void scatter_add_expanded_index(const Tensor& self, const Tensor& index, const Tensor& src) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, self.scalar_type(),
    "scatter_add_contig", [&] {
      cpu_scatter_add_expanded_index_kernel<scalar_t>(self, index, src);
  });
}

void gather_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim, const Tensor& index) {
  cpu_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
    result, dim, index, self,
    "gather_out_cpu", tensor_assign);
}

void scatter_cpu_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src, "scatter_cpu_", tensor_assign);
}

void scatter_fill_cpu_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, value, "scatter_fill_cpu_", tensor_assign);
}

inline bool can_use_expanded_index_path(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  if (!is_radix_sort_available()) { return false; }

  // skip when having empty tensor
  if (self.numel() == 0 || index.numel() == 0 || src.numel() == 0) { return false; }

  // skip when having scalar tensor
  if (self.ndimension() == 0 || index.ndimension() == 0 || src.ndimension() == 0) { return false; }

  // using `spmm` for scatter would require sorting on index,
  // this is only perf beneficial when the inner dimension, aka, `channels`
  // is big enough.
  constexpr int64_t threshold = 16;
  if (index.numel() / index.size(0) < threshold) { return false; }

  // usually the expanded index has stride on the first dimension to be 1,
  // and strides on other dims to be 0 or 1, e.g.
  //   shape [108365, 16]; strides [1, 0]
  //   shape [13264, 1, 7]; strides [1, 1, 0]
  auto index_strides = index.strides().vec();
  bool is_index_expanded = index_strides[0] == 1;
  for (const auto dim : c10::irange(1, index_strides.size())) {
    if (index_strides[dim] > 1) { is_index_expanded = false; }
  }

  // index is expanded
  return dim == 0 && is_index_expanded && src.is_contiguous() && self.is_contiguous();
}

void scatter_add_cpu_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  if (can_use_expanded_index_path(self, dim, index, src)) {
    scatter_add_expanded_index(self, index, src);
  } else {
    cpu_scatter_gather_base_kernel<>()(
      self, dim, index, src,
      "scatter_add_", reduce_add);
  }
}

void scatter_reduce_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                               const Tensor& src, const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
  case SCATTER_GATHER_OP::REDUCE_ADD :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_add_", reduce_add);
    break;
  case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_multiply_", reduce_multiply);
    break;
  default :
    break;
  }
}

void scatter_reduce_two_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                                   const Tensor& src, const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
  case SCATTER_GATHER_OP::REDUCE_ADD :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_sum_", reduce_add);
    break;
  case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_prod_", reduce_multiply);
    break;
  case SCATTER_GATHER_OP::REDUCE_MAXIMUM :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_amax_", reduce_maximum);
    break;
  case SCATTER_GATHER_OP::REDUCE_MINIMUM :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_amin_", reduce_minimum);
    break;
  case SCATTER_GATHER_OP::REDUCE_MEAN :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_mean_", reduce_mean);
    break;
  }
}

void scatter_scalar_reduce_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                                      const Scalar& value, const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
  case SCATTER_GATHER_OP::REDUCE_ADD :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                       "scatter_scalar_reduce_add_", reduce_add);
    break;
  case SCATTER_GATHER_OP::REDUCE_MULTIPLY :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                       "scatter_scalar_reduce_multiply_", reduce_multiply);
    break;
  default:
    break;
  }
}

} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_cpu_kernel);
REGISTER_DISPATCH(scatter_stub, &scatter_cpu_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_cpu_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel);
REGISTER_DISPATCH(scatter_reduce_stub, &scatter_reduce_cpu_kernel);
REGISTER_DISPATCH(scatter_scalar_reduce_stub, &scatter_scalar_reduce_cpu_kernel);
REGISTER_DISPATCH(scatter_reduce_two_stub, &scatter_reduce_two_cpu_kernel);

}} // namespace at::native
