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
#include <ATen/native/cpu/ReduceUtils.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <c10/util/irange.h>
#ifdef USE_FBGEMM
#include <fbgemm/Utils.h>
#endif
#include <ATen/OpMathType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif
namespace at::native {

namespace {

// Implement as functors since lambdas don't get optimized.
class ReduceMultiply {
public:
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data *= opmath_t(*src_data);
  }

  constexpr void operator() (bool * self_data, bool * src_data) const {
    *self_data = *self_data && *src_data;
  }
};
static ReduceMultiply reduce_multiply;

class ReduceAdd {
public:
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data += opmath_t(*src_data);
  }
};
static ReduceAdd reduce_add;

class ReduceMean {
public:
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data += opmath_t(*src_data);
  }
};
static ReduceMean reduce_mean;

class ReduceMaximum {
public:
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data = at::_isnan<scalar_t>(*src_data) ? opmath_t(*src_data) : std::max(*self_data, opmath_t(*src_data));
  }
};
static ReduceMaximum reduce_maximum;

class ReduceMinimum {
public:
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data = at::_isnan<scalar_t>(*src_data) ? opmath_t(*src_data) : std::min(*self_data, opmath_t(*src_data));
  }
};
static ReduceMinimum reduce_minimum;

class TensorAssign {
public:
  template <typename scalar_t>
  constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
    using opmath_t = at::opmath_type<scalar_t>;
    *self_data = opmath_t(*src_data);
  }
};
static TensorAssign tensor_assign;

template <bool is_scatter_like = true>
struct _cpu_scatter_gather_dim_loop {
  template <typename scalar_t, typename func_t>
  void operator()(
    at::opmath_type<scalar_t>* self_data, int64_t self_dim_stride,
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
    at::opmath_type<scalar_t>* self_data, int64_t self_dim_stride,
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

inline void create_acc_buffer(Tensor& buffer, const Tensor& self, bool need_acc) {
  if (need_acc) {
    auto acc_type = at::toOpMathType(self.scalar_type());
    buffer = at::empty(self.sizes(), self.options().dtype(acc_type));
    buffer.copy_(self);
  } else {
    buffer = self;
  }
}

template <bool is_scatter_like = true>
struct cpu_scatter_gather_base_kernel {
  template <typename func_t>
  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Scalar& value,
    const std::string& method_name, func_t& kernel_func) {

    Tensor buffer;
    bool need_acc = isReducedFloatingType(self.scalar_type());
    create_acc_buffer(buffer, self, need_acc);

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
      .add_output(buffer)
      .add_const_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    auto self_dim_size = ensure_nonempty_size(buffer, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto index_upper_bound = self_dim_size;

    // since the index dimension is squashed, need to alter the grain size according
    // to keep equal granularity in parallelism.
    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, self.scalar_type(),
      "scatter_gather_scalar_cpu", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 1;
        using opmath_t = at::opmath_type<scalar_t>;
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== buffer.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              loop_func.template operator()<scalar_t, func_t>(
                (opmath_t*)self_data_bytes, self_dim_stride,
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
                kernel_func((opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride, &temp);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop, grain_size);
      }
    );
    if (need_acc) {
      self.copy_(buffer);
    }
  }

  template <typename func_t>
  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, func_t& kernel_func) {

    Tensor buffer;
    bool need_acc = isReducedFloatingType(self.scalar_type());
    create_acc_buffer(buffer, self, need_acc);

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(buffer)
      .add_const_input(src)
      .add_const_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    auto self_dim_size = ensure_nonempty_size(buffer, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        using opmath_t = at::opmath_type<scalar_t>;
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== buffer.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              loop_func.template operator()<scalar_t, func_t>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
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
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
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
    if (need_acc) {
      self.copy_(buffer);
    }
  }

  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMean& kernel_func) {

    Tensor buffer;
    bool need_acc = isReducedFloatingType(self.scalar_type());
    create_acc_buffer(buffer, self, need_acc);

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(buffer)
      .add_const_input(src)
      .add_const_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    auto self_dim_size = ensure_nonempty_size(buffer, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu_reduce_mean", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        using opmath_t = at::opmath_type<scalar_t>;
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== buffer.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              loop_func.template operator()<scalar_t, ReduceMean>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
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
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
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
    if (need_acc) {
      self.copy_(buffer);
    }
  }

  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMaximum& kernel_func) {
    Tensor buffer;
    bool need_acc = isReducedFloatingType(self.scalar_type());
    create_acc_buffer(buffer, self, need_acc);

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(buffer)
      .add_const_input(src)
      .add_const_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    auto self_dim_size = ensure_nonempty_size(buffer, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu_reduce_amax", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        using opmath_t = at::opmath_type<scalar_t>;
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== buffer.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              loop_func.template operator()<scalar_t, ReduceMaximum>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
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
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
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
    if (need_acc) {
      self.copy_(buffer);
    }
  }

  void operator()(const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src,
    const std::string& method_name, ReduceMinimum& kernel_func) {

    Tensor buffer;
    bool need_acc = isReducedFloatingType(self.scalar_type());
    create_acc_buffer(buffer, self, need_acc);

    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      // NOLINTNEXTLINE(bugprone-argument-comment)
      .declare_static_shape(index.sizes(), /*squash_dim=*/dim)
      .add_output(buffer)
      .add_const_input(src)
      .add_const_input(index)
      .build();

    auto self_dim_stride = ensure_nonempty_stride(buffer, dim);
    auto self_dim_size = ensure_nonempty_size(buffer, dim);

    auto index_dim_stride = ensure_nonempty_stride(index, dim);
    auto index_dim_size = ensure_nonempty_size(index, dim);

    auto src_dim_stride = ensure_nonempty_stride(src, dim);
    auto src_dim_size = ensure_nonempty_size(src, dim);

    auto index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;

    int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / index_dim_size);

    AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool, ScalarType::Half, ScalarType::BFloat16, iter.dtype(1),
      "scatter_gather_tensor_cpu_reduce_amin", [&] {
        constexpr auto SELF_ITER_STRIDE_IDX = 0;
        constexpr auto INDEX_ITER_STRIDE_IDX = 2;
        constexpr auto SRC_ITER_STRIDE_IDX = 1;
        using opmath_t = at::opmath_type<scalar_t>;
        _cpu_scatter_gather_dim_loop<is_scatter_like> loop_func;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
          auto* self_data_bytes = data[SELF_ITER_STRIDE_IDX];
          auto* index_data_bytes = data[INDEX_ITER_STRIDE_IDX];
          auto* src_data_bytes = data[SRC_ITER_STRIDE_IDX];
          // we change the order of TensorIterator-dim loop
          // vs dim-TensorIterator loop order depending on
          // whether dim is the last dimension
          if (dim== buffer.dim() - 1) {
            for (const auto nelem C10_UNUSED : c10::irange(n)) {
              // dim loop is a separate code block
              // for better performance
              loop_func.template operator()<scalar_t, ReduceMinimum>(
                 (opmath_t*)self_data_bytes, self_dim_stride,
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
                  (opmath_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
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
    if (need_acc) {
      self.copy_(buffer);
    }
  }
};

#ifndef USE_FBGEMM
namespace fbgemm {

template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(
    K* const inp_key_buf,
    V* const inp_value_buf,
    K* const tmp_key_buf,
    V* const tmp_value_buf,
    const int64_t elements_count,
    const int64_t max_value) {
  TORCH_INTERNAL_ASSERT(false, "radix_sort_parallel: ATen not compiled with FBGEMM support");
  std::make_pair(nullptr, nullptr);
}

}
#endif

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

template <typename scalar_t, ReductionType reduce>
void cpu_scatter_reduce_expanded_index(const Tensor& self, const Tensor& index, const Tensor& src, bool include_self) {
  const int64_t* index_data = index.const_data_ptr<int64_t>();
  scalar_t* self_data = self.data_ptr<scalar_t>();
  const scalar_t* src_data = src.const_data_ptr<scalar_t>();

  const int64_t M = ensure_nonempty_size(self, 0);
  const int64_t nnz = ensure_nonempty_size(index, 0);
  const int64_t K = index.numel() / nnz;

  const int64_t index_upper_bound = M;

  auto keys = std::make_unique<int64_t[]>(nnz);
  auto values = std::make_unique<int64_t[]>(nnz);
  auto keys_tmp = std::make_unique<int64_t[]>(nnz);
  auto values_tmp = std::make_unique<int64_t[]>(nnz);
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
  std::tie(sorted_col_index_keys, sorted_col_index_values) = fbgemm::radix_sort_parallel(
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
  auto row_index_tmp = std::make_unique<int64_t[]>(num_nonzero_rows);
  auto row_index_offset_tmp = std::make_unique<int64_t[]>(num_nonzero_rows + 1);
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

  using opmath_t = at::opmath_type<scalar_t>;
  Tensor buffer;
  opmath_t* buffer_data = nullptr;
  static constexpr bool need_acc = is_reduced_floating_point_v<scalar_t>;
  if constexpr (need_acc) {
    auto acc_type = at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true);
    buffer = at::zeros({num_threads, K}, self.options().dtype(acc_type));
    buffer_data = buffer.data_ptr<opmath_t>();
  }

  // TODO: do blocking on col dimension to reduce WR bandwidth
  at::parallel_for(0, num_nonzero_rows, 1, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();
    TORCH_CHECK(tid < num_threads,
                "expect thread id smaller than ", num_threads, ", got thread id ", tid);
    opmath_t* buffer_ptr = nullptr;

    for (const auto m : c10::irange(begin, end)) {
      int64_t row = row_index[m];
      int64_t off_start = row_index_offset[m];
      int64_t off_end = row_index_offset[m + 1];
      scalar_t* self_ptr = self_data + row * K;
      if constexpr (need_acc) {
        buffer_ptr = buffer_data + tid * K;
      } else {
        buffer_ptr = reinterpret_cast<opmath_t*>(self_ptr);
      }

      // step 1: reinit rows in `self` if needed
      _init<scalar_t, reduce>(self_ptr, buffer_ptr, K, include_self);

      // step 2: reduce
      for (const auto n : c10::irange(off_start, off_end)) {
        int64_t col = sorted_col_index_values[n];
        update<scalar_t, reduce>(buffer_ptr, src_data + col * K, K);
      }
      if constexpr (need_acc) {
        vec::convert(buffer_ptr, self_ptr, K);
      }

      // step 3: finalize
      int64_t count = include_self ? 1 : 0;
      count += off_end - off_start;
      write<scalar_t, reduce>(self_ptr, count, K);
    }
  });
}

template <typename scalar_t>
void cpu_gather_expanded_index_kernel(const Tensor& result, const Tensor& index, const Tensor& self) {
  const int64_t* index_data = index.const_data_ptr<int64_t>();
  scalar_t* result_data = result.data_ptr<scalar_t>();
  const scalar_t* self_data = self.const_data_ptr<scalar_t>();

  const int64_t M = ensure_nonempty_size(result, 0);
  const int64_t N = ensure_nonempty_size(self, 0);
  const int64_t K = index.numel() / M;

  const int64_t index_upper_bound = N;

  using Vec = vec::Vectorized<scalar_t>;
  int64_t grain_size = std::max((int64_t) 1, at::internal::GRAIN_SIZE / K);
  at::parallel_for(0, M, grain_size, [&](int64_t begin, int64_t end) {
    for (const auto m : c10::irange(begin, end)) {
      scalar_t* result_ptr = result_data + m * K;
      int64_t index = index_data[m];
      TORCH_CHECK(index >= 0 && index < index_upper_bound,
                  "index ", index,
                  " is out of bounds for dimension ", 0,
                  " with size ", index_upper_bound);
      const scalar_t* self_ptr = self_data + index * K;
      int64_t d = 0;
      for (; d < K - (K % Vec::size()); d += Vec::size()) {
        Vec out_vec = Vec::loadu(self_ptr + d);
        out_vec.store(result_ptr + d);
      }
      #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
      # pragma unroll
      #endif
      for (; d < K; d++) {
        result_ptr[d] = self_ptr[d];
      }
    }
  });
}

void scatter_add_expanded_index_kernel(const Tensor& self, const Tensor& index, const Tensor& src) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "scatter_add_expanded_index", [&] {
      cpu_scatter_reduce_expanded_index<scalar_t, ReductionType::SUM>(self, index, src, /*include_self*/true);
  });
}

void scatter_reduce_expanded_index_kernel(
    const Tensor& self, const Tensor& index, const Tensor& src,
    const ReductionType& reduction, bool include_self) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
    ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "scatter_reduce_expanded_index", [&] {
    AT_DISPATCH_REDUCTION_TYPES(reduction, [&]() {
      cpu_scatter_reduce_expanded_index<scalar_t, reduce>(self, index, src, include_self);
    });
  });
}

void gather_expanded_index_kernel(const Tensor& result, const Tensor& self, const Tensor& index) {
  AT_DISPATCH_FLOATING_TYPES_AND(
    ScalarType::BFloat16, self.scalar_type(), "gather_expanded_index", [&] {
      cpu_gather_expanded_index_kernel<scalar_t>(result, index, self);
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

void scatter_add_cpu_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  cpu_scatter_gather_base_kernel<>()(
    self, dim, index, src,
    "scatter_add_", reduce_add);
}

void scatter_reduce_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                               const Tensor& src, const ReductionType& reduce) {
  switch (reduce) {
  case ReductionType::SUM :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_add_", reduce_add);
    break;
  case ReductionType::PROD :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_multiply_", reduce_multiply);
    break;
  default :
    break;
  }
}

void scatter_reduce_two_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                                   const Tensor& src, const ReductionType& reduce) {
  switch (reduce) {
  case ReductionType::SUM :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_sum_", reduce_add);
    break;
  case ReductionType::PROD :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_prod_", reduce_multiply);
    break;
  case ReductionType::MAX :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_amax_", reduce_maximum);
    break;
  case ReductionType::MIN :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_amin_", reduce_minimum);
    break;
  case ReductionType::MEAN :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, src,
                                       "scatter_reduce_mean_", reduce_mean);
    break;
  }
}

void scatter_scalar_reduce_cpu_kernel(const Tensor& self, const int64_t dim, const Tensor& index,
                                      const Scalar& value, const ReductionType& reduce) {
  switch (reduce) {
  case ReductionType::SUM :
    cpu_scatter_gather_base_kernel<>()(self, dim, index, value,
                                       "scatter_scalar_reduce_add_", reduce_add);
    break;
  case ReductionType::PROD :
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

// fast paths for GNN usage
REGISTER_DISPATCH(scatter_add_expanded_index_stub, &scatter_add_expanded_index_kernel);
REGISTER_DISPATCH(scatter_reduce_expanded_index_stub, &scatter_reduce_expanded_index_kernel);
REGISTER_DISPATCH(gather_expanded_index_stub, &gather_expanded_index_kernel);

} // namespace at::native
