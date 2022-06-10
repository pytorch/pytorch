#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/Parallel.h>
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


struct _cpu_scatter_large_index_dim_loop {
  template <typename scalar_t, typename func_t>
  void operator()(
    scalar_t* self_data, int64_t self_dim_stride,
    int64_t* index_data, int64_t index_dim_stride,
    int64_t* index_starting_ptr, IntArrayRef index_shape,
    IntArrayRef* index_strides,
    scalar_t* src_starting_ptr, IntArrayRef src_shape,
    IntArrayRef src_strides,
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

      int ndim = index_shape.size();
      int64_t src_offset = 0;
      int64_t absolute_index_offset = (int64_t)((int64_t*)(index_data + i * index_dim_stride) - index_starting_ptr); //  / sizeof(int64_t); // index tensor has word size = 8
      int64_t index_idx;
      for (int d = ndim - 1; d >= 0; d--) {
        index_idx = (absolute_index_offset / index_strides[d]) % index_shape[d];
        absolute_index_offset -= index_idx * index_strides[d];

        index_idx %= src_shape[d];
        src_offset += src_strides[d] * index_idx;
      }

      f(
        self_data + idx_dim * self_dim_stride,
        src_starting_ptr + src_offset // * sizeof(scalar_t); // source tensor has word size dependent on type
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
            for (const auto nelem : c10::irange(n)) {
              (void)nelem; //Suppress unused variable warning
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
              for (const auto nelem : c10::irange(n)) {
                (void)nelem; //Suppress unused variable warning
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

    auto index_shape = index.sizes();
    auto index_strides = index.strides();
    auto src_shape = src.sizes();
    auto src_strides = src.strides();
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    int ndim = index.dim();

    bool index_larger_than_src_in_scatter = false;
    if (is_scatter_like) {
      for (int i = 0; i < ndim; i++) {
        if (index_shape[i] > src_shape[i]) {
          index_larger_than_src_in_scatter = true;
          break;
        }
      }
    }

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
            for (const auto nelem : c10::irange(n)) {
              (void)nelem; //Suppress unused variable warning
              // dim loop is a separate code block
              // for better performance
              if (!index_larger_than_src_in_scatter)
                _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (scalar_t*)src_data_bytes, src_dim_stride,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );
              else
                _cpu_scatter_large_index_dim_loop()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (int64_t*)index_ptr, index_shape,
                   index_strides,
                   (scalar_t*)src_ptr, src_shape,
                   src_strides,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              if (!index_larger_than_src_in_scatter) src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;

              scalar_t* absolute_src_ptr;
              for (const auto nelem : c10::irange(n)) {
                (void)nelem; //Suppress unused variable warning
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                if (index_larger_than_src_in_scatter) {
                  int64_t src_offset = 0;
                  int64_t absolute_index_offset = (int64_t)((int64_t*)index_data - (int64_t*)index_ptr); //  / sizeof(int64_t); // index tensor has word size = 8
                  int64_t index_idx;
                  for (int d = ndim - 1; d >= 0; d--) {
                    index_idx = (absolute_index_offset / index_strides[d]) % index_shape[d];
                    absolute_index_offset -= index_idx * index_strides[d];

                    index_idx %= src_shape[d];
                    src_offset += src_strides[d] * index_idx;
                  }
                  absolute_src_ptr = (scalar_t*)src_ptr + src_offset; // * sizeof(scalar_t); // source tensor has word size dependent on type
                }

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  !index_larger_than_src_in_scatter ?
                      (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
                    : absolute_src_ptr);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                if (!index_larger_than_src_in_scatter) src_data += strides[SRC_ITER_STRIDE_IDX];
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

    auto index_shape = index.sizes();
    auto index_strides = index.strides();
    auto src_shape = src.sizes();
    auto src_strides = src.strides();
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    int ndim = index.dim();

    bool index_larger_than_src_in_scatter = false;
    if (is_scatter_like) {
      for (int i = 0; i < ndim; i++) {
        if (index_shape[i] > src_shape[i]) {
          index_larger_than_src_in_scatter = true;
          break;
        }
      }
    }

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
            for (const auto nelem : c10::irange(n)) {
              (void)nelem; //Suppress unused variable warning
              // dim loop is a separate code block
              // for better performance
              if (!index_larger_than_src_in_scatter)
                _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (scalar_t*)src_data_bytes, src_dim_stride,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );
              else
                _cpu_scatter_large_index_dim_loop()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (int64_t*)index_ptr, index_shape,
                   index_strides,
                   (scalar_t*)src_ptr, src_shape,
                   src_strides,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              if (!index_larger_than_src_in_scatter) src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;

              scalar_t* absolute_src_ptr;
              for (const auto nelem : c10::irange(n)) {
                (void)nelem; //Suppress unused variable warning
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                if (index_larger_than_src_in_scatter) {
                  int64_t src_offset = 0;
                  int64_t absolute_index_offset = (int64_t)((int64_t*)index_data - (int64_t*)index_ptr); //  / sizeof(int64_t); // index tensor has word size = 8
                  int64_t index_idx;
                  for (int d = ndim - 1; d >= 0; d--) {
                    index_idx = (absolute_index_offset / index_strides[d]) % index_shape[d];
                    absolute_index_offset -= index_idx * index_strides[d];

                    index_idx %= src_shape[d];
                    src_offset += src_strides[d] * index_idx;
                  }
                  absolute_src_ptr = (scalar_t*)src_ptr + src_offset; // * sizeof(scalar_t); // source tensor has word size dependent on type
                }

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  !index_larger_than_src_in_scatter ?
                      (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
                    : absolute_src_ptr);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                if (!index_larger_than_src_in_scatter) src_data += strides[SRC_ITER_STRIDE_IDX];
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

    auto index_shape = index.sizes();
    auto index_strides = index.strides();
    auto src_shape = src.sizes();
    auto src_strides = src.strides();
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    int ndim = index.dim();

    bool index_larger_than_src_in_scatter = false;
    if (is_scatter_like) {
      for (int i = 0; i < ndim; i++) {
        if (index_shape[i] > src_shape[i]) {
          index_larger_than_src_in_scatter = true;
          break;
        }
      }
    }

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
            for (const auto nelem : c10::irange(n)) {
              (void)nelem; //Suppress unused variable warning
              // dim loop is a separate code block
              // for better performance
              if (!index_larger_than_src_in_scatter)
                _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (scalar_t*)src_data_bytes, src_dim_stride,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );
              else
                _cpu_scatter_large_index_dim_loop()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (int64_t*)index_ptr, index_shape,
                   index_strides,
                   (scalar_t*)src_ptr, src_shape,
                   src_strides,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              if (!index_larger_than_src_in_scatter) src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;

              scalar_t* absolute_src_ptr;
              for (const auto nelem : c10::irange(n)) {
                (void)nelem; //Suppress unused variable warning
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                if (index_larger_than_src_in_scatter) {
                  int64_t src_offset = 0;
                  int64_t absolute_index_offset = (int64_t)((int64_t*)index_data - (int64_t*)index_ptr); //  / sizeof(int64_t); // index tensor has word size = 8
                  int64_t index_idx;
                  for (int d = ndim - 1; d >= 0; d--) {
                    index_idx = (absolute_index_offset / index_strides[d]) % index_shape[d];
                    absolute_index_offset -= index_idx * index_strides[d];

                    index_idx %= src_shape[d];
                    src_offset += src_strides[d] * index_idx;
                  }
                  absolute_src_ptr = (scalar_t*)src_ptr + src_offset; // * sizeof(scalar_t); // source tensor has word size dependent on type
                }

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  !index_larger_than_src_in_scatter ?
                      (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
                    : absolute_src_ptr);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                if (!index_larger_than_src_in_scatter) src_data += strides[SRC_ITER_STRIDE_IDX];
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

    auto index_shape = index.sizes();
    auto index_strides = index.strides();
    auto src_shape = src.sizes();
    auto src_strides = src.strides();
    char* src_ptr = (char*)iter.data_ptr(1);
    char* index_ptr = (char*)iter.data_ptr(2);

    int ndim = index.dim();

    bool index_larger_than_src_in_scatter = false;
    if (is_scatter_like) {
      for (int i = 0; i < ndim; i++) {
        if (index_shape[i] > src_shape[i]) {
          index_larger_than_src_in_scatter = true;
          break;
        }
      }
    }

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
            for (const auto nelem : c10::irange(n)) {
              (void)nelem; //Suppress unused variable warning
              // dim loop is a separate code block
              // for better performance
              if (!index_larger_than_src_in_scatter)
                _cpu_scatter_gather_dim_loop<is_scatter_like>()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (scalar_t*)src_data_bytes, src_dim_stride,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );
              else
                _cpu_scatter_large_index_dim_loop()(
                   (scalar_t*)self_data_bytes, self_dim_stride,
                   (int64_t*)index_data_bytes, index_dim_stride,
                   (int64_t*)index_ptr, index_shape,
                   index_strides,
                   (scalar_t*)src_ptr, src_shape,
                   src_strides,
                   dim, index_dim_size, index_upper_bound,
                   kernel_func
                 );

              self_data_bytes += strides[SELF_ITER_STRIDE_IDX];
              index_data_bytes += strides[INDEX_ITER_STRIDE_IDX];
              if (!index_larger_than_src_in_scatter) src_data_bytes += strides[SRC_ITER_STRIDE_IDX];
            }
          }
          else {
            for (const auto i : c10::irange(index_dim_size)) {
              auto* self_data = self_data_bytes;
              auto* index_data = (char*)((int64_t*)index_data_bytes + i * index_dim_stride);
              auto* src_data = src_data_bytes;

              scalar_t* absolute_src_ptr;
              for (const auto nelem : c10::irange(n)) {
                (void)nelem; //Suppress unused variable warning
                int64_t idx_dim = *(int64_t*)index_data;
                // we are not putting idx_dim in the error message because it disables
                // loop optimization in clang-7
                TORCH_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", *(int64_t*)index_data,
                            " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);

                if (index_larger_than_src_in_scatter) {
                  int64_t src_offset = 0;
                  int64_t absolute_index_offset = (int64_t)((int64_t*)index_data - (int64_t*)index_ptr); //  / sizeof(int64_t); // index tensor has word size = 8
                  int64_t index_idx;
                  for (int d = ndim - 1; d >= 0; d--) {
                    index_idx = (absolute_index_offset / index_strides[d]) % index_shape[d];
                    absolute_index_offset -= index_idx * index_strides[d];

                    index_idx %= src_shape[d];
                    src_offset += src_strides[d] * index_idx;
                  }
                  absolute_src_ptr = (scalar_t*)src_ptr + src_offset; // * sizeof(scalar_t); // source tensor has word size dependent on type
                }

                kernel_func(
                  (scalar_t*)self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                  !index_larger_than_src_in_scatter ?
                      (scalar_t*)src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride
                    : absolute_src_ptr);

                self_data += strides[SELF_ITER_STRIDE_IDX];
                index_data += strides[INDEX_ITER_STRIDE_IDX];
                if (!index_larger_than_src_in_scatter) src_data += strides[SRC_ITER_STRIDE_IDX];
              }
            }
          }
        };
        iter.for_each(loop, grain_size);
      }
    );
  }
};

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
