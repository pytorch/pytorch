#include <ATen/native/Indexing.h>

#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {
namespace {

using namespace vec256;

struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          IntArrayRef original_sizes, IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data())
    , original_sizes(original_sizes.data()) {
    AT_ASSERT(original_strides.size() == num_indexers);
    AT_ASSERT(original_sizes.size() == num_indexers);
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;
  const int64_t* original_sizes;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (int j = 0; j < num_indexers; j++) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      int64_t size = original_sizes[j];
      if (value < -size || value >= size) {
        AT_INDEX_ERROR("index ", value, " is out of bounds for dimension ", j, " with size ", size);
      }
      if (value < 0) {
        value += size;
      }
      offset += value * original_strides[j];
    }
    return offset;
  }
};

static bool is_constant_index(int ntensor, const int64_t* strides) {
  AT_ASSERT(ntensor >= 3);
  for (int arg = 2; arg < ntensor; arg++) {
    if (strides[arg] != 0) {
      return false;
    }
  }
  return true;
}

template <typename scalar_t, typename func_t>
void cpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      const func_t& f, bool serial_execution=false)
{
  int ntensor = iter.ntensors();
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
    char* dst = data[0];
    char* src = data[1];
    if (is_constant_index(ntensor, strides)) {
      // specialization for when every element uses the same index
      int64_t offset = indexer.get(0);
      if (strides[0] == sizeof(scalar_t) && strides[1] == sizeof(scalar_t)) {
        for (int64_t i = 0; i < n; i++) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      } else {
        for (int64_t i = 0; i < n; i++) {
          f(dst + strides[0] * i, src + strides[1] * i, offset);
        }
      }
    } else {
      for (int64_t i = 0; i < n; i++) {
        int64_t offset = indexer.get(i);
        f(dst + strides[0] * i, src + strides[1] * i, offset);
      }
    }
  };
  if (serial_execution) {
    iter.serial_for_each(loop, {0, iter.numel()});
  } else {
    iter.for_each(loop);
  }
}

void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, iter.dtype(), "index_cpu", [&] {
    cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
      *(scalar_t*)dst = *(scalar_t*)(src + offset);
    });
  });
}

void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate) {
  // NOTE: duplicate indices are only supported if accumulate is true.
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::Bool, iter.dtype(), "index_put", [&] {
    if (accumulate) {
      // TODO: investigate parallelization of the accumulate kernel. Unlike the non-accumulate case,
      // this needs to be thread-safe.
      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
        *(scalar_t*)(dst + offset) += *(scalar_t*)src;
      }, /*serial_execution=*/true);
    } else {
      cpu_index_kernel<scalar_t>(iter, index_size, index_stride, [](char* dst, char* src, int64_t offset) {
        *(scalar_t*)(dst + offset) = *(scalar_t*)src;
      });
    }
  });
}

inline int64_t ensure_nonempty_dim(int64_t dim) {
  return std::max<int64_t>(dim, 1);
}

inline int64_t ensure_nonempty_size(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.size(dim);
}

inline int64_t ensure_nonempty_stride(const Tensor& t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.stride(dim);
}

template <typename scalar_t, typename func_t>
void cpu_apply_dim_kernel(
  Tensor& self,
  int64_t dim,
  const Tensor& index,
  const Tensor& src,
  const func_t& f
) {
  auto loop = [&](int64_t begin, int64_t end) {
    scalar_t* self_data = self.data_ptr<scalar_t>();
    int64_t self_dim_stride = ensure_nonempty_stride(self, dim);

    int64_t* index_data = index.data_ptr<int64_t>();
    int64_t index_dim_stride = ensure_nonempty_stride(index, dim);

    scalar_t* src_data = src.data_ptr<scalar_t>();
    int64_t src_dim_stride = ensure_nonempty_stride(src, dim);

    int64_t index_dim = ensure_nonempty_dim(index.dim());
    auto counter = at::DimCounter(index.sizes(), {begin, end});

    // offset pointers according to counts
    for (int64_t d = 0; d < index_dim; ++d) {
      if (d == dim) {
        continue;
      }
      self_data += counter.values[d] * ensure_nonempty_stride(self, d);
      index_data += counter.values[d] * ensure_nonempty_stride(index, d);
      src_data += counter.values[d] * ensure_nonempty_stride(src, d);
    }

    for (int64_t i = begin; i < end; ++i) {
      f(
        self_data, self_dim_stride,
        index_data, index_dim_stride,
        src_data, src_dim_stride
      );

      for (int64_t d = 0; d < index_dim; ++d) {
        if (d == dim) {
          continue;
        }

        counter.values[d]++;
        self_data += ensure_nonempty_stride(self, d);
        index_data += ensure_nonempty_stride(index, d);
        src_data += ensure_nonempty_stride(src, d);

        if (counter.values[d] == ensure_nonempty_size(index, d)) {
          self_data -= counter.values[d] * ensure_nonempty_stride(self, d);
          index_data -= counter.values[d] * ensure_nonempty_stride(index, d);
          src_data -= counter.values[d] * ensure_nonempty_stride(src, d);
          counter.values[d] = 0;
        }
        else {
          break;
        }
      }
    }
  };

  int64_t max_num_iters = index.numel() / ensure_nonempty_size(index, dim);
  if (max_num_iters < internal::GRAIN_SIZE || at::get_num_threads() == 1) {
    loop(0, max_num_iters);
  }
  else {
    at::parallel_for(0, max_num_iters, internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
      loop(begin, end);
    });
  }
}

/*
  * Used for 'scatter' and 'scatter_add'
  * 
  * Tests:
  *  1. index.size(d) <= src.size(d) for all d
  *  2. index.size(d) <= self.size(d) for all d != dim
  */
struct ScatterSizeCheckFunctor {
  void operator()(
    const Tensor& self, int64_t dim,
    const Tensor& index, const Tensor& src
  ) {
    bool is_wrong_shape = false;
    int64_t self_dims = ensure_nonempty_dim(self.dim());
    for (int64_t d = 0; d < self_dims; ++d) {
      int64_t index_d_size = ensure_nonempty_size(index, d);
      if (index_d_size > ensure_nonempty_size(src, d)) {
        is_wrong_shape = true;
        break;
      }
      if (d != dim) {
        if (index_d_size > ensure_nonempty_size(self, d)) {
          is_wrong_shape = true;
          break;
        }
      }
    }

    TORCH_CHECK(!is_wrong_shape,
      "Expected ", index, " ", index.sizes(),
      "to be smaller size than ", src, " ", src.sizes(),
      " and to be smaller than ", self, " ", self.sizes(),
      " apart from dimension ", dim
    );
  }
};

void scatter_add_cpu_kernel(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
  if (index.numel() == 0) {
    return;
  }

  ScatterSizeCheckFunctor()(self, dim, index, src);

  dim = maybe_wrap_dim(dim, self.dim());
  int64_t index_dim_size = ensure_nonempty_size(index, dim);
  int64_t self_dim_size = ensure_nonempty_size(self, dim);

  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Bool, ScalarType::Half, self.scalar_type(),
    "scatter_add_", [&] {
      cpu_apply_dim_kernel<scalar_t>(self, dim, index, src,
        [&] (
          auto* self_data, auto self_dim_stride,
          const auto* index_data, auto index_dim_stride,
          const auto* src_data, auto src_dim_stride
        ) {
          for (int64_t i = 0; i < index_dim_size; ++i) {
            int64_t idx_dim = index_data[i * index_dim_stride];
            TORCH_CHECK(idx_dim >= 0 && idx_dim < self_dim_size,
              "Invalid index in scatter_add");
            self_data[idx_dim * self_dim_stride] += src_data[i * src_dim_stride];
          }
        }
      );
    }
  );
}

} // anonymous namespace


REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);

REGISTER_DISPATCH(scatter_add_stub, &scatter_add_cpu_kernel);

}} // namespace at::native
