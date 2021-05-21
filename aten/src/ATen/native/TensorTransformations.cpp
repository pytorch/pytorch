#include <ATen/native/TensorTransformations.h>
#include <ATen/WrapDimUtilsMulti.h>

#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <vector>

namespace at {
namespace native {
namespace {

constexpr size_t dim_bitset_size = 64;

Tensor build_index(Tensor input, int64_t flip_dim) {
  int64_t element_size_bytes = input.element_size();
  auto num_dims = input.ndimension();
  auto dim_size = input.size(flip_dim);
  auto stride = input.stride(flip_dim);

  auto new_shape = std::vector<int64_t>(num_dims, 1);
  new_shape[flip_dim] = dim_size;

  TensorOptions tensor_options =
    TensorOptions(c10::kLong).
    device(c10::kCPU);

  auto index = at::empty(new_shape, tensor_options);
  auto input_index_ptr = index.data_ptr<int64_t>();

  for(int64_t i = 0; i < dim_size; i++) {
    input_index_ptr[i] = static_cast<int64_t>(dim_size - i - 1) * stride * element_size_bytes;
  }
  return index;
}


std::vector<Tensor> build_indices_loop(Tensor input, IntArrayRef flip_dims) {
  std::vector<Tensor> indices;
  for(auto dim: flip_dims) {
    auto index = build_index(input, dim);
    indices.push_back(index);
  }
  return indices;
}

static TensorIterator make_index_iterator(const Tensor& input, const std::vector<Tensor> indices) {
  TensorIteratorConfig config;

  auto output_tensor = Tensor();
  if(input.is_quantized()) {
    double scale = input.q_scale();
    int64_t zero_point = input.q_zero_point();
    output_tensor = at::_empty_affine_quantized(
        input.sizes(), at::device(c10::kCPU).dtype(input.scalar_type()), scale, zero_point);
  }

  config.set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .declare_static_dtype_and_device(input.scalar_type(), input.device())
        .add_owned_output(output_tensor)
        .add_borrowed_input(input);
  for (auto& index : indices) {
    config.add_input(index);
  }
  return config.build();
}

struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides)
      : num_indexers(num_indexers),
        indexers(indexers),
        indexer_strides(indexer_strides) {}

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;

  int64_t get(int64_t idx) {
    int64_t offset = *(int64_t*)&indexers[0][idx * indexer_strides[0]];
    for (int j = 1; j < num_indexers; j++) {
      offset += *(int64_t*)&indexers[j][idx * indexer_strides[j]];
    }
    return offset;
  }
};

template <typename scalar_t>
void flip_cpu_kernel(TensorIterator& iter) {
  int ntensor = iter.ntensors();
  // When launch the index parallel version, set a relative small grain size
  // less than the INTERNAL::GRAIN_SIZE to make the whole available thread
  // numbers get more balanced work load and a better cache location. The grain
  // size here is chosen by the op benchmark to overcome the thread launch
  // overhead. This value was taken from the AdvancedIndexing kernel.
  const int index_parallel_grain_size = 3000;
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto indexer = Indexer(ntensor - 2, &data[2], &strides[2]);
    char* dst = data[0];
    char* src = data[1];

    for (int64_t i = 0; i < n; i++) {
      int64_t offset = indexer.get(i);
      *(scalar_t*)(dst + strides[0] * i) =
          *(scalar_t*)(src + strides[1] * i + offset);
    }
  };

  iter.for_each(loop, index_parallel_grain_size);
}
} // anonymous namespace

Tensor flip_cpu(const Tensor& self, IntArrayRef dims) {
  if(dims.size() == 0) {
    return self.clone();
  }

  auto input = self;
  const int64_t total_dims = input.dim();
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);

  std::vector<int64_t> flip_dims;
  for(int64_t i = 0; i < total_dims; i++) {
      if(flip_dims_b[i]) {
        flip_dims.push_back(i);
      }
  }

  auto shape = input.sizes().vec();
  auto strides = input.strides().vec();

  // Set stride to zero on the dimensions that are going to be flipped
  for(auto dim: flip_dims) {
    strides[dim] = 0;
  }

  // Restride the input to index only on the dimensions to flip
  auto restrided_input = input.as_strided(shape, strides);
  auto indices = build_indices_loop(input, flip_dims);
  auto iter = make_index_iterator(restrided_input, indices);

  if (input.is_quantized()) {
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(
        input.scalar_type(), "flip_quantized_cpu", [&] {
          flip_cpu_kernel<scalar_t>(iter);
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kHalf, kBFloat16, input.scalar_type(), "flip_cpu", [&] {
          flip_cpu_kernel<scalar_t>(iter);
        });
  }
  auto result = iter.output();
  return result;
}

Tensor roll_cpu(const Tensor& self, IntArrayRef shifts, IntArrayRef dims) {
  if (dims.size() != 1 || shifts.size() != 1) {
    return roll_common(self, shifts, dims);
  }
  // avoid a div zero error below.
  if (self.numel() == 0) {
    return self.clone(at::MemoryFormat::Preserve);
  }
  int64_t dim = dims[0];
  int64_t size = self.size(dim);
  int64_t start = (size - shifts[0]) % size;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if (start < 0) {
    start = start + size;
  }
  auto t0 = self.narrow(dim, start, size-start);
  auto t1 = self.narrow(dim, 0, start);
  return at::cat({t0, t1}, dim);
}

Tensor rot90(const Tensor& self, int64_t k, IntArrayRef dims) {
  const int64_t total_dims = self.dim(), total_rot_dims = dims.size();

  TORCH_CHECK(total_rot_dims == 2,
    "expected total rotation dims == 2, but got dims = ", total_rot_dims);

  TORCH_CHECK(total_dims >= 2,
    "expected total dims >= 2, but got total dims = ", total_dims);

  TORCH_CHECK(dims[0] != dims[1] && std::abs(dims[0] - dims[1]) != total_dims,
    "expected rotation dims to be different, but got dim0 = ", dims[0],
    " and dim1 = ", dims[1]);

  // check range of dims
  TORCH_CHECK(dims[0] < total_dims && dims[0] >= -total_dims,
    "Rotation dim0 out of range, dim0 = ", dims[0]);

  TORCH_CHECK(dims[1] < total_dims && dims[1] >= -total_dims,
    "Rotation dim1 out of range, dim1 = ", dims[1]);

  // handle modulo with negative k
  k = (4 + (k % 4)) % 4;

  switch(k) {
    case 1:
      return self.flip({dims[1]}).transpose_(dims[0], dims[1]);
    case 2:
      return self.flip(dims);
    case 3:
      return self.flip({dims[0]}).transpose_(dims[0], dims[1]);
    default:
      return self.clone(at::MemoryFormat::Contiguous);
  }
}

Tensor fliplr(const Tensor& self) {
  TORCH_CHECK(self.dim() >= 2, "Input must be >= 2-d.");

  return self.flip({1});
}

Tensor flipud(const Tensor& self) {
  TORCH_CHECK(self.dim() >= 1, "Input must be >= 1-d.");

  return self.flip({0});
}

Tensor atleast_1d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1});
    default:
      return self;
  }
}

std::vector<Tensor> atleast_1d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_1d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_2d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1});
    case 1: {
      return self.unsqueeze(0);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_2d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_2d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

Tensor atleast_3d(const Tensor& self) {
  switch (self.dim()) {
    case 0:
      return self.reshape({1, 1, 1});
    case 1: {
      return self.unsqueeze(0).unsqueeze(-1);
    }
    case 2: {
      return self.unsqueeze(-1);
    }
    default:
      return self;
  }
}

std::vector<Tensor> atleast_3d(TensorList tensors) {
  std::vector<Tensor> result(tensors.size());
  auto transform_lambda = [](const Tensor& input) -> Tensor {
    return at::native::atleast_3d(input);
  };
  std::transform(tensors.cbegin(), tensors.cend(), result.begin(), transform_lambda);
  return result;
}

}} // namespace at::native
