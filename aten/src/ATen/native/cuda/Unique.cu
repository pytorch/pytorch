#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <type_traits>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

namespace at {
namespace native{

namespace {


// binary operators of these classes assumes n are the same
// we just don't care the case when it's different

template <typename scalar_t> class TupleIterator;

template <typename scalar_t>
class TupleRef {
  int64_t n;
  scalar_t *ptr;
  friend TupleRef<scalar_t> TupleIterator<scalar_t>::operator*();
  friend TupleRef<scalar_t> TupleIterator<scalar_t>::operator[](int64_t i);
  TupleRef(int64_t n, scalar_t ptr):n(n), ptr(ptr) {}
public:
  bool operator<(const TupleRef &other) {
    int64_t i = 0;
    while(i < n - 1 && ptr[i] == other.ptr[i]) i++;
    return ptr[i] < other.ptr[i];
  }
  TupleRef &operator=(const TupleRef &other) {
    int64_t i = 0;
    while(i < n) ptr[i] = other.ptr[i];
    ptr = other.ptr;
    return *this;
  }
  bool operator==(const TupleRef &other) {
    int64_t i = 0;
    while(i < n - 1 && ptr[i] == other.ptr[i]) i++;
    return ptr[i] == other.ptr[i];
  }
  bool operator!=(const TupleRef &other) {
    return !(*this == other);
  }
};

template <typename scalar_t>
class TupleIterator {
  int64_t n;
  scalar_t *ptr;
public:
  TupleIterator(int64_t n, scalar_t ptr):n(n), ptr(ptr){}
  TupleIterator(int64_t n):TupleIterator(n, nullptr){}
  TupleRef<scalar_t> operator*() {
    return TupleRef<scalar_t>(n, ptr);
  }
  TupleRef<scalar_t> operator[](int64_t i) {
    return TupleRef<scalar_t>(n, ptr + i * n);
  }
  TupleIterator operator+(int64_t i) {
    return TupleIterator(n, ptr + i * n);
  }
  TupleIterator operator-(int64_t i) {
    return TupleIterator(n, ptr - i * n);
  }
  int64_t operator-(const TupleIterator &other) {
    return (ptr - other.ptr) / n;
  }
  TupleIterator &operator+=(int64_t i) {
    ptr += i * n;
    return *this;
  }
  TupleIterator &operator-=(int64_t i) {
    ptr -= i * n;
    return *this;
  }
  TupleIterator &operator++() {
    return *this += 1;
  }
  TupleIterator &operator++(int) {
    return *++*this - 1;
  }
  TupleIterator &operator--() {
    return *this -= 1;
  }
  TupleIterator &operator--(int) {
    return --*this + 1;
  }
  bool operator<(const TupleIterator &other) {
    return ptr < other.ptr;
  }
  bool operator<=(const TupleIterator &other) {
    return ptr <= other.ptr;
  }
  bool operator>(const TupleIterator &other) {
    return ptr > other.ptr;
  }
  bool operator>=(const TupleIterator &other) {
    return ptr >= other.ptr;
  }
  bool operator==(const TupleIterator &other) {
    return ptr == other.ptr;
  }
  bool operator!=(const TupleIterator &other) {
    return ptr != other.ptr;
  }
};

template <typename Iterator>
std::tuple<Tensor, Tensor, int64_t> compute_unique(Iterator data, int64_t num_inp, const bool return_inverse, const bool return_counts, TensorOptions options) {
  using data_t = c10::guts::remove_reference_t<decltype(*data)>;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  //sort
  Tensor inverse_indices;
  if (!return_inverse) {
      inverse_indices = at::empty({0}, options);
      thrust::sort(policy, data, data + num_inp);
  } else {
      Tensor sorted_indices = at::arange(0, num_inp, options);
      int64_t* sorted_indices_ptr = sorted_indices.data<int64_t>();
      thrust::sort_by_key(policy, data, data + num_inp, sorted_indices_ptr);
      Tensor inv_loc = at::empty({num_inp}, options);
      inverse_indices = at::empty({num_inp}, options);
      int64_t* inv_loc_ptr = inv_loc.data<int64_t>();
      int64_t* inverse_indices_ptr = inverse_indices.data<int64_t>();
      thrust::adjacent_difference(policy, data, data + num_inp, inv_loc_ptr, [=] __device__ (data_t a, data_t b) -> int64_t { if (a != b) {return 1;} else { return 0; }});
      inv_loc[0] = 0;
      thrust::inclusive_scan(policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
      thrust::scatter(policy, inv_loc_ptr, inv_loc_ptr + num_inp, sorted_indices_ptr, inverse_indices_ptr);
  }

  // unique
  Tensor counts = at::empty({0}, options);
  int64_t num_out;
  if (!return_counts) {
      num_out = thrust::unique(policy, data, data + num_inp) - data;
  } else {
      Tensor sorted_indices = at::arange(0, num_inp + 1, options);
      int64_t* sorted_indices_ptr = sorted_indices.data<int64_t>();
      num_out = thrust::unique_by_key(policy, data, data + num_inp, sorted_indices_ptr).first - data;
      sorted_indices[num_out] = num_inp;
      counts.resize_(num_out);
      int64_t* counts_ptr = counts.data<int64_t>();
      thrust::adjacent_difference(policy, sorted_indices_ptr + 1, sorted_indices_ptr + num_out + 1, counts_ptr);
  }

  THCudaCheck(cudaGetLastError());
  return std::tuple<Tensor, Tensor, int64_t>(inverse_indices, counts, num_out);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> _unique_cuda_template(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts) {

    Tensor output = self.contiguous().reshape(-1).clone();
    int64_t num_inp = output.numel();
    scalar_t* output_data = output.data<scalar_t>();

    Tensor inverse_indices, counts;
    int64_t num_out;
    std::tie(inverse_indices, counts, num_out) = compute_unique(output_data, num_inp, return_inverse, return_counts, self.options().dtype(kLong));
    output.resize_(num_out);

    if (return_inverse) {
        inverse_indices.resize_(self.sizes());
    }

    return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> _unique_dim_cuda_template(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts) {

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
    auto policy = thrust::cuda::par(allocator).on(stream);

    Tensor input_flat = self.transpose(dim, 0);
    auto orig_sizes = input_flat.sizes().vec();
    input_flat = input_flat.contiguous().view({input_flat.size(0), -1});

    scalar_t* input_flat_ptr = input_flat.data<scalar_t>();

    Tensor indices = at::arange(0, input_flat.size(0), self.options().dtype(kLong));
    int64_t* indices_ptr = indices.data<int64_t>();
    int64_t numel = input_flat.size(1);

    // sort indices using data
    thrust::sort(policy, indices_ptr, indices_ptr + indices.numel(),
      [=] __device__ (int64_t a, int64_t b) -> bool {
        for (int64_t i = 0; i < numel; ++i) {
          scalar_t lhs = input_flat_ptr[i + a * numel];
          scalar_t rhs = input_flat_ptr[i + b * numel];
          if (lhs < rhs) {
            return true;
          } else if (lhs > rhs) {
            return false;
          }
        }
        return false;
      });

    Tensor input_sorted = input_flat.index_select(0, indices);

    // get unique tensors
    scalar_t* input_sorted_ptr = input_sorted.data<scalar_t>();
    Tensor input_sorted_indices = at::arange(0, input_sorted.size(0), self.options().dtype(kLong));
    int64_t* input_sorted_indices_ptr = input_sorted_indices.data<int64_t>();
    auto last = thrust::unique(policy, input_sorted_indices_ptr, input_sorted_indices_ptr + input_sorted_indices.numel(),
      [=] __device__ (int64_t a, int64_t b) -> bool {
        for (int64_t i = 0; i < numel; ++i) {
          scalar_t lhs = input_sorted_ptr[i + a * numel];
          scalar_t rhs = input_sorted_ptr[i + b * numel];
          if (lhs != rhs) {
            return false;
          }
        }
        return true;
      });
    input_sorted_indices.resize_(last - input_sorted_indices_ptr);
    Tensor output = input_sorted.index_select(0, input_sorted_indices);

    // reshape back
    auto new_sizes = std::vector<int64_t>(orig_sizes);
    new_sizes[0] = -1;
    output = output.view(new_sizes);
    output = output.transpose(0, dim);

    // calculate inverse indices and counts
    Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
    Tensor counts = at::zeros(output.size(dim), self.options().dtype(kLong));
    if (return_inverse || return_counts) {
      int64_t size = self.size(dim);
      inverse_indices.resize_(size);
      Tensor mask = at::empty(input_sorted.size(0), self.options().dtype(kLong));
      mask[0] = 1;
      for (int i = 0; i < input_sorted.size(0) - 1; ++i) {
        if (!at::equal(input_sorted[i], input_sorted[i+1])) {
          mask[i+1] = 1;
        } else {
          mask[i+1] = 0;
        }
      }

      Tensor imask = at::cumsum(mask, 0) - 1;
      for (int i = 0; i < indices.size(0); ++i) {
        inverse_indices[indices[i]] = imask[i];
        counts[inverse_indices[indices[i]]] += 1;
      }
    }

    THCudaCheck(cudaGetLastError());
    return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
  }

} // namespace


std::tuple<Tensor, Tensor, Tensor>
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return _unique_cuda_template<scalar_t>(self, return_inverse, return_counts);
  });
}

std::tuple<Tensor, Tensor, Tensor>
_unique_dim_cuda(const Tensor& self, const int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "unique_dim", [&] {
    return _unique_dim_cuda_template<scalar_t>(self, dim, return_inverse, return_counts);
  });
}

}  // namespace native
}  // namespace at
