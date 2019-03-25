#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCGeneral.h>
#include <THC/THCThrustAllocator.cuh>
#include <thrust/execution_policy.h>

#include <tuple>
#include <iterator>
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
public:
  __device__ TupleRef(int64_t n, scalar_t *ptr):n(n), ptr(ptr) {}
  __device__ TupleRef(): TupleRef(0, nullptr) {}

  __device__ bool operator<(const TupleRef &other) const {
    int64_t i = 0;
    while(i < n - 1 && ptr[i] == other.ptr[i]) i++;
    return ptr[i] < other.ptr[i];
  }
  __device__ TupleRef &operator=(const TupleRef &other) {
    int64_t i = 0;
    while(i < n) ptr[i] = other.ptr[i];
    ptr = other.ptr;
    return *this;
  }
  __device__ bool operator==(const TupleRef &other) const {
    int64_t i = 0;
    while(i < n - 1 && ptr[i] == other.ptr[i]) i++;
    return ptr[i] == other.ptr[i];
  }
  __device__ bool operator!=(const TupleRef &other) const {
    return !(*this == other);
  }

  // used by thrust::adjacent_difference, we actually don't care what the
  // results of the conversion are, we just need the conversion to be
  // there to satisfy the requirement of thrust.
  __device__ TupleRef(int64_t):n(0), ptr(nullptr){}
  __device__ operator int64_t () const {
    return 0;
  }
};

template <typename scalar_t>
class TupleIterator {
  int64_t n;
  scalar_t *ptr;
public:
  using difference_type = int64_t;
  using value_type = TupleRef<scalar_t>;
  using pointer = TupleIterator<scalar_t>;
  using reference = TupleRef<scalar_t>;
  using iterator_category = std::random_access_iterator_tag;

  __device__ TupleIterator(int64_t n, scalar_t *ptr):n(n), ptr(ptr){}

  __device__ TupleRef<scalar_t> operator*() {
    return TupleRef<scalar_t>(n, ptr);
  }
  __device__ TupleRef<scalar_t> operator[](int64_t i) {
    return TupleRef<scalar_t>(n, ptr + i * n);
  }
  __device__ TupleIterator<scalar_t> operator+(int64_t i) const {
    return TupleIterator<scalar_t>(n, ptr + i * n);
  }
  __device__ TupleIterator operator-(int64_t i) const {
    return TupleIterator<scalar_t>(n, ptr - i * n);
  }
  __device__ int64_t operator-(const TupleIterator &other) const {
    return (ptr - other.ptr) / n;
  }
  __device__ TupleIterator<scalar_t> &operator+=(int64_t i) {
    ptr += i * n;
    return *this;
  }
  __device__ TupleIterator<scalar_t> &operator-=(int64_t i) {
    ptr -= i * n;
    return *this;
  }
  __device__ TupleIterator<scalar_t> &operator++() {
    return *this += 1;
  }
  __device__ TupleIterator<scalar_t> &operator--() {
    return *this -= 1;
  }
  __device__ bool operator<(const TupleIterator<scalar_t> &other) const {
    return ptr < other.ptr;
  }
  __device__ bool operator==(const TupleIterator<scalar_t> &other) const {
    return ptr == other.ptr;
  }
  __device__ bool operator!=(const TupleIterator<scalar_t> &other) const {
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

    Tensor input_flat = self.transpose(dim, 0);
    auto orig_sizes = input_flat.sizes().vec();
    int64_t num_inp = orig_sizes[0];
    input_flat = input_flat.contiguous().view({num_inp, -1});
    int64_t size = input_flat.size(1);

    Tensor output = input_flat.clone();
    TupleIterator<scalar_t> output_data = TupleIterator<scalar_t>(size, output.data<scalar_t>());

    Tensor inverse_indices, counts;
    int64_t num_out;
    std::tie(inverse_indices, counts, num_out) = compute_unique(output_data, num_inp, return_inverse, return_counts, self.options().dtype(kLong));
    output.resize_({num_out, size});

    orig_sizes[0] = num_out;
    return std::tuple<Tensor, Tensor, Tensor>(output.reshape(orig_sizes).transpose(dim, 0), inverse_indices, counts);
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
