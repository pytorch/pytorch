#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/TypeCast.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>

namespace at {
namespace native {
namespace {

static void copy_kernel(TensorIterator& iter, bool non_blocking) {
  ScalarType dtype = iter.dtype(0);
  if (dtype == iter.dtype(1)) {
    // TODO: as the majority of these operations can be done treating
    // their datatypes as opaque bit patterns, we don't actually need
    // separate instantiations per dtype; we only need a separate
    // instantiation per dtype size.  This would probably save us a
    // little bit of code size here
    // TODO: not sure if optimizer is able to compile two levels of
    // conditionals into a single jump table.  We should have a
    // single jump table here; might be worth just writing out the
    // dispatch statement by hand instead of using AT_DISPATCH
    if (dtype == ScalarType::Half) {
      cpu_kernel(iter, [=](at::Half a) -> at::Half { return a; });
    } else if (dtype == ScalarType::ComplexHalf) {
      cpu_kernel(iter, [=](c10::complex<at::Half> a) -> c10::complex<at::Half> { return a; });
    } else if (isQIntType(dtype)) {
      AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return a; },
            [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; });
      });
    } else if (isComplexType(dtype)) {
      if (iter.tensor(0).is_conj() == iter.tensor(1).is_conj()) {
        AT_DISPATCH_COMPLEX_TYPES(dtype, "copy_kernel", [&] {
            cpu_kernel_vec(
              iter,
              [=](scalar_t a) -> scalar_t { return a; },
              [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; });
          });
      } else {
        AT_DISPATCH_COMPLEX_TYPES(dtype, "conj_kernel", [&] {
            cpu_kernel_vec(
              iter,
              [=](scalar_t a) -> scalar_t { return conj_impl(a); },
              [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a.conj(); });
          });
      }
    } else {
      AT_DISPATCH_ALL_TYPES_AND2(
          ScalarType::Bool, ScalarType::BFloat16,dtype, "copy_kernel", [&] {
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return a; },
                [=](Vectorized<scalar_t> a) { return a; });
          });
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
      using dest_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, iter.dtype(1), "copy_", [&] {
        // Note (@zasdfgbnm):
        //
        // The code below can not be simplified as
        //    cpu_kernel(iter, c10::static_cast_with_inter_type<dest_t, scalar_t>::apply);
        //
        // because this would force the compiler to instantiate the inline function and generate a function call in the loop
        // instead of inlining it, making all the optimizations like vectorization impossible.
        // You can verify this by looking the the symbols of `libtorch_cpu.so`:
        //
        //    readelf -Ws libtorch_cpu.so | grep static_cast_with_inter_type
        //
        // If done correctly, the above command should have no output.
        //
        // See: https://github.com/pytorch/pytorch/issues/31271
        cpu_kernel(iter, [](scalar_t src) -> dest_t {
          return c10::static_cast_with_inter_type<dest_t, scalar_t>::apply(src); });
      });
    });
    if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
      iter.tensor(0).conj_physical_();
    }
  }
}

template <typename scalar_t>
void transpose_copy_kernel_impl(Tensor& self, const Tensor& src) {
  scalar_t* self_data = self.data_ptr<scalar_t>();
  scalar_t* src_data = src.data_ptr<scalar_t>();

  int64_t M = src.size(0);
  int64_t N = src.size(1);

  constexpr int64_t BLOCK_SIZE = 8;
  int64_t K = divup(M, BLOCK_SIZE);

  // parallel on outer most dimension
  // TODO: vectorize the remainder
  int64_t grain_size = at::internal::GRAIN_SIZE / N / BLOCK_SIZE;
  at::parallel_for(0, K, grain_size, [&] (int64_t begin, int64_t end) {
    int64_t rbegin = begin * BLOCK_SIZE;
    int64_t rend = std::min(end * BLOCK_SIZE, M);

    int64_t i = rbegin;
    for (; i < rend - (rend % BLOCK_SIZE); i += BLOCK_SIZE) {
      int64_t j = 0;
      for (; j < N - (N % BLOCK_SIZE); j += BLOCK_SIZE) {
        vec::transpose_kernel_8x8<scalar_t>(
            &src_data[j * M + i], M, &self_data[i * N + j], N);
      }
      for (; j < N; j++) {
        for (int64_t k = i; k < i + BLOCK_SIZE; k++) {
          self_data[k * N + j] = src_data[j * M + k];
        }
      }
    }
    for (; i < rend; i++) {
      for (int64_t j = 0; j < N; j++) {
        self_data[i * N + j] = src_data[j * M + i];
      }
    }
  });
}

static void  transpose_copy_kernel(Tensor& self, const Tensor& src) {
  TORCH_CHECK(self.is_contiguous(), "self is not contiguous");
  TORCH_CHECK(src.numel() > 0, "expect src number of elements > 0");
  TORCH_CHECK(src.dim() == 2 && self.dim() == 2,
      "expect src and self dims to be 2, self dim: ", src.dim(),
      "; self dim: ", self.dim());
  TORCH_CHECK(src.stride(0) == 1, "src first dimension is not contiguous");
  TORCH_CHECK(src.stride(1) == src.size(0), "expect src.stride(1) == src.size(0)");
  TORCH_CHECK(src.scalar_type() == self.scalar_type(),
      "expect same data type for src and self, src data type: ", src.scalar_type(),
      "; self data type: ", self.scalar_type());

  AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, src.scalar_type(), "transpose_copy_kernel", [&] {
    transpose_copy_kernel_impl<scalar_t>(self, src);
  });
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(copy_stub, &copy_kernel);
REGISTER_DISPATCH(transpose_copy_stub, &transpose_copy_kernel);

} // namespace native
} // namespace at
