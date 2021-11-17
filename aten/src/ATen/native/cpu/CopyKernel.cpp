#include <ATen/core/op_registration/op_allowlist.h>
#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/TypeCast.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/Parallel.h>

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
    if (iter.tensor(0).is_neg() == iter.tensor(1).is_neg()) {
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
        // This case should never actually happen since currently there's no way to get a complex tensor
        // with negative bit.
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
      if (dtype == ScalarType::Half) {
        cpu_kernel(iter, [=](at::Half a) -> at::Half { return -a; });
      } else if (isComplexType(dtype)) {
        if (iter.tensor(0).is_conj() == iter.tensor(1).is_conj()) {
          AT_DISPATCH_COMPLEX_TYPES(dtype, "copy_kernel", [&] {
              cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return -a; },
                [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a.neg(); });
            });
        } else {
          AT_DISPATCH_COMPLEX_TYPES(dtype, "conj_kernel", [&] {
              cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return -1 * conj_impl(a); },
                [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a.neg().conj(); });
            });
        }
      } else {
          AT_DISPATCH_ALL_TYPES_AND2(
            ScalarType::Bool, ScalarType::BFloat16,dtype, "copy_kernel", [&] {
              cpu_kernel_vec(
                  iter,
                  [=](scalar_t a) -> scalar_t { return -a; },
                  [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a.neg(); });
            });
      }
    }
  } else {
    if ((dtype == kFloat) && (iter.dtype(1) == kBFloat16) &&
        (iter.tensor(0).is_contiguous() && iter.tensor(1).is_contiguous() ||
         iter.tensor(0).is_contiguous(at::MemoryFormat::ChannelsLast) &&
         iter.tensor(1).is_contiguous(at::MemoryFormat::ChannelsLast))) {
      using dest_t = float;
      using scalar_t = BFloat16;
      int64_t grain_size = at::internal::GRAIN_SIZE;
      dest_t* output_data = iter.tensor(0).data_ptr<dest_t>();
      scalar_t* input_data = iter.tensor(1).data_ptr<scalar_t>();
      parallel_for(0, iter.tensor(1).numel(), grain_size, [&] (int64_t begin, int64_t end) {
        int64_t size = end - begin;
        int64_t d = 0;
        using Vecd = Vectorized<dest_t>;
        using Vecs = Vectorized<scalar_t>;
        for (; d < size - (size % Vecs::size()); d += Vecs::size()) {
          Vecs data_vec = Vecs::loadu(input_data + begin + d);
          Vecd data_vec0, data_vec1;
          std::tie(data_vec0, data_vec1) = convert_bfloat16_float(data_vec);
          data_vec0.store(output_data + begin + d);
          data_vec1.store(output_data + begin + d + Vecd::size());
        }
        if (size - d > 0) {
          Vecs data_vec = Vecs::loadu(input_data + begin + d, size - d);
          Vecd data_vec0, data_vec1;
          std::tie(data_vec0, data_vec1) = convert_bfloat16_float(data_vec);
          data_vec0.store(output_data + begin + d, ((size - d) > Vecd::size())?  Vecd::size() : (size - d));
          data_vec1.store(output_data + begin + d + Vecd::size(), ((size - d) > Vecd::size())? (size - d - Vecd::size()) : 0);
        }
      });
    } else if ((dtype == kBFloat16) && (iter.dtype(1) == kFloat) &&
               (iter.tensor(0).is_contiguous() && iter.tensor(1).is_contiguous() ||
                iter.tensor(0).is_contiguous(at::MemoryFormat::ChannelsLast) &&
                iter.tensor(1).is_contiguous(at::MemoryFormat::ChannelsLast))) {
      using dest_t = BFloat16;
      using scalar_t = float;
      int64_t grain_size = at::internal::GRAIN_SIZE;
      dest_t* output_data = iter.tensor(0).data_ptr<dest_t>();
      scalar_t* input_data = iter.tensor(1).data_ptr<scalar_t>();
      parallel_for(0, iter.tensor(1).numel(), grain_size, [&] (int64_t begin, int64_t end) {
        int64_t size = end - begin;
        int64_t d = 0;
        using Vecd = Vectorized<dest_t>;
        using Vecs = Vectorized<scalar_t>;
        for (; d < size - (size % (Vecs::size() * 2)); d += Vecs::size() * 2) {
          Vecs data_vec0 = Vecs::loadu(input_data + begin + d);
          Vecs data_vec1 = Vecs::loadu(input_data + begin + d + Vecs::size());
          convert_float_bfloat16(data_vec0, data_vec1).store(output_data + begin + d);
        }
        if (size - d > 0) {
          Vecs data_vec0 = Vecs::loadu(input_data + begin + d, ((size - d) > Vecs::size())?  Vecs::size() : (size - d));
          Vecs data_vec1 = Vecs::loadu(input_data + begin + d + Vecs::size(), ((size - d) > Vecs::size())?  (size - d - Vecs::size()) : 0);
          convert_float_bfloat16(data_vec0, data_vec1).store(output_data + begin + d, size - d);
        }
      });
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
    }
    if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
      iter.tensor(0).conj_physical_();
    }
    if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
      iter.tensor(0).neg_();
    }
  }
}

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &copy_kernel);

} // namespace native
} // namespace at
