#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/Copy.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/CopyKernel.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/util/TypeCast.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/TensorIteratorInternal.h>
#include <ATen/Parallel.h>

namespace at::native {
inline namespace CPU_CAPABILITY {

static void float_bfloat16_copy_kernel(TensorIteratorBase &iter, bool requires_neg) {
  auto strides_out = iter.strides(0);
  auto strides_in = iter.strides(1);
  auto shape = iter.shape();
  c10::SmallBuffer<int64_t, 8> strides(2 * std::max(iter.ndim(), 2));
  auto get_strides = [](int64_t* strides, IntArrayRef strides_out, IntArrayRef strides_in, int64_t ndim) {
      for (const auto dim : c10::irange(ndim)) {
        for (const auto arg : c10::irange(2)) {
          *strides++ = arg == 0? strides_out[dim] : strides_in[dim];
        }
      }
      // Always at least 2d strides to support 2d for_each loops
      if (ndim < 2) {
        std::fill_n(strides, (2 - ndim) * 2, 0);
      }
    };
  get_strides(strides.data(), strides_out, strides_in, iter.ndim());
  if ((iter.dtype(0) == kFloat) && (iter.dtype(1) == kBFloat16)) {
    using dest_t = float;
    using scalar_t = BFloat16;
    using Vecd = Vectorized<dest_t>;
    using Vecs = Vectorized<scalar_t>;
    c10::SmallBuffer<char*, 2> ptrs(2);
    dest_t* output_data = iter.tensor_base(0).data_ptr<dest_t>();
    scalar_t* input_data = iter.tensor_base(1).data_ptr<scalar_t>();
    ptrs[0] = reinterpret_cast<char*>(output_data);
    ptrs[1] = reinterpret_cast<char*>(input_data);

    int64_t grain_size = at::internal::GRAIN_SIZE;

    auto loop = [strides_in, requires_neg](char** base, const int64_t* strides, int64_t size0, int64_t size1) {
      std::array<char*, 2> data;
      std::copy_n(base, 2, data.data());
      const int64_t *outer_strides = &strides[2];

      for (const auto it C10_UNUSED : c10::irange(size1)) {
        Vecd dst_s;
        if (strides_in[0] == 0) {
          dst_s = Vecd(dest_t(*((scalar_t*)data[1])));
          if (requires_neg) {
            dst_s = dst_s.neg();
          }
        }
        int64_t i = 0;
        for (; i <= size0 - Vecs::size(); i += Vecs::size()) {
          if (strides_in[0] != 0) {
            Vecs data_vec = Vecs::loadu(data[1] + i * sizeof(scalar_t));
            Vecd data_vec0, data_vec1;
            std::tie(data_vec0, data_vec1) = convert_bfloat16_float(data_vec);
            if (requires_neg) {
              data_vec0 = data_vec0.neg();
              data_vec1 = data_vec1.neg();
            }
            data_vec0.store(data[0] + i * sizeof(dest_t));
            data_vec1.store(data[0] + (i + Vecd::size()) * sizeof(dest_t));
          } else {
            dst_s.store(data[0] + i * sizeof(dest_t));
            dst_s.store(data[0] + (i + Vecd::size()) * sizeof(dest_t));
          }
        }
        if (i < size0) {
          if (strides_in[0] != 0) {
            Vecs data_vec = Vecs::loadu(data[1] + i * sizeof(scalar_t), size0 - i);
            Vecd data_vec0, data_vec1;
            std::tie(data_vec0, data_vec1) = convert_bfloat16_float(data_vec);
            if (requires_neg) {
              data_vec0 = data_vec0.neg();
              data_vec1 = data_vec1.neg();
            }
            data_vec0.store(data[0] + i * sizeof(dest_t), ((size0 - i) > Vecd::size())?  Vecd::size() : (size0 - i));
            data_vec1.store(data[0] + (i + Vecd::size()) * sizeof(dest_t), ((size0 - i) > Vecd::size())? (size0 - i - Vecd::size()) : 0);
          } else {
            dst_s.store(data[0] + i * sizeof(dest_t), ((size0 - i) > Vecd::size())?  Vecd::size() : (size0 - i));
            dst_s.store(data[0] + (i + Vecd::size()) * sizeof(dest_t), ((size0 - i) > Vecd::size())? (size0 - i - Vecd::size()) : 0);
          }
        }
        data[0] += outer_strides[0];
        data[1] += outer_strides[1];
      }

    };

    parallel_for(0, iter.numel(), grain_size, [&] (int64_t begin, int64_t end) {
      at::internal::serial_for_each(shape, strides, ptrs.data(), 2, loop, {begin, end});
    });
  } else if ((iter.dtype(0) == kBFloat16) && (iter.dtype(1) == kFloat)) {
    using dest_t = BFloat16;
    using scalar_t = float;
    using Vecd = Vectorized<dest_t>;
    using Vecs = Vectorized<scalar_t>;
    c10::SmallBuffer<char*, 2> ptrs(2);
    dest_t* output_data = iter.tensor_base(0).data_ptr<dest_t>();
    scalar_t* input_data = iter.tensor_base(1).data_ptr<scalar_t>();

    ptrs[0] = reinterpret_cast<char*>(output_data);
    ptrs[1] = reinterpret_cast<char*>(input_data);

    int64_t grain_size = at::internal::GRAIN_SIZE;

    auto loop = [strides_in, requires_neg](char** base, const int64_t* strides, int64_t size0, int64_t size1) {
      std::array<char*, 2> data;
      std::copy_n(base, 2, data.data());
      const int64_t *outer_strides = &strides[2];

      for (const auto it C10_UNUSED : c10::irange(size1)) {
        Vecd dst_s;
        if (strides_in[0] == 0) {
          dst_s = Vecd(dest_t(*((scalar_t*)data[1])));
          if (requires_neg) {
            dst_s = dst_s.neg();
          }
        }
        int64_t i = 0;
        for (; i <= size0 - 2 * Vecs::size(); i += 2 * Vecs::size()) {
          if (strides_in[0] != 0) {
            Vecs data_vec0 = Vecs::loadu(data[1] + i * sizeof(scalar_t));
            Vecs data_vec1 = Vecs::loadu(data[1] + (i + Vecs::size()) * sizeof(scalar_t));
            auto data_vec = convert_float_bfloat16(data_vec0, data_vec1);
            if (requires_neg) {
              data_vec = data_vec.neg();
            }
            data_vec.store(data[0] + i * sizeof(dest_t));
          } else {
            dst_s.store(data[0] + i * sizeof(dest_t));
          }

        }
        if (i < size0) {
          if (strides_in[0] != 0) {
            Vecs data_vec0 = Vecs::loadu(data[1] + i * sizeof(scalar_t), ((size0 - i) > Vecs::size())?  Vecs::size() : (size0 - i));
            Vecs data_vec1 = Vecs::loadu(data[1] + (i + Vecs::size()) * sizeof(scalar_t), ((size0 - i) > Vecs::size())?  (size0 - i - Vecs::size()) : 0);
            auto data_vec = convert_float_bfloat16(data_vec0, data_vec1);
            if (requires_neg) {
              data_vec = data_vec.neg();
            }
            data_vec.store(data[0] + i * sizeof(dest_t), size0 - i);
          } else {
            dst_s.store(data[0] + i * sizeof(dest_t), size0 - i);
          }
        }
        data[0] += outer_strides[0];
        data[1] += outer_strides[1];
      }

    };
    parallel_for(0, iter.numel(), grain_size, [&] (int64_t begin, int64_t end) {
      at::internal::serial_for_each(shape, strides, ptrs.data(), 2, loop, {begin, end});
    });
  }
}

void direct_copy_kernel(TensorIteratorBase &iter) {
  // TODO: we don't actually need separate instantiations per dtype;
  // we only need a separate instantiation per dtype size. This would
  // probably save us a little bit of code size here
  // TODO: not sure if optimizer is able to compile two levels of
  // conditionals into a single jump table.  We should have a
  // single jump table here; might be worth just writing out the
  // dispatch statement by hand instead of using AT_DISPATCH
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_kernel", [&] {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return a; },
          [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; });
    });
  } else if (dtype == ScalarType::ComplexHalf) {
    cpu_kernel(iter, [=](c10::complex<at::Half> a) -> c10::complex<at::Half> { return a; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kHalf, kBFloat16, dtype, "copy_kernel", [&] {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return a; },
          [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; });
    });
  }
}

static void neg_conj_kernel(TensorIteratorBase &iter) {
  // fused a = b.neg().conj_physical()
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_cpu", [&] {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return -conj_impl(a); },
        [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a.neg().conj(); });
  });
}

static void copy_same_dtype(TensorIteratorBase &iter, bool requires_conj, bool requires_neg) {
  if (requires_neg) {
    // This case should never actually happen since currently there's no way to get a complex tensor
    // with negative bit.
    if (requires_conj) {
      neg_conj_kernel(iter);
    } else {
      neg_kernel(iter);
    }
  } else {
    if (requires_conj) {
      conj_kernel(iter);
    } else {
      direct_copy_kernel(iter);
    }
  }
}

void copy_kernel(TensorIterator& iter, bool /*non_blocking*/) {
  ScalarType dtype = iter.dtype(0);
  const bool requires_conj = (
      isComplexType(dtype) && (iter.tensor_base(0).is_conj() != iter.tensor_base(1).is_conj()));
  const bool requires_neg = (iter.tensor_base(0).is_neg() != iter.tensor_base(1).is_neg());

  auto strides_out = iter.strides(0);
  auto strides_in = iter.strides(1);
  if (dtype == iter.dtype(1)) {
    copy_same_dtype(iter, requires_conj, requires_neg);
  } else if (!requires_conj && ((iter.dtype(1) == kBFloat16 && iter.dtype(0) == kFloat &&
     sizeof(float) == strides_out[0] && (sizeof(BFloat16) == strides_in[0] || strides_in[0] == 0)) ||
    (iter.dtype(1) == kFloat && iter.dtype(0) == kBFloat16 &&
    sizeof(BFloat16) == strides_out[0] && (sizeof(float) == strides_in[0] || strides_in[0] == 0)))) {
    float_bfloat16_copy_kernel(iter, requires_neg);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::ComplexHalf, ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
      using dest_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::ComplexHalf, ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, iter.dtype(1), "copy_", [&] {
        if (iter.has_contiguous_first_dim()) {
          TORCH_INTERNAL_ASSERT(iter.ninputs() == 1);
          TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

          iter.for_each([](char **data, const int64_t *strides, int64_t size) {
            auto src = reinterpret_cast<const scalar_t*>(data[1]);
            auto dst = reinterpret_cast<dest_t*>(data[0]);
            at::vec::convert(src, dst, size);
          });
        } else {
          cpu_kernel(iter, [](scalar_t x) -> dest_t {
            return c10::convert<dest_t>(x);
          });
        }
      });
    });

    if (requires_conj || requires_neg) {
      // This inplace "copy" will perform any missing neg or conj operations
      auto self = iter.tensor_base(0);
      auto iter = TensorIterator::unary_op(self, self);
      copy_same_dtype(iter, requires_conj, requires_neg);
    }
  }
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(copy_stub, &copy_kernel);

} // namespace at::native
