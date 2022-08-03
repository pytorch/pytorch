#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace {
const auto exp_modified_bessel_k_1_string = modified_bessel_i_1_string + jiterator_stringify(
    template<typename T>
    T exp_modified_bessel_k_1_forward(T x) {
        static const T A[] = {
                -7.02386347938628759343e-18,
                -2.42744985051936593393e-15,
                -6.66690169419932900609e-13,
                -1.41148839263352776110e-10,
                -2.21338763073472585583e-08,
                -2.43340614156596823496e-06,
                -1.73028895751305206302e-04,
                -6.97572385963986435018e-03,
                -1.22611180822657148235e-01,
                -3.53155960776544875667e-01,
                +1.52530022733894777053e+00,
        };

        static const T B[] = {
                -5.75674448366501715755e-18,
                +1.79405087314755922667e-17,
                -5.68946255844285935196e-17,
                +1.83809354436663880070e-16,
                -6.05704724837331885336e-16,
                +2.03870316562433424052e-15,
                -7.01983709041831346144e-15,
                +2.47715442448130437068e-14,
                -8.97670518232499435011e-14,
                +3.34841966607842919884e-13,
                -1.28917396095102890680e-12,
                +5.13963967348173025100e-12,
                -2.12996783842756842877e-11,
                +9.21831518760500529508e-11,
                -4.19035475934189648750e-10,
                +2.01504975519703286596e-09,
                -1.03457624656780970260e-08,
                +5.74108412545004946722e-08,
                -3.50196060308781257119e-07,
                +2.40648494783721712015e-06,
                -1.93619797416608296024e-05,
                +1.95215518471351631108e-04,
                -2.85781685962277938680e-03,
                +1.03923736576817238437e-01,
                +2.72062619048444266945e+00,
        };

        if (x == T(0.0)) {
            return INFINITY;
        }

        if (x < T(0.0)) {
            return NAN;
        }

        T p;
        T q = 0.0;

        if (x <= T(2.0)) {
            T a = A[0];

            for (uint8_t index = 1; index < 11; index++) {
                p = q;
                q = a;
                a = (x * x - T(2.0)) * q - p + A[index];
            }

            return (log(T(0.5) * x) * modified_bessel_i_1_forward(x) + T(0.5) * (a - p) / x) * exp(x);
        }

        T b = B[0];

        for (uint8_t index = 1; index < 25; index++) {
            p = q;
            q = b;
            b = (T(8.0) / x - T(2.0)) * q - p + B[index];
        }

        return (T(0.5) * (b - p) / sqrt(x));
    } // T exp_modified_bessel_k_1_forward(T x)
); // exp_modified_bessel_k_1_string

const char exp_modified_bessel_k_1_name[] = "exp_modified_bessel_k_1_forward";

void exp_modified_bessel_k_1_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_1_cuda", [&]() {
    jitted_gpu_kernel<exp_modified_bessel_k_1_name, scalar_t, scalar_t, 1>(iterator, exp_modified_bessel_k_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_1_cuda", [&]() {
    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return exp_modified_bessel_k_1_forward(a);
    });
  });
#endif // AT_USE_JITERATOR()
} // void exp_modified_bessel_k_1_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(special_exp_modified_bessel_k_1_stub, &exp_modified_bessel_k_1_kernel_cuda);
} // namespace native
} // namespace at
