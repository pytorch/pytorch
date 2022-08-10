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
const auto modified_bessel_k_0_string = modified_bessel_i_0_string + jiterator_stringify(
    template<typename T>
    T modified_bessel_k_0(T x) {
        static const T A[] = {
                +1.37446543561352307156e-16,
                +4.25981614279661018399e-14,
                +1.03496952576338420167e-11,
                +1.90451637722020886025e-09,
                +2.53479107902614945675e-07,
                +2.28621210311945178607e-05,
                +1.26461541144692592338e-03,
                +3.59799365153615016266e-02,
                +3.44289899924628486886e-01,
                -5.35327393233902768720e-01,
        };

        static const T B[] = {
                +5.30043377268626276149e-18,
                -1.64758043015242134646e-17,
                +5.21039150503902756861e-17,
                -1.67823109680541210385e-16,
                +5.51205597852431940784e-16,
                -1.84859337734377901440e-15,
                +6.34007647740507060557e-15,
                -2.22751332699166985548e-14,
                +8.03289077536357521100e-14,
                -2.98009692317273043925e-13,
                +1.14034058820847496303e-12,
                -4.51459788337394416547e-12,
                +1.85594911495471785253e-11,
                -7.95748924447710747776e-11,
                +3.57739728140030116597e-10,
                -1.69753450938905987466e-09,
                +8.57403401741422608519e-09,
                -4.66048989768794782956e-08,
                +2.76681363944501510342e-07,
                -1.83175552271911948767e-06,
                +1.39498137188764993662e-05,
                -1.28495495816278026384e-04,
                +1.56988388573005337491e-03,
                -3.14481013119645005427e-02,
                +2.44030308206595545468e+00,
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

            for (uint8_t index = 1; index < 10; index++) {
                p = q;
                q = a;
                a = (x * x - T(2.0)) * q - p + A[index];
            }

            return T(0.5) * (a - p) - log(0.5 * x) * modified_bessel_i_0(x);
        }

        T b = B[0];

        for (uint8_t index = 1; index < 25; index++) {
            p = q;
            q = b;
            b = (T(8.0) / x - T(2.0)) * q - p + B[index];
        }

        return exp(-x) * (T(0.5) * (b - p)) / sqrt(x);
    } // modified_bessel_k_0(T x)
); // modified_bessel_k_0_string

const char modified_bessel_k_0_name[] = "modified_bessel_k_0";

void modified_bessel_k_0_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cuda", [&]() {
    jitted_gpu_kernel<modified_bessel_k_0_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_k_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cuda", [&]() {
    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t x) -> scalar_t {
      return x;
    });
  });
#endif // AT_USE_JITERATOR()
} // void modified_bessel_k_0_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(special_modified_bessel_k_0_stub, &modified_bessel_k_0_kernel_cuda);
} // namespace native
} // namespace at
