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
const auto modified_bessel_i_0_string = jiterator_stringify(
    template<typename T>
    T modified_bessel_i_0_forward(T x) {
        static const T A[] = {
                -4.41534164647933937950e-18,
                +3.33079451882223809783e-17,
                -2.43127984654795469359e-16,
                +1.71539128555513303061e-15,
                -1.16853328779934516808e-14,
                +7.67618549860493561688e-14,
                -4.85644678311192946090e-13,
                +2.95505266312963983461e-12,
                -1.72682629144155570723e-11,
                +9.67580903537323691224e-11,
                -5.18979560163526290666e-10,
                +2.65982372468238665035e-09,
                -1.30002500998624804212e-08,
                +6.04699502254191894932e-08,
                -2.67079385394061173391e-07,
                +1.11738753912010371815e-06,
                -4.41673835845875056359e-06,
                +1.64484480707288970893e-05,
                -5.75419501008210370398e-05,
                +1.88502885095841655729e-04,
                -5.76375574538582365885e-04,
                +1.63947561694133579842e-03,
                -4.32430999505057594430e-03,
                +1.05464603945949983183e-02,
                -2.37374148058994688156e-02,
                +4.93052842396707084878e-02,
                -9.49010970480476444210e-02,
                +1.71620901522208775349e-01,
                -3.04682672343198398683e-01,
                +6.76795274409476084995e-01,
        };

        static const T B[] = {
                -7.23318048787475395456e-18,
                -4.83050448594418207126e-18,
                +4.46562142029675999901e-17,
                +3.46122286769746109310e-17,
                -2.82762398051658348494e-16,
                -3.42548561967721913462e-16,
                +1.77256013305652638360e-15,
                +3.81168066935262242075e-15,
                -9.55484669882830764870e-15,
                -4.15056934728722208663e-14,
                +1.54008621752140982691e-14,
                +3.85277838274214270114e-13,
                +7.18012445138366623367e-13,
                -1.79417853150680611778e-12,
                -1.32158118404477131188e-11,
                -3.14991652796324136454e-11,
                +1.18891471078464383424e-11,
                +4.94060238822496958910e-10,
                +3.39623202570838634515e-09,
                +2.26666899049817806459e-08,
                +2.04891858946906374183e-07,
                +2.89137052083475648297e-06,
                +6.88975834691682398426e-05,
                +3.36911647825569408990e-03,
                +8.04490411014108831608e-01,
        };

        T p;
        T q = 0.0;

        if (abs(x) <= T(8.0)) {
            T a = A[0];

            for (uint8_t index = 1; index < 30; index++) {
                p = q;
                q = a;
                a = ((abs(x) / T(2.0)) - T(2.0)) * q - p + A[index];
            }

            return exp(abs(x)) * (T(0.5) * (a - p));
        }

        T b = B[0];

        for (uint8_t index = 1; index < 25; index++) {
            p = q;
            q = b;
            b = (T(32.0) / abs(x) - T(2.0)) * q - p + B[index];
        }

        return exp(abs(x)) * (T(0.5) * (b - p)) / sqrt(abs(x));
    } // modified_bessel_i_0_forward(T x)
); // modified_bessel_i_0_string

const char modified_bessel_i_0_name[] = "modified_bessel_i_0_forward";

void modified_bessel_i_0_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_0_cuda", [&]() {
    jitted_gpu_kernel<modified_bessel_i_0_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_i_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_0_cuda", [&]() {
    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return modified_bessel_i_0_forward(a);
    });
  });
#endif // AT_USE_JITERATOR()
} // void modified_bessel_i_0_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(special_modified_bessel_i_0_stub, &modified_bessel_i_0_kernel_cuda);
} // namespace native
} // namespace at
