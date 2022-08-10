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
const auto modified_bessel_i_1_string = jiterator_stringify(
    template<typename T>
    T modified_bessel_i_1(T x) {
        static const T A[] = {
                +2.77791411276104639959e-18,
                -2.11142121435816608115e-17,
                +1.55363195773620046921e-16,
                -1.10559694773538630805e-15,
                +7.60068429473540693410e-15,
                -5.04218550472791168711e-14,
                +3.22379336594557470981e-13,
                -1.98397439776494371520e-12,
                +1.17361862988909016308e-11,
                -6.66348972350202774223e-11,
                +3.62559028155211703701e-10,
                -1.88724975172282928790e-09,
                +9.38153738649577178388e-09,
                -4.44505912879632808065e-08,
                +2.00329475355213526229e-07,
                -8.56872026469545474066e-07,
                +3.47025130813767847674e-06,
                -1.32731636560394358279e-05,
                +4.78156510755005422638e-05,
                -1.61760815825896745588e-04,
                +5.12285956168575772895e-04,
                -1.51357245063125314899e-03,
                +4.15642294431288815669e-03,
                -1.05640848946261981558e-02,
                +2.47264490306265168283e-02,
                -5.29459812080949914269e-02,
                +1.02643658689847095384e-01,
                -1.76416518357834055153e-01,
                +2.52587186443633654823e-01,
        };

        static const T B[] = {
                +7.51729631084210481353e-18,
                +4.41434832307170791151e-18,
                -4.65030536848935832153e-17,
                -3.20952592199342395980e-17,
                +2.96262899764595013876e-16,
                +3.30820231092092828324e-16,
                -1.88035477551078244854e-15,
                -3.81440307243700780478e-15,
                +1.04202769841288027642e-14,
                +4.27244001671195135429e-14,
                -2.10154184277266431302e-14,
                -4.08355111109219731823e-13,
                -7.19855177624590851209e-13,
                +2.03562854414708950722e-12,
                +1.41258074366137813316e-11,
                +3.25260358301548823856e-11,
                -1.89749581235054123450e-11,
                -5.58974346219658380687e-10,
                -3.83538038596423702205e-09,
                -2.63146884688951950684e-08,
                -2.51223623787020892529e-07,
                -3.88256480887769039346e-06,
                -1.10588938762623716291e-04,
                -9.76109749136146840777e-03,
                +7.78576235018280120474e-01,
        };

        T p;
        T q = 0.0;

        if (abs(x) <= T(8.0)) {
            T a = A[0];

            for (uint8_t index = 1; index < 29; index++) {
                p = q;
                q = a;
                a = ((abs(x) / T(2.0)) - T(2.0)) * q - p + A[index];
            }

            if (x < T(0.0)) {
                return -(T(0.5) * (a - p) * abs(x) * exp(abs(x)));
            }

            return T(0.5) * (a - p) * abs(x) * exp(abs(x));
        }

        T b = B[0];

        for (uint8_t index = 1; index < 25; index++) {
            p = q;
            q = b;
            b = (T(32.0) / abs(x) - T(2.0)) * q - p + B[index];
        }

        if (x < T(0.0)) {
            return -(exp(abs(x)) * (T(0.5) * (b - p)) / sqrt(abs(x)));
        }

        return exp(abs(x)) * (T(0.5) * (b - p)) / sqrt(abs(x));
    } // modified_bessel_i_1(T x)
); // modified_bessel_i_1_string

const char modified_bessel_i_1_name[] = "modified_bessel_i_1";

void modified_bessel_i_1_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_1_cuda", [&]() {
    jitted_gpu_kernel<modified_bessel_i_1_name, scalar_t, scalar_t, 1>(iterator, modified_bessel_i_1_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_1_cuda", [&]() {
    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t x) -> scalar_t {
      return x;
    });
  });
#endif // AT_USE_JITERATOR()
} // void modified_bessel_i_1_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(special_modified_bessel_i_1_stub, &modified_bessel_i_1_kernel_cuda);
} // namespace native
} // namespace at
