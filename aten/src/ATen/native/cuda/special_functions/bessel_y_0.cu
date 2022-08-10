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
const auto bessel_y_0_string = bessel_j_0_string + jiterator_stringify(
    template<typename T>
    T bessel_y_0(T x) {
        static const T PP[] = {
                +7.96936729297347051624e-04,
                +8.28352392107440799803e-02,
                +1.23953371646414299388e+00,
                +5.44725003058768775090e+00,
                +8.74716500199817011941e+00,
                +5.30324038235394892183e+00,
                +9.99999999999999997821e-01,
        };

        static const T PQ[] = {
                +9.24408810558863637013e-04,
                +8.56288474354474431428e-02,
                +1.25352743901058953537e+00,
                +5.47097740330417105182e+00,
                +8.76190883237069594232e+00,
                +5.30605288235394617618e+00,
                +1.00000000000000000218e+00,
        };

        static const T QP[] = {
                -1.13663838898469149931e-02,
                -1.28252718670509318512e+00,
                -1.95539544257735972385e+01,
                -9.32060152123768231369e+01,
                -1.77681167980488050595e+02,
                -1.47077505154951170175e+02,
                -5.14105326766599330220e+01,
                -6.05014350600728481186e+00,
        };

        static const T QQ[] = {
                +6.43178256118178023184e+01,
                +8.56430025976980587198e+02,
                +3.88240183605401609683e+03,
                +7.24046774195652478189e+03,
                +5.93072701187316984827e+03,
                +2.06209331660327847417e+03,
                +2.42005740240291393179e+02,
        };

        static const T YP[] = {
                +1.55924367855235737965e+04,
                -1.46639295903971606143e+07,
                +5.43526477051876500413e+09,
                -9.82136065717911466409e+11,
                +8.75906394395366999549e+13,
                -3.46628303384729719441e+15,
                +4.42733268572569800351e+16,
                -1.84950800436986690637e+16,
        };

        static const T YQ[] = {
                +1.04128353664259848412e+03,
                +6.26107330137134956842e+05,
                +2.68919633393814121987e+08,
                +8.64002487103935000337e+10,
                +2.02979612750105546709e+13,
                +3.17157752842975028269e+15,
                +2.50596256172653059228e+17,
        };

        if (x <= T(5.0)) {
            if (x == T(0.0)) {
                return NEG_INFINITY;
            }

            if (x < T(0.0)) {
                NAN;
            }

            T yp = 0.0;

            for (uint8_t index = 0; index <= 7; index++) {
                yp = yp * (x * x) + YP[index];
            }

            T yq = 0.0;

            for (uint8_t index = 0; index <= 6; index++) {
                yq = yq * (x * x) + YQ[index];
            }

            return yp / yq + (T(0.636619772367581343075535053490057448) * log(x) * bessel_j_0(x));
        }

        T pp = 0.0;

        for (uint8_t index = 0; index <= 6; index++) {
            pp = pp * (T(25.0) / (x * x)) + PP[index];
        }

        T pq = 0.0;

        for (uint8_t index = 0; index <= 6; index++) {
            pq = pq * (T(25.0) / (x * x)) + PQ[index];
        }

        T qp = 0.0;

        for (uint8_t index = 0; index <= 7; index++) {
            qp = qp * (T(25.0) / (x * x)) + QP[index];
        }

        T qq = 0.0;

        for (uint8_t index = 0; index <= 6; index++) {
            qq = qq * (T(25.0) / (x * x)) + QQ[index];
        }

        return (pp / pq * sin(x - T(0.785398163397448309615660845819875721)) + T(5.0) / x * (qp / qq) * cos(x - T(0.785398163397448309615660845819875721))) * T(0.797884560802865355879892119868763737) / sqrt(x);
    } // bessel_y_0(T x)
); // bessel_y_0_string

const char bessel_y_0_name[] = "bessel_y_0";

void bessel_y_0_kernel_cuda(TensorIteratorBase &iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_0_cuda", [&]() {
    jitted_gpu_kernel<bessel_y_0_name, scalar_t, scalar_t, 1>(iterator, bessel_y_0_string);
  });
#else
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_0_cuda", [&]() {
    gpu_kernel(iterator, []GPU_LAMBDA(scalar_t x) -> scalar_t {
      return x;
    });
  });
#endif // AT_USE_JITERATOR()
} // void bessel_y_0_kernel_cuda(TensorIteratorBase &iterator)
} // namespace (anonymous)
REGISTER_DISPATCH(special_bessel_y_0_stub, &bessel_y_0_kernel_cuda);
} // namespace native
} // namespace at
