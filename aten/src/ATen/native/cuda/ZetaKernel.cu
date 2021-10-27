#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>

namespace at { namespace native {
namespace {

/*
 * This function is derived from the implementation of the zeta function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */
// TODO: do we want this as a string or a resource string or ... ?
#define stringify(...) std::string(#__VA_ARGS__);
const auto zeta_string = stringify(
  template <typename scalar_t>
  scalar_t zeta(scalar_t x, scalar_t q) {
    const scalar_t MACHEP = scalar_t{1.11022302462515654042E-16};
    constexpr scalar_t zero = scalar_t{0.0};
    constexpr scalar_t half = scalar_t{0.5};
    constexpr scalar_t one = scalar_t{1.0};
    static const scalar_t A[] = {
        12.0,
        -720.0,
        30240.0,
        -1209600.0,
        47900160.0,
        -1.8924375803183791606e9, /*1.307674368e12/691*/
        7.47242496e10,
        -2.950130727918164224e12, /*1.067062284288e16/3617*/
        1.1646782814350067249e14, /*5.109094217170944e18/43867*/
        -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
        1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
        -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
    };

    int i = 0;
    scalar_t a, b, k, s, t, w;
    if (x == one) {
      return __int_as_float(0x7f800000);
    }

    if (x < one) {
      return __int_as_float(0x7fffffff);
    }

    if (q <= zero) {
      if (q == ::floor(q)) {
        return __int_as_float(0x7f800000);
      }
      if (x != ::floor(x)) {
        return __int_as_float(0x7fffffff);
      }
    }

    s = ::pow(q, -x);
    a = q;
    i = 0;
    b = zero;
    while ((i < 9) || (a <= scalar_t{9.0})) {
      i += 1;
      a += one;
      b = ::pow(a, -x);
      s += b;
      if ((-MACHEP * s < b) && (b < MACHEP * s)) {
        return static_cast<scalar_t>(s);
      }
    };

    w = a;
    s += b * w / (x - one);
    s -= half * b;
    a = one;
    k = zero;
    for (int i = 0; i < 12; i++) {
      a *= x + k;
      b /= w;
      t = a * b / A[i];
      s = s + t;
      t = ::fabs(t / s);
      if (t < MACHEP) {
        return static_cast<scalar_t>(s);
      }
      k += one;
      a *= x + k;
      b /= w;
      k += one;
    }
    return static_cast<scalar_t>(s);
  }
); // stringify
#undef stringify

const char zeta_name[] = "zeta";
void zeta_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "zeta_cuda", [&]() {
    jitted_gpu_kernel</*name=*/zeta_name,
                      /*return_dtype=*/ scalar_t,
                      /*common_dtype=*/ scalar_t,
                      /*arity=*/ 2>(iter, zeta_string);
  });
}

}  // namespace (anonymous)

REGISTER_DISPATCH(zeta_stub, &zeta_kernel_cuda);

}} // namespace at::native
