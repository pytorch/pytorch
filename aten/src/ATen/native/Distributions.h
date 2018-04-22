#ifdef __CUDA_ARCH__
#include <nvfunctional>
#endif

namespace at {
namespace native {

#ifdef __CUDA_ARCH__
#define nvfunction_or_function nvstd::function
#define deviceforcuda __device__
#else
#define nvfunction_or_function std::function
#define deviceforcuda
#endif

template<typename precision_t>
struct BaseSampler {
  nvfunction_or_function<precision_t(void)> sampler;
  deviceforcuda BaseSampler(nvfunction_or_function<precision_t(void)> sampler): sampler(sampler) {}
  deviceforcuda precision_t sample() {
    return sampler();
  }
};

// The function `sample_gamma` is
// is adapted from Numpy's distributions.c implementation.
// It is MIT licensed, so here is the copyright:

/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

template<typename precision_t>
#ifdef __CUDA_ARCH__
__device__
#endif
precision_t sample_gamma(precision_t alpha, BaseSampler<precision_t>& standard_uniform, BaseSampler<precision_t>& standard_normal) {
  precision_t scale = 1.0;

  // Boost alpha for higher acceptance probability.
  if (alpha < 1.0) {
    scale *= ::pow(1 - standard_uniform.sample(), 1.0 / alpha);
    alpha += 1.0;
  }

  // This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
  // doi:10.1145/358407.358414
  const precision_t d = alpha - 1.0 / 3.0;
  const precision_t c = 1.0 / ::sqrt(9.0 * d);
  for (;;) {
    precision_t x, y;
    do {
      x = standard_normal.sample();
      y = 1.0 + c * x;
    } while (y <= 0);
    const precision_t v = y * y * y;
    const precision_t u = 1 - standard_uniform.sample();
    const precision_t xx = x * x;
    if (u < 1.0 - 0.0331 * xx * xx)
      return scale * d * v;
    if (::log(u) < 0.5 * xx + d * (1.0 - v + ::log(v)))
      return scale * d * v;
  }
}

}} // at::native
