#include "ATen/Config.h"
#include <functional>
#if AT_CUDA_ENABLED()
#include <nvfunctional>
#endif

namespace at {
namespace native {
namespace dist {

  // this wraps sampling primitives to expose a common interface
  template<typename precision_t>
  struct baseSampler {
#if AT_CUDA_ENABLED()
    nvstd::function<precision_t(void)> sampler;
    __device__ baseSampler(nvstd::function<precision_t(void)> sampler): sampler(sampler) {}
    __device__ precision_t sample() {
      return sampler();
    }
#else
    std::function<precision_t(void)> sampler;
    baseSampler(std::function<precision_t(void)> sampler): sampler(sampler) {}
    precision_t sample() {
      return sampler();
    }
#endif
  };

  template<typename precision_t>
#if AT_CUDA_ENABLED()
  __host__ __device__
#endif
  precision_t sample_gamma(precision_t alpha, baseSampler<precision_t>& standard_uniform, baseSampler<precision_t>& standard_normal) {
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
} // dist
} // native
} // at
