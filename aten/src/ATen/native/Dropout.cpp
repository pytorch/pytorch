#include "ATen/ATen.h"
#include "ATen/Error.h"
#include "ATen/NativeFunctions.h"
#include "ATen/CPUGenerator.h"
#include "ATen/CheckGenerator.h"
#include "ATen/Generator.h"

namespace at {
namespace native {

Tensor dropout(const Tensor& self, double p, bool featurewise, bool train, Generator *gen) {
  AT_CHECK(0 <= p && p <= 1, "dropout() expects 0 <= p <= 1, but got p = ", p);
  if (train) {
    if (p == 1) {
      return at::zeros_like(self);
    } else {
      auto keep_p = 1. - p;
      if (featurewise) {
        auto dim = self.dim();
        AT_CHECK(dim > 2,
                 "feature_dropout() expects input to have at least 3 "
                 "dimensions, but got input with size ", self.sizes());
        auto noise_shape = self.sizes().vec();
        for (int64_t i = 2; i < dim; i++) {
          noise_shape[i] = 1;
        }
        auto noise = at::empty(noise_shape, self.type()).bernoulli_(keep_p, gen).div_(keep_p);
        return self * noise;
      } else {
        auto noise = at::empty_like(self).bernoulli_(keep_p, gen).div_(keep_p);
        return self * noise;
      }
    }
  } else {
    return self.clone();
  }
}

Tensor& dropout_(Tensor& self, double p, bool featurewise, bool train, Generator *gen) {
  AT_CHECK(0 <= p && p <= 1, "dropout() expects 0 <= p <= 1, but got p = ", p);
  if (train) {
    if (p == 1) {
      return self.zero_();
    } else {
      auto keep_p = 1. - p;
      if (featurewise) {
        auto dim = self.dim();
        AT_CHECK(dim > 2,
                 "feature_dropout() expects input to have at least 3 "
                 "dimensions, but got input with size ", self.sizes());
        auto noise_shape = self.sizes().vec();
        for (int64_t i = 2; i < dim; i++) {
          noise_shape[i] = 1;
        }
        auto noise = at::empty(noise_shape, self.type()).bernoulli_(keep_p, gen).div_(keep_p);
        return self.mul_(noise);
      } else {
        auto noise = at::empty_like(self).bernoulli_(keep_p, gen).div_(keep_p);
        return self.mul_(noise);
      }
    }
  } else {
    return self;
  }
}



}
}
