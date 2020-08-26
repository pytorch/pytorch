#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

struct PoolingParams {
  int64_t NB;
  int64_t NC;

  int64_t IH;
  int64_t IW;
  int64_t OH;
  int64_t OW;

  int64_t KH;
  int64_t SI;
  int64_t PI;
  int64_t DI;

  int64_t KW;
  int64_t SJ;
  int64_t PJ;
  int64_t DJ;
};

using pooling_fn = void (*)(Tensor&, const Tensor&, const PoolingParams&);

DECLARE_DISPATCH(pooling_fn, max_pool2d_stub);

} // namespace native
} // namespace at