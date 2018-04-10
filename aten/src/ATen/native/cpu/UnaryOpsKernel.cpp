#include "ATen/native/cpu/UnaryOpsKernel.h"

#include <cmath>
#include <iostream>
#include "ATen/Dispatch.h"
#include "ATen/Parallel.h"
#include "ATen/cpu/vec256/vec256.h"
#include "ATen/native/cpu/CapabilityDispatch.h"

namespace at { namespace native { namespace {

using namespace vec256;

template <typename scalar_t, typename F>
static void unary_kernel(scalar_t* arr_out, const scalar_t* arr_in, int64_t size, F func) {
  using Vec = Vec256<scalar_t>;
  int64_t size_rounded = size - (size % Vec::size);
  int64_t k = 0;
  for (; k != size_rounded; k += Vec::size) {
    auto value = func(Vec::s_load(arr_in + k));
    value.store(arr_out + k);
  }
  auto leftover = size - k;
  if (leftover > 0) {
    Vec a;
    a.load_partial(arr_in + k, leftover);
    func(a).store_partial(arr_out + k, leftover);
  }
}

template <class scalar_t, class F>
static void parallel_apply(Tensor& result, const Tensor& self, F f) {
  internal::init_tbb_num_threads();

  static tbb::affinity_partitioner ap;

  auto arr_out = result.data<scalar_t>();
  auto arr_in = self.data<scalar_t>();
  int64_t size = self.numel();
  if (size < internal::TBB_GRAIN_SIZE) {
    unary_kernel(arr_out, arr_in, size, f);
  } else {
    tbb::parallel_for(
        tbb::blocked_range<int64_t>(0, size, internal::TBB_GRAIN_SIZE),
        [&](const tbb::blocked_range<int64_t>& r) {
          auto size = r.end() - r.begin();
          unary_kernel(arr_out + r.begin(), arr_in + r.begin(), size, f);
        },
        ap);
  }
}

static void abs_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_ALL_TYPES(self.type(), "abs", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.abs();
    });
  });
}

static void ceil_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "ceil", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.ceil();
    });
  });
}

static void cos_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "cos", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.cos();
    });
  });
}

static void exp_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "exp", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.exp();
    });
  });
}

static void floor_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "floor", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.floor();
    });
  });
}

static void log_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "log", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.log();
    });
  });
}

static void round_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "round", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.round();
    });
  });
}

static void sin_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sin", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.sin();
    });
  });
}

static void sqrt_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "sqrt", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.sqrt();
    });
  });
}

static void trunc_kernel(Tensor& result, const Tensor& self) {
  AT_DISPATCH_FLOATING_TYPES(self.type(), "trunc", [&] {
    parallel_apply<scalar_t>(result, self, [](const Vec256<scalar_t>& x) {
      return x.trunc();
    });
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(absImpl, &abs_kernel);
REGISTER_DISPATCH(ceilImpl, &ceil_kernel);
REGISTER_DISPATCH(cosImpl, &cos_kernel);
REGISTER_DISPATCH(expImpl, &exp_kernel);
REGISTER_DISPATCH(floorImpl, &floor_kernel);
REGISTER_DISPATCH(logImpl, &log_kernel);
REGISTER_DISPATCH(roundImpl, &round_kernel);
REGISTER_DISPATCH(sinImpl, &sin_kernel);
REGISTER_DISPATCH(sqrtImpl, &sqrt_kernel);
REGISTER_DISPATCH(truncImpl, &trunc_kernel);

}} // namespace at::native
