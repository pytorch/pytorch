#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/RangeFactories.h>
#include <ATen/native/RangeUtils.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <c10/util/irange.h>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/linspace.h>
#include <ATen/ops/logspace.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/linspace_native.h>
#include <ATen/ops/logspace_native.h>
#include <ATen/ops/range_native.h>
#endif

namespace at::native {

Tensor& linspace_out(const Tensor& start, const Tensor& end, int64_t steps, Tensor& result) {
  TORCH_CHECK(start.dim() == 0 && end.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s) and end with ", end.dim()," dimension(s).");
  return at::linspace_out(result, start.item(), end.item(), steps);
}

Tensor& linspace_out(const Tensor& start, const Scalar& end, int64_t steps, Tensor& result) {
  TORCH_CHECK(start.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s).");
  return at::linspace_out(result, start.item(), end, steps);
}

Tensor& linspace_out(const Scalar& start, const Tensor& end, int64_t steps, Tensor& result) {
  TORCH_CHECK(end.dim() == 0, "linspace only supports 0-dimensional start and end tensors, "
    "but got end with ", end.dim()," dimension(s).");
  return at::linspace_out(result, start, end.item(), steps);
}

Tensor& linspace_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  if (result.numel() != steps) {
    result.resize_({steps});
  }

  if (result.device() == kMeta) {
    return result;
  }

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    result.fill_(start);
  } else {
    Tensor r = result.is_contiguous() ? result : result.contiguous();
    auto iter = TensorIterator::borrowing_nullary_op(r);
    linspace_stub(iter.device_type(), iter, start, end, steps);
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  }

  return result;
}

Tensor& logspace_out(const Tensor& start, const Tensor& end, int64_t steps, double base, Tensor& result) {
  TORCH_CHECK(start.dim() == 0 && end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s) and end with ", end.dim()," dimension(s).");
  return at::logspace_out(result, start.item(), end.item(), steps, base);
}

Tensor& logspace_out(const Tensor& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  TORCH_CHECK(start.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got start with ", start.dim(), " dimension(s).");
  return at::logspace_out(result, start.item(), end, steps, base);
}

Tensor& logspace_out(const Scalar& start, const Tensor& end, int64_t steps, double base, Tensor& result) {
  TORCH_CHECK(end.dim() == 0, "logspace only supports 0-dimensional start and end tensors, "
    "but got end with ", end.dim()," dimension(s).");
  return at::logspace_out(result, start, end.item(), steps, base);
}

Tensor& logspace_out(const Scalar& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }

  if (result.device() == kMeta) {
    return result;
  }

  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    if (isComplexType(r.scalar_type())){
      r.fill_(std::pow(base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(std::pow(base, start.to<double>()));
    }
  } else if (isComplexType(r.scalar_type())) {
    AT_DISPATCH_COMPLEX_TYPES(r.scalar_type(), "logspace_cpu", [&]() {
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t *data_ptr = r.data_ptr<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      const int64_t halfway = steps / 2;
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        scalar_t is = static_cast<scalar_t>(p_begin);
        for (int64_t i = p_begin; i < p_end; ++i, is+=1) { //std::complex does not support ++operator
          if (i < halfway) {
            data_ptr[i] = std::pow(scalar_base, scalar_start + step*is);
          } else {
            data_ptr[i] = std::pow(scalar_base, scalar_end - (step * static_cast<scalar_t>(steps - i - 1)));
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, r.scalar_type(), "logspace_cpu", [&]() {
      double scalar_base = static_cast<double>(base); // will be autopromoted anyway
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t *data_ptr = r.data_ptr<scalar_t>();
      double step = static_cast<double>(scalar_end - scalar_start) / (steps - 1);
      const int64_t halfway = steps / 2;
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        for (const auto i : c10::irange(p_begin, p_end)) {
          if (i < halfway) {
            data_ptr[i] = std::pow(scalar_base, scalar_start + step*i);
          } else {
            data_ptr[i] = std::pow(scalar_base, scalar_end - step * (steps - i - 1));
          }
        }
      });
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  return result;
}

Tensor& range_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, result.scalar_type(), "range_cpu", [&]() {
    using accscalar_t = at::acc_type<scalar_t, false>;
    auto xstart = start.to<accscalar_t>();
    auto xend = end.to<accscalar_t>();
    auto xstep = step.to<accscalar_t>();

    TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
    TORCH_CHECK(std::isfinite(static_cast<double>(xstart)) &&
             std::isfinite(static_cast<double>(xend)),
             "unsupported range: ", xstart, " -> ", xend);
    TORCH_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
             "upper bound and lower bound inconsistent with step sign");
    int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
    if (result.numel() != size) {
      result.resize_({size});
    }

    if (result.device() == kMeta) {
      return;
    }

    Tensor r = result.is_contiguous() ? result : result.contiguous();
    scalar_t *data_ptr = r.data_ptr<scalar_t>();

    at::parallel_for(0, size, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
      accscalar_t is = p_begin;
      for (int64_t i = p_begin; i < p_end; ++i, ++is) {
        data_ptr[i] = xstart + is * xstep;
      }
    });
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  return result;
}

Tensor& range_out_no_step(const Scalar& start, const Scalar& end, Tensor& result) {
  return range_out(start, end, /*step = */ 1, result);
}

Tensor& arange_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, result.scalar_type(), "arange_cpu", [&]() {
    int64_t size = compute_arange_size<scalar_t>(start, end, step);
    int64_t numel = result.numel();

    if (numel != size) {
      if(numel > 0){
        TORCH_WARN("The number of elements in the out tensor of shape ", result.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
      }
      result.resize_({size});
    }

    if (result.device() == kMeta) {
      return;
    }

    Tensor r = result.is_contiguous() ? result : result.contiguous();
    auto iter = TensorIterator::borrowing_nullary_op(r);
    arange_stub(iter.device_type(), iter, start, size, step);
    if (!result.is_contiguous()) {
      result.copy_(r);
    }
  });

  return result;
}

DEFINE_DISPATCH(arange_stub);
DEFINE_DISPATCH(linspace_stub);

} // namespace at::native
