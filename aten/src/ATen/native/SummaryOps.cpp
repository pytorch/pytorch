// Returns the frequency of elements of input non-negative integer tensor.
#include <ATen/native/SummaryOps.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

#include <tuple>

namespace at { namespace native {

DEFINE_DISPATCH(histc_stub);

///////////////// bincount /////////////////
namespace {

template <typename input_t, typename weights_t>
Tensor _bincount_cpu_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  if (self.dim() == 1 && self.numel() == 0) {
    return native::zeros({minlength}, kLong);
  }
  if (self.dim() != 1 || *self.min().data_ptr<input_t>() < 0) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.size(0) != self.size(0)) {
    AT_ERROR("input and weights should have the same length");
  }

  Tensor output;
  int64_t nbins = static_cast<int64_t>(*self.max().data_ptr<input_t>()) + 1L;
  nbins = std::max(nbins, minlength); // at least minlength # of bins

  const input_t* self_p = self.data_ptr<input_t>();
  if (has_weights) {
    output = native::zeros({nbins}, weights.options());
    weights_t* output_p = output.data_ptr<weights_t>();
    const weights_t* weights_p = weights.data_ptr<weights_t>();
    for (int64_t i = 0; i < self.size(0); i++) {
      output_p[self_p[i]] += weights_p[i];
    }
  } else {
    output = native::zeros({nbins}, kLong);
    int64_t* output_p = output.data_ptr<int64_t>();
    for (int64_t i = 0; i < self.size(0); i++) {
      output_p[self_p[i]] += 1L;
    }
  }
  return output;
}
} // namespace

Tensor
_bincount_cpu(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_cpu", [&] {
    const auto scalar = weights.scalar_type();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cpu_template<scalar_t, float>(self.contiguous(), weights.contiguous(), minlength);
    return _bincount_cpu_template<scalar_t, double>(
        self.contiguous(), weights.contiguous().to(kDouble), minlength);
  });
}

Tensor& histc_out(Tensor& hist, const Tensor &self, int64_t nbins, Scalar minvalue, Scalar maxvalue) {
  TORCH_CHECK(nbins > 0, "histc: bins must be > 0");
  TORCH_CHECK(at::isIntegralType(hist.scalar_type()), "hist only supports integral-point dtypes, hist got: ", hist.scalar_type());

  hist.resize_({nbins});
  hist.zero_();

  auto iter = at::TensorIterator();
  iter.add_input(self);
  iter.build();
  TORCH_CHECK(iter.device_type() == at::kCPU, "Native histc only supports CPU");
  histc_stub(iter.device_type(), iter, hist, nbins, minvalue, maxvalue);
  return hist;
}

Tensor histc(const Tensor& self, int64_t nbins, Scalar minvalue, Scalar maxvalue) {
  Tensor hist = at::empty({0}, self.options().dtype(kLong));
  at::histc_out(hist, self, nbins, minvalue, maxvalue);
  return hist;
}

}} // namespace at::native
