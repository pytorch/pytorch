#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  AT_ERROR("mkldnn_softmax: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

namespace {
// TODO: move this to ideep
struct ideep_softmax_forward
    : public ideep::softmax_forward,
      public ideep::utils::computation_cache<ideep_softmax_forward> {
  template <typename... Ts>
  ideep_softmax_forward(
      const ideep::tensor::descriptor& src_desc,
      const ideep::tensor::descriptor& dst_desc,
      Ts&&... args) {
    init(src_desc, dst_desc, std::forward<Ts>(args)...);
  }

  template <class alloc>
  static void compute(
      const ideep::tensor& src,
      ideep::tensor& dst,
      int softmax_axis) {
    if (dst.get_descriptor() != src.get_descriptor()) {
      dst.reinit<alloc, ideep_softmax_forward>(src.get_descriptor());
    }
    ideep::key_t key;
    ideep::utils::create_key(
        key,
        src.get_data_type(),
        src.get_dims(),
        src.get_internal_format(),
        softmax_axis);
    fetch_or_create_m(
        comp, key, src.get_descriptor(), dst.get_descriptor(), softmax_axis);
    comp.execute(src, dst);
  }
};
} // namespace

Tensor mkldnn_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on Mkldnn");
  const int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
  ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  ideep_softmax_forward::compute<AllocForMKLDNN>(x, y, wrapped_dim);
  return new_with_itensor_mkldnn(std::move(y), self.options());
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
