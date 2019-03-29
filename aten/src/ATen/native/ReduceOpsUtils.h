#pragma once

namespace at { namespace native {

static Tensor &_dimreduce_setup(Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntArrayRef self_sizes = self.sizes();
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  result.resize_(result_sizes);
  return result;
}

static bool _dimreduce_return_trivial(Tensor &result, const Tensor &self,
                                      Scalar ident, int64_t dim, bool keepdim) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }
  // Return identity
  if (self.numel() == 0) {
    _dimreduce_setup(result, self, dim);
    result.fill_(ident);
    if (!keepdim) result.squeeze_(dim);
    return true;
  }
  return false;
}

static bool _dimreduce_return_trivial_no_ident(Tensor &result, const Tensor &self,
                                               int64_t dim, bool keepdim, const char *fn_name) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }

  if (self.numel() == 0) {
    AT_ERROR("cannot perform reduction function ", fn_name,
             " on tensor with no elements because the operation does not have an identity");
  }
  return false;
}

static c10::optional<Tensor> _allreduce_return_trivial(
    const Tensor& self,
    Scalar ident) {
  // Return identity
  if (self.numel() == 0) {
    return at::scalar_tensor(ident, self.options());
  }
  return c10::nullopt;
}
}}  // at::native
