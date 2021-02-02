// ${generated_comment}

#include <ATen/Functions.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/hacky_wrapper_for_legacy_signatures.h>

${static_dispatch_extra_headers}

namespace at {

Tensor var(const Tensor& self, int dim) {
  return at::var(self, IntArrayRef{dim});
}

std::tuple<Tensor,Tensor> var_mean(const Tensor& self, int dim) {
  return at::var_mean(self, IntArrayRef{dim});
}

Tensor std(const Tensor& self, int dim) {
  return at::std(self, IntArrayRef{dim});
}

std::tuple<Tensor,Tensor> std_mean(const Tensor& self, int dim) {
  return at::std_mean(self, IntArrayRef{dim});
}

${function_definitions}

}
