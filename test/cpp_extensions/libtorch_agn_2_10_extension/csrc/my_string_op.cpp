#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <string>
#include <string_view>

using torch::stable::Tensor;

std::tuple<std::vector<std::string>, int64_t> my_string_op(Tensor t, std::string_view accessor, std::string passthru) {
  int64_t res;
  if (accessor == "dim") {
    res = t.dim();
  } else if (accessor == "size") {
    res = t.size(0);
  } else if (accessor == "stride") {
    res = t.stride(0);
  } else {
    STD_TORCH_CHECK(false, "Unsupported accessor value: ", std::string(accessor).c_str())
  }

  auto vec = std::vector<std::string>({std::string(accessor), std::to_string(res), passthru});
  return std::make_tuple(vec, res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_string_op(Tensor t, str accessor, str passthru) -> (str[], int)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_string_op", TORCH_BOX(&my_string_op));
}
