// This file is intended to test (const) std::string& and const std::string_view& arguments with TORCH_BOX
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>

#include <string>
#include <string_view>

using torch::stable::Tensor;

// Helper function to process accessor
static int64_t process_accessor(Tensor t, std::string_view accessor) {
  if (accessor == "dim") {
    return t.dim();
  } else if (accessor == "size") {
    return t.size(0);
  } else if (accessor == "stride") {
    return t.stride(0);
  } else {
    STD_TORCH_CHECK(false, "Unsupported accessor value: ", std::string(accessor).c_str())
  }
}

// Test const std::string&
std::tuple<std::vector<std::string>, int64_t> my_string_op_const_string_ref(
    Tensor t,
    const std::string& accessor,
    const std::string& passthru) {
  int64_t res = process_accessor(t, accessor);
  auto vec = std::vector<std::string>({accessor, std::to_string(res), passthru});
  return std::make_tuple(vec, res);
}

// Test const std::string_view&
std::tuple<std::vector<std::string>, int64_t> my_string_op_const_string_view_ref(
    Tensor t,
    const std::string_view& accessor,
    const std::string_view& passthru) {
  int64_t res = process_accessor(t, accessor);
  auto vec = std::vector<std::string>({std::string(accessor), std::to_string(res), std::string(passthru)});
  return std::make_tuple(vec, res);
}

// Test std::string& (non-const)
std::tuple<std::vector<std::string>, int64_t> my_string_op_string_ref(
    Tensor t,
    std::string& accessor,
    std::string& passthru) {
  int64_t res = process_accessor(t, accessor);
  auto vec = std::vector<std::string>({accessor, std::to_string(res), passthru});
  return std::make_tuple(vec, res);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_string_op_const_string_ref(Tensor t, str accessor, str passthru) -> (str[], int)");
  m.def("my_string_op_const_string_view_ref(Tensor t, str accessor, str passthru) -> (str[], int)");
  m.def("my_string_op_string_ref(Tensor t, str accessor, str passthru) -> (str[], int)");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_string_op_const_string_ref", TORCH_BOX(&my_string_op_const_string_ref));
  m.impl("my_string_op_const_string_view_ref", TORCH_BOX(&my_string_op_const_string_view_ref));
  m.impl("my_string_op_string_ref", TORCH_BOX(&my_string_op_string_ref));
}
