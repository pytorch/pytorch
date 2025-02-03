#include <torch/csrc/utils/pybind.h>

#ifdef USE_CUSPARSELT
#include <ATen/native/sparse/cuda/cuSPARSELtOps.h>

namespace {

size_t getVersionInt() {
  return CUSPARSELT_VERSION;
}

std::tuple<int64_t, int64_t, bool, int64_t> mmSearch(
    const at::Tensor& compressed_A,
    const at::Tensor& dense_B,
    const std::optional<at::Tensor>& bias_opt,
    const std::optional<at::Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result) {
  int alg_id_int = 0;
  int split_k = 1;
  bool split_k_one_kernel = true;
  auto result = at::native::_cslt_sparse_mm_impl(
      compressed_A,
      dense_B,
      bias_opt,
      alpha_opt,
      out_dtype_opt,
      transpose_result,
      alg_id_int,
      split_k,
      split_k_one_kernel,
      true);
  return {
      (int64_t)std::get<1>(result),
      (int64_t)std::get<2>(result),
      (bool)std::get<3>(result),
      (int64_t)std::get<4>(result)};
}

} // namespace

namespace torch::cuda::shared {

void initCusparseltBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto cusparselt = m.def_submodule("_cusparselt", "libcusparselt.so bindings");
  cusparselt.def("getVersionInt", getVersionInt);
  cusparselt.def("mm_search", mmSearch);
}

} // namespace torch::cuda::shared
#endif
