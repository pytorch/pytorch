#include <pybind11/pybind11.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace py = pybind11;

// Converts model to nnapi and serializes it for mobile
// Returns a dictionary string with one entry:
// Key: "NnapiModule"
// Value: a string of the nnapi module, saved for mobile
//
// method_compile_spec should contain an input Tensor with the following format:
// {"forward": {"inputs": Tensor}}
c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  // Import the python function for converting modules to nnapi
  py::gil_scoped_acquire gil;
  py::object pyModule = py::module_::import("torch.backends._nnapi.prepare");
  py::object pyMethod = pyModule.attr("convert_model_to_nnapi");

  // Wrap the c module in a RecursiveScriptModule and call the python conversion
  // function on it
  auto out =
      py::module::import("torch.jit._recursive").attr("wrap_cpp_module")(mod);
  out.attr("eval")();
  // TODO: throw exception if compile_spec doesn't contain inputs
  torch::Tensor inp =
      method_compile_spec.at("forward").toGenericDict().at("inputs").toTensor();
  auto nnapi_pyModel = pyMethod(out, inp);

  // Cast the returned py object and save it for mobile
  std::stringstream ss;
  auto nnapi_model = py::cast<torch::jit::Module>(nnapi_pyModel.attr("_c"));
  nnapi_model._save_for_mobile(ss);

  c10::Dict<c10::IValue, c10::IValue> dict(
      c10::StringType::get(), c10::StringType::get());
  dict.insert("NnapiModule", ss.str());
  return dict;
}

constexpr auto backend_name = "nnapi";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto pre_reg =
    torch::jit::backend_preprocess_register(backend_name, preprocess);
