#include <pybind11/pybind11.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace py = pybind11;

// Converts model to Android NNAPI backend and serializes it for mobile
// Returns a dictionary with preprocessed items:
//    "shape_compute_module": torch::jit::Module,
//    "ser_model": torch::Tensor,
//    "weights": List[torch.Tensor],
//    "inp_mem_fmts": List[int],
//    "out_mem_fmts": List[int]
//
// method_compile_spec should contain a Tensor or
// Tensor List which bundles several input parameters:
// shape, dtype, quantization, and dimorder (NHWC/NCHW)
// For input shapes, use 0 for run/load time flexible input
//
// The compile_spec should include the format:
// {"forward": {"inputs": at::Tensor}}
// OR {"forward": {"inputs": c10::List<at::Tensor>}}
// Example input Tensor:
// torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
//
// In the future, preprocess will accept a dedicated object, NnapiArg
c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& method_compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  // Import the python function for processing modules to Android NNAPI backend
  py::gil_scoped_acquire gil;
  py::object pyModule = py::module_::import("torch.backends._nnapi.prepare");
  py::object pyMethod = pyModule.attr("process_for_nnapi");

  // Wrap the c module in a RecursiveScriptModule
  auto wrapped_mod =
      py::module::import("torch.jit._recursive").attr("wrap_cpp_module")(mod);
  wrapped_mod.attr("eval")();

  // Convert input to a Tensor or a python list of Tensors
  auto inp = method_compile_spec.at("forward").toGenericDict().at("inputs");
  py::list nnapi_processed;
  if (inp.isTensor()) {
    nnapi_processed = pyMethod(wrapped_mod, inp.toTensor());
  } else {
    py::list pyInp;
    for (torch::Tensor inpElem : inp.toTensorList()) {
      pyInp.append(inpElem);
    }
    nnapi_processed = pyMethod(wrapped_mod, pyInp);
  }

  c10::Dict<c10::IValue, c10::IValue> dict(
    c10::StringType::get(), c10::AnyType::get());

  // Cast processed items from python objects to C++ classes and add to dict
  dict.insert("ser_model", py::cast<torch::Tensor>(nnapi_processed[1]));
  // serialize shape_compute_module for mobile
  auto shape_compute_module = py::cast<torch::jit::Module>(nnapi_processed[0].attr("_c"));
  std::stringstream ss;
  shape_compute_module._save_for_mobile(ss);
  dict.insert("shape_compute_module", ss.str());
  // transform Python lists to c++ c10::List and add to dictionary
  c10::List<torch::Tensor> weights;
  for (auto element: nnapi_processed[2]) {
    weights.push_back(py::cast<torch::Tensor>(element));
  }
  c10::List<int64_t> inp_mem_fmts;
  for (auto element: nnapi_processed[3]) {
    inp_mem_fmts.push_back(py::cast<int>(element));
  }
  c10::List<int64_t> out_mem_fmts;
    for (auto element: nnapi_processed[4]) {
    out_mem_fmts.push_back(py::cast<int>(element));
  }
  dict.insert("weights", weights);
  dict.insert("inp_mem_fmts", inp_mem_fmts);
  dict.insert("out_mem_fmts", out_mem_fmts);

  return dict;
}

constexpr auto backend_name = "nnapi";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static auto pre_reg =
    torch::jit::backend_preprocess_register(backend_name, preprocess);
