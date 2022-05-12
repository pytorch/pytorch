#ifndef _WIN32
#include <functorch/csrc/CustomFunction.h>
#include <ATen/ATen.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/FunctionsManual.h>

#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/FunctionsManual.h>

namespace at { namespace functorch {

class PythonKernelHolder : public c10::OperatorKernel {
  PyObject* func_;

public:

  PythonKernelHolder(py::object func) : func_(func.release().ptr()) {}
  // This is a generally useful pattern and safer than directly using pybind11's
  // py::object destructor.  This is because this object may outlive
  // libtorch_python, so we want to disarm the deallocation if that happens.
  // PyInterpreter does this correctly, pybind11 does not.
  ~PythonKernelHolder() override {
    getPyInterpreter()->decref(func_, /*is_tensor*/false);
  }

  void operator()(const c10::OperatorHandle& op, c10::DispatchKeySet, torch::jit::Stack* stack) {
    const auto& schema = op.schema();

    const auto num_arguments = schema.arguments().size();
    auto arguments = torch::jit::pop(*stack, num_arguments);

    // TODO: Some duplication with torch/csrc/autograd/python_variable.cpp

    py::gil_scoped_acquire g;

    // Pre-scan for arguments that match defaults
    int64_t default_suffix_len = 0;
    for (int64_t idx = arguments.size() - 1; idx >= 0; idx--) {
      const auto& arg = schema.arguments()[idx];
      if (!arg.default_value().has_value()) {
        break;
      }
      const auto& default_ivalue = *arg.default_value();
      const auto& ivalue = arguments[idx];
      if (default_ivalue != ivalue) {
        break;
      }
      default_suffix_len++;
    }

    auto args = py::reinterpret_steal<py::object>(PyTuple_New(num_arguments - default_suffix_len));
        // TODO: actually populate kwargs sometimes?  At the moment, every argument
        // // just gets passed positionally
    py::dict kwargs;

    for (int64_t idx = 0; idx < (int64_t)arguments.size() - default_suffix_len; idx++) {
      PyTuple_SET_ITEM(args.ptr(), idx, torch::jit::toPyObject(std::move(arguments[idx])).release().ptr());
    }

    auto out = py::reinterpret_steal<py::object>(PyObject_Call(func_, args.ptr(), kwargs.ptr()));
    if (out.ptr() == nullptr) {
      throw python_error();
    }

    if (op.schema().returns().size() == 1) {
      torch::jit::push(stack, torch::jit::toIValue(out.ptr(), op.schema().returns()[0].type()));
    } else {
      auto outs = py::cast<py::sequence>(out);
      for (unsigned idx = 0; idx < outs.size(); idx++) {
        torch::jit::push(stack, torch::jit::toIValue(outs[idx].ptr(), op.schema().returns()[idx].type()));
      }
    }
  }
};

torch::Library::Kind parseKind(const std::string& k) {
  static std::unordered_map<std::string, torch::Library::Kind> kind_map = {
    {"DEF", torch::Library::DEF},
    {"IMPL", torch::Library::IMPL},
    {"FRAGMENT", torch::Library::FRAGMENT},
  };
  auto it = kind_map.find(k);
  TORCH_CHECK(it != kind_map.end(), "could not parse ", k);
  return it->second;
}
c10::AliasAnalysisKind parseAliasAnalysisKind(const std::string& k) {
  static std::unordered_map<std::string, c10::AliasAnalysisKind> key_map = {
    {"CONSERVATIVE", c10::AliasAnalysisKind::CONSERVATIVE},
    {"FROM_SCHEMA", c10::AliasAnalysisKind::FROM_SCHEMA},
    {"PURE_FUNCTION", c10::AliasAnalysisKind::PURE_FUNCTION},
    {"", c10::AliasAnalysisKind::FROM_SCHEMA},  // default
  };
  auto it = key_map.find(k);
  TORCH_CHECK(it != key_map.end(), "could not parse ", k);
  return it->second;
}


template <typename Func>
inline torch::CppFunction dispatch_str(const char* key, Func&& raw_f) {
  auto mb_key = std::string(key) == "" ? c10::nullopt : c10::make_optional(c10::parseDispatchKey(key));
  if (mb_key) {
    return torch::dispatch(*mb_key, std::forward<Func>(raw_f));
  } else {
    torch::CppFunction f(std::forward<Func>(raw_f));
    return f;
  }
}

std::vector<at::Tensor> unpack(at::TensorList tl, const char *name, int pos) {
  std::vector<at::Tensor> ret(tl.size());
  for (const auto i : c10::irange(tl.size())) {
    const auto &t = tl[i];
    if (!t.defined()) {
      continue;
    }
    ret[i] = static_cast<const torch::autograd::Variable&>(t);
  }
  return ret;
}

std::vector<Tensor> invoke_backward_fn(
    PyObject* backward_function,
    TensorList grads,
    TensorList intermediates) {
  std::vector<Tensor> result;

  py::gil_scoped_acquire g;
  auto args = py::reinterpret_steal<py::object>(PyTuple_New(grads.size() + intermediates.size()));
  py::dict kwargs;
  for (int64_t idx = 0; idx < (int64_t) grads.size(); idx++) {
    PyTuple_SET_ITEM(args.ptr(), idx, torch::jit::toPyObject(grads[idx]).release().ptr());
  }
  for (int64_t idx = 0; idx < (int64_t) intermediates.size(); idx++) {
    PyTuple_SET_ITEM(args.ptr(), idx, torch::jit::toPyObject(intermediates[idx + grads.size()]).release().ptr());
  }

  auto out = py::reinterpret_steal<py::object>(PyObject_Call(backward_function, args.ptr(), kwargs.ptr()));
  if (out.ptr() == nullptr) {
    throw python_error();
  }

  for (unsigned idx = 0; idx < grads.size(); idx++) {
    auto ivalue = torch::jit::toIValue(PyTuple_GetItem(out.ptr(), idx), TensorType::get());
    result.emplace_back(ivalue.toTensor());
  }
  return result;
}

// TODO: figure out what this is
using torch::autograd::variable_list;
using custom_function_t = std::vector<Tensor> (at::TensorList);

void copy_range(variable_list& out, torch::autograd::IndexRange range, at::ArrayRef<Tensor> t) {
  AT_ASSERT(range.second <= out.size());
  std::cout << range.second << ", " << range.first << ", " << t.size() << std::endl;
  AT_ASSERTM(range.second - range.first == t.size(), "inconsistent range for TensorList output");
  std::copy(t.begin(), t.end(), out.begin() + range.first);
}

struct GenericPythonBackward : public torch::autograd::TraceableFunction {
  using TraceableFunction::TraceableFunction;

  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "GenericPythonBackward"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& t : saved_tensors_) {
      t.reset_data();
    }
  }
  std::vector<torch::autograd::SavedVariable> saved_tensors_;
  int64_t num_inputs_;
  optional<c10::OperatorHandle> backward_fn_;
};

variable_list GenericPythonBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  torch::autograd::generated::details::IndexRangeGenerator gen;
  auto tensors_ix = gen.range(saved_tensors_.size());
  variable_list grad_inputs(num_inputs_);

  std::vector<Tensor> args;
  for (auto& g : grads) {
    args.emplace_back(std::move(g));
  }
  for (const auto& saved : saved_tensors_) {
    args.emplace_back(saved.unpack(shared_from_this()));
  }

  if (should_compute_output({ tensors_ix })) {
    auto handle = backward_fn_->typed<custom_function_t>();
    auto grad_result = handle.call(args);
    grad_inputs = grad_result;
    // copy_range(grad_inputs, tensors_ix, grad_result);
  }
  return grad_inputs;
}

using custom_python_function_t = TensorList (*)(TensorList);

using torch::autograd::compute_requires_grad;
using torch::autograd::collect_next_edges;
using torch::autograd::deleteNode;
using torch::autograd::flatten_tensor_args;

void customFunctionBoxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto tensors = torch::jit::pop(stack).toTensorList().vec();
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad(tensors);
  (void)_any_requires_grad;

  std::string schema_name = op.schema().name();
  std::string vjp_fn_name = schema_name + "_vjp";

  std::shared_ptr<GenericPythonBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GenericPythonBackward>(new GenericPythonBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges(tensors));
    grad_fn->backward_fn_ = c10::Dispatcher::singleton().findSchemaOrThrow(vjp_fn_name.c_str(), "");
    grad_fn->num_inputs_ = tensors_.size();
  }

  auto typed_handle = op.typed<custom_function_t>();
  std::vector<Tensor> _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return typed_handle.call(tensors_);
  })();
  auto result = std::move(_tmp);
  if (grad_fn) {
    for (auto& tensor : result) {
      // TODO: is this right?
      bool is_input = false;
      for (const auto& input : tensors_) {
        if (tensor.unsafeGetTensorImpl() == input.unsafeGetTensorImpl()) {
          is_input = true;
        }
      }

      if (!is_input) {
        set_history(tensor, grad_fn);
      }
      grad_fn->saved_tensors_.emplace_back(tensor, !is_input);
    }
  }
  torch::jit::push(stack, result);
}

void initDispatchBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<torch::Library>(m, "_DispatchModule", py::module_local())
    .def("def_", [](py::object self, const char* schema, const char* alias) {
      self.cast<torch::Library&>().def(torch::schema(schema, at::functorch::parseAliasAnalysisKind(alias)));
      return self;
    }, "", py::arg("schema"), py::arg("alias") = "")
    .def("impl", [](py::object self, const char* name, const char* dispatch, py::object func) {
      self.cast<torch::Library&>().impl(
        name,
        dispatch_str(dispatch, torch::CppFunction::makeFromBoxedFunctor(std::make_unique<at::functorch::PythonKernelHolder>(std::move(func))))
      );
    }, "", py::arg("name"), py::arg("dispatch"), py::arg("func"))
    .def("gen_backward_binding", [](py::object self, const char* name, const char* dispatch) {
      self.cast<torch::Library&>().impl(
        name,
        dispatch_str(dispatch,
          torch::CppFunction::makeFromBoxedFunction<&customFunctionBoxed>())
      );
    }, "", py::arg("name"), py::arg("dispatch"))
    .def("fallback_fallthrough", [](py::object self, const char* dispatch) {
      self.cast<torch::Library&>().fallback(
        dispatch_str(dispatch, torch::CppFunction::makeFallthrough())
      );
      return self;
    }, "", py::arg("dispatch") = "")
  ;

  m.def("_dispatch_library", [](const char* kind, std::string name, const char* dispatch) {
    auto mb_key = std::string(dispatch) == "" ? c10::nullopt : c10::make_optional(c10::parseDispatchKey(dispatch)      );
    return std::make_unique<torch::Library>(parseKind(kind), std::move(name), mb_key, "/dev/null", 0);
  });
}


}} // at::functorch
#endif // #ifndef _WIN32
