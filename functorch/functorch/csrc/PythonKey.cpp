#include <functorch/csrc/PythonKey.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace at {
namespace functorch {
// The following are publically exposed as methods of Tensor
bool PythonTensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  TORCH_CHECK(
      memory_format == at::MemoryFormat::Contiguous,
      "NYI: querying is_contiguous inside of python tensor for memory_format ",
      "other than torch.contiguous_format");
  return is_contiguous_;
}

// The following are some internal inherited methods that we do not support.
// They should never get called.
void PythonTensorImpl::set_size(int64_t dim, int64_t new_size) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_size for PythonTensorImpl");
}
void PythonTensorImpl::set_stride(int64_t dim, int64_t new_stride) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_stride for PythonTensorImpl");
}
void PythonTensorImpl::set_storage_offset(int64_t storage_offset) {
  TORCH_INTERNAL_ASSERT(false, "Can't set_storage_offset for PythonTensorImpl");
}

bool isPythonTensor(const at::Tensor& tensor) {
  return tensor.unsafeGetTensorImpl()->key_set().has(
      c10::DispatchKey::FuncTorchPython);
}
PythonTensorImpl* getPythonImpl(const at::Tensor& tensor) {
  return static_cast<PythonTensorImpl*>(tensor.unsafeGetTensorImpl());
}

at::Tensor addPythonKey(const py::object& tensor) {
  return at::detail::make_tensor<PythonTensorImpl>(tensor);
}
bool hasPythonKey(const at::Tensor& tensor) {
  return isPythonTensor(tensor);
}

py::object removePythonKey(const at::Tensor& tensor) {
  assert(isPythonTensor(tensor));
  return getPythonImpl(tensor)->value_;
}


py::object pyIdentity(py::object x) {
  return x;
}
template <class T>
py::tuple vectorToPyTuple(
    const std::vector<T>& data,
    std::function<py::object(T)> converter) {
  PyObject* tuple = PyTuple_New(data.size());
  if (!tuple)
    throw std::runtime_error("Unable to allocate memory for Python tuple");
  for (unsigned int i = 0; i < data.size(); i++) {
    PyObject* num = converter(data[i]).ptr();
    if (!num) {
      Py_DECREF(tuple);
      throw std::runtime_error("Unable to allocate memory for Python tuple");
    }
    Py_INCREF(
        num); // todo: dunno?? Need it to fix segfaults, but probably not right
    PyTuple_SET_ITEM(tuple, i, num);
  }
  return py::cast<py::tuple>(tuple);
}

void pythonFallBack(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto num_returns = schema.returns().size();

  const auto num_arguments = schema.arguments().size();
  const auto arguments = torch::jit::last(stack, num_arguments);
  py::gil_scoped_acquire g;
  std::vector<py::object> pyArgs;
  std::vector<py::object> pyTensorArgs;
  std::vector<torch::jit::IValue> unwrappedArgs;

  for (unsigned idx = 0; idx < arguments.size(); idx++) {
    const auto ivalue = arguments[idx];
    if (ivalue.isTensor() && isPythonTensor(ivalue.toTensor())) {
      auto pyTensor = getPythonImpl(ivalue.toTensor());
      pyArgs.push_back(pyTensor->value_);
      pyTensorArgs.push_back(pyTensor->value_);
      unwrappedArgs.push_back(getValueFromPyTensor(pyTensor->value_));
    } else {
      if (ivalue.isList()) {
        auto l = ivalue.toList();
        auto unwrappedL =
            c10::impl::GenericList(op.schema()
                                       .arguments()[idx]
                                       .type()
                                       ->expectRef<torch::jit::ListType>()
                                       .getElementType());
        py::list pyL;

        for (unsigned jdx = 0; jdx < l.size(); jdx++) {
          auto nv = l.get(jdx);
          if (nv.isTensor() && isPythonTensor(nv.toTensor())) {
            auto pyTensor = getPythonImpl(nv.toTensor());
            pyTensorArgs.push_back(pyTensor->value_);
            unwrappedL.push_back(getValueFromPyTensor(pyTensor->value_));
            pyL.append(pyTensor->value_);
          } else {
            unwrappedL.push_back(l.get(jdx));
            pyL.append(torch::jit::toPyObject(l.get(jdx)));
          }
        }
        pyArgs.push_back(pyL);
        unwrappedArgs.push_back(unwrappedL);
      } else {
        pyArgs.push_back(torch::jit::toPyObject(ivalue));
        unwrappedArgs.push_back(ivalue);
      }
    }
  }
  py::object torch_function =
      PyObject_FastGetAttrString(pyTensorArgs[0].ptr(), (char *)"__torch_function__");
  for (auto v : unwrappedArgs) {
    torch::jit::push(stack, v);
  }
  op.callBoxed(stack);
  std::vector<c10::IValue> realOuts = torch::jit::pop(*stack, num_returns);
  py::tuple py_types = py::cast<py::tuple>(
      vectorToPyTuple<py::object>(pyArgs, [](py::object x) -> py::object {
        return py::reinterpret_borrow<py::object>(PyObject_Type(x.ptr()));
      }));

  py::dict kwargs;
  std::vector<py::object> t;
  for (auto x : realOuts) {
    t.push_back(torch::jit::toPyObject(x));
  }
  kwargs["val"] = vectorToPyTuple<py::object>(t, pyIdentity);

  std::string func_name = op.operator_name().name;
  std::string delimiter = "aten::";
  func_name = func_name.substr(func_name.find(delimiter) + delimiter.size());

  py::object torch_api_function =
      PyObject_FastGetAttrString(THPVariableClass, (char*)func_name.c_str());

  torch_api_function = py::str(op.operator_name().name);
  auto pyTupleArgs = vectorToPyTuple<py::object>(pyArgs, pyIdentity);

  auto out = PyObject_CallFunctionObjArgs(
      torch_function.ptr(),
      torch_api_function.ptr(),
      py_types.ptr(),
      pyTupleArgs.ptr(),
      kwargs.ptr(),
      0);
  if (out == nullptr) {
    throw std::runtime_error("call failed");
  }
  py::list outs = py::cast<py::list>(out);
  torch::jit::drop(stack, num_arguments);
  std::vector<c10::IValue> ret_ivalues;
  assert(outs.size() == op.schema().returns().size());
  for (unsigned idx = 0; idx < outs.size(); idx++) {
    auto ret_type = op.schema().returns()[idx].type();
    if (ret_type->kind() == c10::TensorType::Kind) {
      torch::jit::push(stack, addPythonKey(py::cast<py::object>(outs[idx])));
    } else {
      auto ivalue_out = torch::jit::toTypeInferredIValue(outs[idx]);
      torch::jit::push(stack, ivalue_out);
    }
  }
  return;
}
TORCH_LIBRARY_IMPL(_, FuncTorchPython, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&pythonFallBack>());
}

c10::intrusive_ptr<c10::TensorImpl> PythonTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<PythonTensorImpl>(value_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
  impl->set_version_counter(version_counter);
  impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> PythonTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<PythonTensorImpl>(value_);
  impl->set_version_counter(version_counter);
  impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  return impl;
}
}}
