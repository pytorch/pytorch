#pragma once

#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <ATen/core/stack.h>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/six.h>
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/rpc/py_rref.h>
#include <torch/csrc/distributed/rpc/rref_impl.h>
#endif

#include <ATen/core/function_schema.h>
#include <c10/core/Stream.h>
#ifdef USE_C10D_NCCL
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// The visibility attribute is to avoid a warning about storing a field in the
// struct that has a different visibility (from pybind) than the struct.
#ifdef _WIN32
#define VISIBILITY_HIDDEN
#else
#define VISIBILITY_HIDDEN __attribute__((visibility("hidden")))
#endif

namespace torch {
namespace jit {

void clear_registered_instances(void* ptr);

TORCH_API IValue toIValue(
    py::handle obj,
    const TypePtr& type,
    c10::optional<int32_t> N = c10::nullopt);

py::object toPyObject(IValue ivalue);

// Hack to overload the behavior of toIValue to accept Python
// numbers in places where a Tensor is expected
// See also torch::should_allow_numbers_as_tensors
class ToIValueAllowNumbersAsTensors {
  bool old_;

 public:
  ToIValueAllowNumbersAsTensors(bool enable);
  ~ToIValueAllowNumbersAsTensors();
};

// Wrap Python function to guard deref
// NB: Need VISIBILITY_HIDDEN for silencing compiler error,
// 'torch::jit::PythonFunctionGuard' declared with greater visibility than the
// type of its field 'torch::jit::PythonFunctionGuard::func_'
struct VISIBILITY_HIDDEN PythonFunctionGuard {
  explicit PythonFunctionGuard(py::function func) : func_(std::move(func)) {}

  ~PythonFunctionGuard() {
    pybind11::gil_scoped_acquire ag;
    func_.dec_ref();
    // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
    // decref on the PyObject again.
    // See Note [Destructing py::object] in python_ivalue.h
    func_.ptr() = nullptr;
  }

  py::function func_;
};

// The PythonFutureWrapper for ivalue::Future
//
// NB: VISIBILITY_HIDDEN is for silencing compiling error,
// "error: 'torch::jit::PythonFutureWrapper' declared with greater visibility
// than the type of its field 'torch::jit::PythonFutureWrapper::unwrap_func'
// [-Werror=attributes]"
//
// NB: inherit from enable_shared_from_this because then(py::function) needs to
//     get a shared_ptr from this pointer.
struct VISIBILITY_HIDDEN PythonFutureWrapper
    : std::enable_shared_from_this<PythonFutureWrapper> {
  using UnwrapFunc = std::function<void(py::object)>;

  explicit PythonFutureWrapper(
      c10::intrusive_ptr<c10::ivalue::Future> fut,
      c10::optional<UnwrapFunc> unwrap_func = c10::nullopt)
      : fut(std::move(fut)), unwrap_func(std::move(unwrap_func)) {}

  explicit PythonFutureWrapper(const PythonFutureWrapper&) = delete;
  PythonFutureWrapper& operator=(const PythonFutureWrapper&) = delete;

  bool done() {
    return fut->completed();
  }

  py::object value() {
    // acquiring GIL as toPyObject creates new py::object
    // without grabbing the GIL.
    py::gil_scoped_acquire acquire;
    py::object py_obj = toPyObject(fut->value());
    // unwrap_func is a general compositional function that takes in a
    // py::object and executes some python function. It is currently mostly used
    // to throw python exceptions.
    if (unwrap_func) {
      (*unwrap_func)(py_obj);
    }
    return py_obj;
  }

  py::object wait() {
    fut->wait();
    if (jit::tracer::isTracing()) {
      auto graph = jit::tracer::getTracingState()->graph;

      Value* fut_val = jit::tracer::getValueTrace(fut);
      auto output = graph->insert(aten::wait, {fut_val});
      jit::tracer::setValueTrace(fut->value(), output);
    }
    return value();
  }

  // The py::function cb arg must take a std::shared_ptr<PythonFutureWrapper>
  // (i.e., torch._C.Future) as the only argument. If the type mismatches, an
  // error will be thrown when waiting for the value of this returned Future.
  std::shared_ptr<PythonFutureWrapper> then(py::function cb) {
    // We need this an additional layer of wrapper here to guard the
    // destruction of the py::function object. Because, the
    // Future owns a reference to the py::function in its callback
    // vector, but Future does not acquire GIL on destruction.
    auto pf = std::make_shared<PythonFunctionGuard>(std::move(cb));

    return std::make_shared<jit::PythonFutureWrapper>(fut->then(
        // Capture a copy of the ivalue::Future instead of the `this` pointer
        // because the PythonFutureWrapper object could have been deleted
        // when the callbacks are fired. For example, RPC only captures the
        // ivalue::Future instead of PythonFutureWrapper in JitFuture's
        // callback functions. Hence, if user code does not hold a reference to
        // this PythonFutureWrapper object, there is no guarantee that the
        // PythonFutureWrapper is still valid when running the callback.
        [pyFut(this->getPtr()),
         pf(std::move(pf))](c10::ivalue::Future& /* unused */) -> IValue {
          try {
            pybind11::gil_scoped_acquire ag;
            return toIValue(pf->func_(pyFut), PyObjectType::get());
          } catch (py::error_already_set& e) {
            auto err = std::runtime_error(c10::str(
                "Got the following error when running the callback: ",
                e.what()));
            {
              pybind11::gil_scoped_acquire ag;
              // Release ownership on py::objects and also restore Python
              // Error Indicator.
              e.restore();
              // Clear the Python Error Indicator as we has recorded the
              // exception in the response message.
              PyErr_Clear();
            }

            throw err;
          }
        },
        PyObjectType::get()));
  }

  void add_done_callback(py::function cb) {
    auto pf = std::make_shared<PythonFunctionGuard>(std::move(cb));
    // NOLINTNEXTLINE(modernize-avoid-bind)
    fut->addCallback(std::bind(
        [pyFut(this->getPtr())](std::shared_ptr<PythonFunctionGuard> pf) {
          try {
            pybind11::gil_scoped_acquire ag;
            pf->func_(pyFut);
          } catch (py::error_already_set& e) {
            {
              pybind11::gil_scoped_acquire ag;
              // Release ownership on py::objects and also restore Python
              // Error Indicator.
              e.restore();
              // Clear the Python Error Indicator as we has recorded the
              // exception in the response message.
              PyErr_Clear();
            }
            // Log and ignore exceptions raised through the callback
            LOG(ERROR) << "Got the following error when running the callback: "
                       << e.what();

          } catch (const std::exception& e) {
            // Log and ignore exceptions raised through the callback
            LOG(ERROR) << "Got the following error when running the callback: "
                       << e.what();
          }
        },
        std::move(pf)));
  }

  void markCompleted(const py::object& pyValue) {
    DCHECK(PyGILState_Check());
    IValue value = toIValue(pyValue, PyObjectType::get());

    py::gil_scoped_release release;
    fut->markCompleted(std::move(value));
  }

  c10::intrusive_ptr<c10::ivalue::Future> fut;
  // unwrap_func works like a callback for the value returned by
  // PythonFutureWrapper::wait().
  c10::optional<UnwrapFunc> unwrap_func;

 private:
  std::shared_ptr<PythonFutureWrapper> getPtr() {
    return shared_from_this();
  }
};

// error reporting: when reporting user-caused errors, these functions should
// not use AT_ERROR macros, since these macros add stack trace information
// that is confusing to display to the end user since it always reports
// locations in libtorch code rather than user code.

inline std::shared_ptr<CompilationUnit> get_python_cu() {
  return py::module::import("torch.jit._state")
      .attr("_python_cu")
      .cast<std::shared_ptr<CompilationUnit>>();
}

struct TypedIValue : public std::pair<IValue, TypePtr> {
  using pair::pair;

  IValue& ivalue() {
    return this->first;
  }
  TypePtr& type() {
    return this->second;
  }
};

inline TypedIValue toDictKeyIValue(py::handle key) {
  if (py::isinstance<py::str>(key)) {
    return TypedIValue(
        ConstantString::create(py::cast<std::string>(key)), StringType::get());
  } else if (py::isinstance<py::int_>(key)) {
    return TypedIValue(py::cast<int64_t>(key), IntType::get());
  } else if (py::isinstance<py::float_>(key)) {
    return TypedIValue(py::cast<double>(key), FloatType::get());
  } else {
    AT_ERROR("Dictionary inputs may only have string, int, or float keys");
  }
}

inline c10::optional<TypePtr> unifyOrInitializeType(
    const TypePtr& accum,
    const TypePtr& unify) {
  if (!accum) {
    return unify;
  }
  return unifyTypes(accum, unify);
}

using InferredType = c10::InferredType;

InferredType tryToInferContainerType(py::handle input);

// Try to infer the type of a Python object
// The type cannot be inferred if:
//   input is an empty container (list, dict)
//   input is an list with element types that cannot be unified
//   input is an dict with key or value types that cannot be unified
inline InferredType tryToInferType(py::handle input) {
  // Try tensor types
  if (THPVariable_Check(input.ptr())) {
    return InferredType(TensorType::get());
  }

  if (input.is(py::none())) {
    return InferredType(NoneType::get());
  }

  if (py::isinstance<StrongFunctionPtr>(input)) {
    auto fn = py::cast<StrongFunctionPtr>(input).function_;
    return InferredType(FunctionType::create(fn));
  }

  // Try basic types first
  if (py::isinstance<py::bool_>(input)) {
    return InferredType(BoolType::get());
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (py::isinstance<py::int_>(input)) {
    return InferredType(IntType::get());
  } else if (py::isinstance<py::float_>(input)) {
    return InferredType(FloatType::get());
  } else if (PyComplex_CheckExact(input.ptr())) {
    return InferredType(ComplexType::get());
  } else if (py::isinstance<py::str>(input)) {
    return InferredType(StringType::get());
  } else if (THPLayout_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPDevice_Check(input.ptr())) {
    return InferredType(DeviceObjType::get());
  } else if (THPStream_Check(input.ptr())) {
    return InferredType(StreamObjType::get());
  } else if (THPDtype_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPQScheme_Check(input.ptr())) {
    return InferredType(IntType::get());
  } else if (THPLayout_Check(input.ptr())) {
    return InferredType(IntType::get());
  }

  auto enum_type = py::module::import("enum").attr("Enum");
  py::bool_ isEnumValue = py::isinstance(input, enum_type);
  if (py::cast<bool>(isEnumValue)) {
    auto enum_class = input.attr("__class__");
    auto enum_type = py::cast<TypePtr>(
        py::module::import("torch.jit.annotations")
            .attr("try_ann_to_type")(enum_class, SourceRange()));
    return InferredType(enum_type);
  }

  py::bool_ isClass =
      py::module::import("inspect").attr("isclass")(input.get_type());
  if (py::cast<bool>(isClass)) {
    // Assume that the class is compiled already or will compile. Invalidate
    // this later if needed.
    bool class_compiled = true;

    // Check if the type is already compiled.
    py::object existing_ty = py::module::import("torch.jit._state")
                                 .attr("_get_script_class")(input.get_type());

    if (existing_ty.is_none()) {
      // If not, try to compile it.
      py::bool_ can_compile = py::module::import("torch._jit_internal")
                                  .attr("can_compile_class")(input.get_type());

      if (py::cast<bool>(can_compile)) {
        // Try to compile the class. This is wrapped in a try-catch because
        // compilation of class types can raise an Exception and in that case,
        // we want to defer to other attempts at type inference below rather
        // than fail compilation altogether.
        try {
          py::module::import("torch.jit._script")
              .attr("_recursive_compile_class")(
                  input.get_type(), SourceRange());
        } catch (...) {
          // Invalidate the assumption that the class compiled so that we don't
          // look up and return its JIT type as the type for the input.
          class_compiled = false;
        }
      }
    }

    // If the class compiled successfully, look up the existing JIT type by
    // qualified name and return it.
    if (class_compiled) {
      auto script_class = py::module::import("torch.jit._state")
                              .attr("_get_script_class")(input.get_type());

      if (!script_class.is_none()) {
        auto class_type = py::cast<ClassTypePtr>(script_class);

        if (class_type && !class_type->is_module()) {
          return InferredType(class_type);
        }
      }
    }
  }

  if (py::isinstance<Object>(input)) {
    auto object = py::cast<Object>(input);
    return InferredType(object.type());
#ifdef USE_RPC
  } else if (py::isinstance<torch::distributed::rpc::PyRRef>(input)) {
    auto rref_ivalue = input.cast<torch::distributed::rpc::PyRRef>().toIValue();
    return InferredType(rref_ivalue.type());
#endif
  }

  if (as_module(py::cast<py::object>(input))) {
    return InferredType("Cannot infer type of ScriptModule");
  }

  auto module_type = py::module::import("torch.nn").attr("Module");
  py::bool_ is_module = py::isinstance(input, module_type);
  if (py::cast<bool>(is_module)) {
    return InferredType("Cannot infer concrete type of torch.nn.Module");
  }

  // Try container types
  return tryToInferContainerType(input);
}

inline InferredType tryToInferContainerType(py::handle input) {
  if (six::isTuple(input)) {
    py::tuple tuple = py::cast<py::tuple>(input);
    std::vector<TypePtr> element_types;
    element_types.reserve(tuple.size());

    for (py::handle elem : tuple) {
      auto type_match = tryToInferType(elem);
      if (type_match.success()) {
        element_types.push_back(type_match.type());
      } else {
        // Forward error message along
        return type_match.reason();
      }
    }
    return InferredType(TupleType::create(element_types));
  } else if (PyDict_Check(input.ptr())) {
    // Check to make sure we can generate useful input/output types
    auto dict = py::cast<py::dict>(input);
    size_t len = py::len(dict);
    if (!len) {
      return InferredType("Dictionary inputs must have entries");
    }

    TypePtr key_type = nullptr;
    TypePtr value_type = nullptr;

    for (auto entry : dict) {
      // Try to infer the key type and unify it with the existing one
      auto entry_key_type_match = tryToInferType(entry.first);
      if (!entry_key_type_match.success()) {
        return entry_key_type_match.reason();
      }
      auto unified_key =
          unifyOrInitializeType(key_type, entry_key_type_match.type());
      if (!unified_key) {
        return InferredType(c10::str(
            "Dictionary inputs to traced functions must have consistent type. Found ",
            key_type->repr_str(),
            " and ",
            (entry_key_type_match.type())->repr_str()));
      }

      // Try to infer the value type and unify it with the existing one
      auto entry_value_type_match = tryToInferType(entry.second);
      if (!entry_value_type_match.success()) {
        return entry_value_type_match.reason();
      }
      auto unified_value =
          unifyOrInitializeType(value_type, entry_value_type_match.type());
      if (!unified_value) {
        return InferredType(c10::str(
            "Dictionary inputs to traced functions must have consistent type. Found ",
            value_type->repr_str(),
            " and ",
            (entry_value_type_match.type())->repr_str()));
      }

      key_type = *unified_key;
      value_type = *unified_value;
    }
    return InferredType(DictType::create(key_type, value_type));
  } else if (PyList_Check(input.ptr())) {
    auto list = py::cast<py::list>(input);
    size_t len = py::len(list);
    if (!len) {
      return InferredType("List trace inputs must have elements");
    }

    TypePtr element_type = nullptr;
    for (auto elem : list) {
      auto element_type_match = tryToInferType(elem);
      if (!element_type_match.success()) {
        return InferredType(c10::str(
            "Could not infer type of list element: ",
            element_type_match.reason()));
      }
      auto unified_type =
          unifyOrInitializeType(element_type, element_type_match.type());
      if (!unified_type) {
        return InferredType(c10::str(
            "List inputs to traced functions must have consistent element type. Found ",
            element_type->repr_str(),
            " and ",
            (element_type_match.type())->repr_str()));
      }
      element_type = *unified_type;
    }
    return InferredType(ListType::create(element_type));
  } else {
    // TODO: this message is not correct anymore, since this InferredType is
    // used from a bunch of circumstances unrelated to tracing. We can re-use
    // this instead of the attribute_failure stuff in concreteType
    return InferredType(c10::str(
        "Only tensors and (possibly nested) tuples of tensors, lists, or dicts",
        "are supported ",
        "as inputs or outputs of traced functions",
        ", but instead got value of type ",
        py::str(input.get_type().attr("__name__")),
        "."));
  }
}

inline bool isTraceableType(const TypePtr& type) {
  if (type->isSubtypeOf(*TensorType::get())) {
    return true;
  }

  if (auto list_type = type->cast<ListType>()) {
    return isTraceableType(list_type->getElementType());
  }

  if (auto tuple_type = type->cast<TupleType>()) {
    return std::all_of(
        tuple_type->elements().begin(),
        tuple_type->elements().end(),
        [](const TypePtr& element_type) {
          return isTraceableType(element_type);
        });
  }

  if (auto dict_type = type->cast<DictType>()) {
    return isTraceableType(dict_type->getValueType());
  }

  return false;
}

inline IValue toTypeInferredIValue(py::handle input) {
  auto match = tryToInferType(input);
  if (!match.success()) {
    AT_ERROR(
        "Tracer cannot infer type of ", py::str(input), "\n:", match.reason());
  }
  return toIValue(input, match.type());
}

inline Stack toTraceableStack(const py::tuple& inputs) {
  auto info = toTypeInferredIValue(inputs);
  TORCH_CHECK(
      isTraceableType(info.type()),
      "Type '",
      info.type()->repr_str(),
      "' cannot be traced. Only Tensors and (possibly nested) Lists, Dicts, and"
      " Tuples of Tensors can be traced");
  return info.toTupleRef().elements().vec();
}

inline IValue createGenericList(py::handle obj, const TypePtr& elem_type) {
  auto elems = c10::impl::GenericList(elem_type);
  for (auto elem : obj) {
    elems.push_back(toIValue(elem, elem_type));
  }
  return IValue(std::move(elems));
}

inline IValue createGenericDict(
    const py::dict& obj,
    const TypePtr& key_type,
    const TypePtr& value_type) {
  c10::impl::GenericDict elems(key_type, value_type);
  elems.reserve(py::len(obj));
  for (auto& entry : obj) {
    elems.insert(
        toIValue(entry.first, key_type), toIValue(entry.second, value_type));
  }
  return IValue(std::move(elems));
}

template <class T>
inline void guardAgainstNamedTensor(const T& var) {
  TORCH_CHECK(
      !var.has_names(),
      "NYI: Named tensors are currently unsupported in TorchScript. As a  "
      "workaround please drop names via `tensor = tensor.rename(None)`.");
}

// Defined in pybind_utils.cpp to break a circular dependency with
// python_ivalue.h
IValue toIValue(py::handle obj, const TypePtr& type, c10::optional<int32_t> N);

// Extract custom class registered with torchbind
template <typename T>
c10::intrusive_ptr<T> toCustomClass(py::handle obj) {
  static_assert(
      std::is_base_of<CustomClassHolder, T>::value, "T is not a CustomClass");
  const auto& type = c10::getCustomClassType<c10::intrusive_ptr<T>>();
  c10::IValue ivalue = toIValue(obj, type);
  return std::move(ivalue).toCustomClass<T>();
}

// Small wrapper around getting the type name string from Python to make
// types easier to interpret, e.g. give the structural type for a NamedTuple
inline std::string friendlyTypeName(py::handle obj) {
  if (py::isinstance<py::tuple>(obj) && py::hasattr(obj, "_fields")) {
    auto field_names =
        py::cast<std::vector<std::string>>(py::getattr(obj, "_fields"));
    std::stringstream ss;
    ss << py::str(obj.get_type().attr("__name__"));
    ss << " (aka NamedTuple(";
    bool first = true;
    for (auto& field_name : field_names) {
      if (!first) {
        ss << ", ";
      }
      ss << field_name;
      first = false;
    }
    ss << "))";
    return ss.str();
  } else {
    return py::str(obj.get_type().attr("__name__"));
  }
}

// Thrown when trying to create a schema for a list of python
// arguments that cannot be converted.
// Can be caught by the caller to attempt to use other schema
// when there is an overloaded operator.
struct schema_match_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline IValue argumentToIValue(
    const FunctionSchema& schema,
    size_t argumentPosition,
    py::handle object) {
  const auto& argument = schema.arguments().at(argumentPosition);
  try {
    return toIValue(object, argument.real_type(), argument.N());
  } catch (const py::cast_error& error) {
    throw schema_match_error(c10::str(
        schema.formatTypeMismatchMsg(
            argument,
            friendlyTypeName(object),
            argumentPosition,
            py::repr(object)),
        "\nCast error details: ",
        error.what()));
  } catch (const py::error_already_set& error) {
    throw schema_match_error(c10::str(
        schema.formatTypeMismatchMsg(
            argument,
            friendlyTypeName(object),
            argumentPosition,
            py::repr(object)),
        "\n Python error details: ",
        error.what()));
  }
}

inline IValue returnToIValue(const TypePtr& type, py::handle object) {
  try {
    return toIValue(object, type);
  } catch (const py::cast_error& error) {
    throw std::runtime_error(c10::str(
        " expected value of type ",
        type->str(),
        " for return value but instead got value of type ",
        py::str(object.get_type().attr("__name__")),
        ".",
        "\nValue: ",
        py::repr(object),
        "\nCast error details: ",
        error.what()));
  }
}

inline py::object getScriptedClassOrError(const c10::NamedTypePtr& classType) {
  auto py_class =
      py::module::import("torch.jit._state")
          .attr("_get_python_class")(classType->name()->qualifiedName());
  if (py_class.is_none()) {
    std::stringstream err;
    err << "Unknown reference to ScriptClass ";
    err << classType->name()->qualifiedName();
    err << ". (Did you forget to import it?)";
    throw std::runtime_error(err.str());
  }
  return py_class;
}

inline py::object toPyObject(IValue ivalue) {
  if (ivalue.isNone()) {
    return py::none();
  } else if (ivalue.isTensor()) {
    auto tensor = std::move(ivalue).toTensor();
    if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
      TORCH_INTERNAL_ASSERT(tensor.device().is_cpu());
      auto scalar_type = tensor.scalar_type();
      switch (scalar_type) {
        case at::ScalarType::Bool:
          return py::cast(*tensor.data_ptr<bool>());
        case at::ScalarType::Long:
          return py::cast(*tensor.data_ptr<int64_t>());
        case at::ScalarType::Double:
          return py::cast(*tensor.data_ptr<double>());
        case at::ScalarType::ComplexDouble:
          // TODO: https://github.com/pytorch/pytorch/issues/77134
          return py::cast(static_cast<std::complex<double>>(
              *tensor.data_ptr<c10::complex<double>>()));
        default:
          TORCH_CHECK(
              false,
              "Missing cases in 'toPyObject' wrapped number handling! Can't convert ",
              scalar_type,
              " to a Python object");
      }
    } else {
      guardAgainstNamedTensor<at::Tensor>(tensor);
      return py::cast(autograd::Variable(std::move(tensor)));
    }
  } else if (ivalue.isStorage()) {
    return py::cast(ivalue.toStorage());
  } else if (ivalue.isGenerator()) {
    return py::cast(ivalue.toGenerator());
  } else if (ivalue.isDouble()) {
    return py::cast(std::move(ivalue).toDouble());
  } else if (ivalue.isComplexDouble()) {
    return py::cast(
        static_cast<std::complex<double>>(std::move(ivalue).toComplexDouble()));
  } else if (ivalue.isInt()) {
    return py::cast(std::move(ivalue).toInt());
  } else if (ivalue.isBool()) {
    return py::cast(std::move(ivalue).toBool());
  } else if (ivalue.isString()) {
    return py::cast(std::move(ivalue).toStringRef());
  } else if (ivalue.isList()) {
    auto list = std::move(ivalue).toList();
    py::list t{list.size()};
    for (const auto i : c10::irange(list.size())) {
      t[i] = toPyObject(IValue{list.get(i)});
    }
    return std::move(t);
  } else if (ivalue.isTuple()) {
    auto tuple = std::move(ivalue).toTuple();
    const auto& elements = tuple->elements();

    py::tuple t{elements.size()};
    for (const auto i : c10::irange(elements.size())) {
      t[i] = toPyObject(IValue{elements.at(i)});
    }

    // If we have a NamedTuple
    if (tuple->type() && tuple->type()->schema() &&
        tuple->type()->schema()->name() != "") {
      auto unqualName = tuple->type()->name()->name();

      const std::vector<Argument>& tuple_args =
          tuple->type()->schema()->arguments();

      std::vector<pybind11::object> defaults;
      auto it = std::find_if(
          tuple_args.begin(), tuple_args.end(), [](const Argument& arg) {
            return arg.default_value().has_value();
          });
      std::transform(
          it,
          tuple_args.end(),
          std::back_inserter(defaults),
          [](const Argument& arg) { return toPyObject(*arg.default_value()); });

      std::vector<std::string> fieldNames =
          fmap(tuple_args, [](const Argument& arg) { return arg.name(); });

      return py::module::import("torch._jit_internal")
          .attr("_create_named_tuple")(
              t, unqualName, fieldNames, py::make_tuple(defaults));
    } else {
      return std::move(t);
    }
  } else if (ivalue.isDevice()) {
    return py::cast<py::object>(THPDevice_New(std::move(ivalue).toDevice()));
  } else if (ivalue.isGenericDict()) {
    auto dict = std::move(ivalue).toGenericDict();
    py::dict py_dict;
    for (auto& pair : dict) {
      py_dict[toPyObject(IValue{pair.key()})] =
          toPyObject(IValue{pair.value()});
    }
    return std::move(py_dict);
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    auto RRefPtr =
        c10::dynamic_intrusive_pointer_cast<torch::distributed::rpc::RRef>(
            std::move(ivalue).toRRef());
    return py::cast(torch::distributed::rpc::PyRRef(RRefPtr));
#else
    AT_ERROR("RRef is only supported with the distributed package");
#endif
  } else if (ivalue.isObject()) {
    const auto obj = std::move(ivalue).toObject();
    if (obj->type()->is_module()) {
      return py::cast(Module(obj));
    }

    auto pyCu = get_python_cu();
    if (obj->name().find("__torch__.torch.classes") == 0) {
      return py::cast(Object(obj));
    }
    const auto classType = pyCu->get_class(c10::QualifiedName(obj->name()));
    AT_ASSERT(classType);
    auto pyClass = getScriptedClassOrError(obj->type());
    auto pyObj = pyClass.attr("__new__")(pyClass);

    const auto numAttrs = classType->numAttributes();

    for (const auto slot : c10::irange(numAttrs)) {
      const auto& attrName = classType->getAttributeName(slot);
      IValue v = obj->getSlot(slot);
      py::setattr(pyObj, attrName.c_str(), toPyObject(std::move(v)));
    }
    return pyObj;
  } else if (ivalue.isPyObject()) {
    // return borrowed reference to ensure it correctly incref the underlying
    // PyObject
    return py::reinterpret_borrow<py::object>(ivalue.toPyObject());
  } else if (ivalue.isCapsule()) {
    return py::cast(c10::Capsule(ivalue.toCapsule()));
  } else if (ivalue.isFuture()) {
    return py::cast(std::make_shared<PythonFutureWrapper>(ivalue.toFuture()));
  } else if (ivalue.isEnum()) {
    auto enum_holder = ivalue.toEnumHolder();
    auto py_class = getScriptedClassOrError(enum_holder->type());
    return py_class.attr(enum_holder->name().c_str());
  } else if (ivalue.isRRef()) {
#ifdef USE_RPC
    return py::cast(torch::distributed::rpc::PyRRef(
        c10::static_intrusive_pointer_cast<distributed::rpc::RRef>(
            ivalue.toRRef())));
#else
    TORCH_CHECK(false, "RRef is only supported with the distributed package");
#endif
  } else if (ivalue.isSymInt()) {
    auto si = ivalue.toSymInt();
    return py::cast(si);
  } else {
    AT_ERROR(
        "Missing cases in 'toPyObject'! Can't convert ",
        ivalue.tagKind(),
        " to a Python object");
  }
}

struct VISIBILITY_HIDDEN tuple_slice {
  /*implicit*/ tuple_slice(py::tuple tup_)
      : tup(std::move(tup_)), b(0), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_)
      : tup(std::move(tup_)), b(b_), e(tup.size()) {}
  tuple_slice(py::tuple tup_, int64_t b_, int64_t e_)
      : tup(std::move(tup_)), b(b_), e(e_) {}
  py::detail::tuple_iterator begin() const {
    return {tup, static_cast<pybind11::ssize_t>(b)};
  }
  py::detail::tuple_iterator end() const {
    return {tup, static_cast<pybind11::ssize_t>(e)};
  }
  size_t size() const {
    return e - b;
  }
  py::detail::tuple_accessor operator[](size_t index) const {
    return {tup, static_cast<size_t>(b + index)};
  }

 private:
  py::tuple tup;
  int64_t b;
  int64_t e;
};

inline Stack createStackForSchema(
    const FunctionSchema& schema,
    const tuple_slice& args,
    const py::kwargs& kwargs,
    c10::optional<IValue> self) {
  size_t all_arguments = (self ? 1 : 0) + args.size() + kwargs.size();
  if (all_arguments > schema.arguments().size()) {
    throw schema_match_error(c10::str(
        schema.name(),
        "() expected at most ",
        schema.arguments().size(),
        " argument(s) but received ",
        all_arguments,
        " argument(s). Declaration: ",
        schema));
  }
  Stack stack;
  stack.reserve(schema.arguments().size());

  int64_t arg_idx = 0;
  if (self) {
    push(stack, std::move(*self));
    arg_idx++;
  }
  // First push all positional args.
  for (const auto& arg : args) {
    // ...but refuse to do it if the schema says that this was supposed
    // to be keyword only
    if (schema.arguments()[arg_idx].kwarg_only()) {
      throw schema_match_error(c10::str(
          schema.name(),
          "() takes ",
          arg_idx,
          " positional argument(s) but ",
          self ? 1 + args.size() : args.size(),
          " was/were given.  Declaration: ",
          schema));
    }
    // Use the type information from the schema to convert the PyObject.
    push(stack, argumentToIValue(schema, stack.size(), arg));
    arg_idx++;
  }

  // Now for every remaining non-positional argument in the schema, look for it
  // in the kwargs dict and push it if found, or use its default value if it
  // has one.
  size_t consumed_kwargs = 0;
  for (size_t i = stack.size(); i < schema.arguments().size(); ++i) {
    const auto& arg = schema.arguments()[i];
    if (kwargs.contains(arg.name().c_str())) {
      push(stack, argumentToIValue(schema, i, kwargs[arg.name().c_str()]));
      consumed_kwargs += 1;
    } else if (arg.default_value()) {
      push(stack, *arg.default_value());
    } else {
      throw schema_match_error(c10::str(
          schema.name(),
          "() is missing value for argument '",
          arg.name(),
          "'. Declaration: ",
          schema));
    }
  }

  if (consumed_kwargs != kwargs.size()) {
    std::vector<std::string> names;
    for (const auto& kwarg : kwargs) {
      names.emplace_back(py::cast<std::string>(kwarg.first));
    }
    throw schema_match_error(schema.findErrorInKwargs(names));
  }

  return stack;
}

inline py::object createPyObjectForStack(Stack&& stack) {
  if (stack.empty()) {
    return py::none();
  }

  // Return a simple value and not a single-element tuple if there is only one
  // return value.
  if (stack.size() == 1) {
    return toPyObject(std::move(stack[0]));
  }

  // If there is more than one return value, pop them into a py::tuple.
  py::tuple return_values(stack.size());
  for (const auto ret : c10::irange(return_values.size())) {
    return_values[ret] = toPyObject(std::move(stack[ret]));
  }

  return std::move(return_values);
}

// TODO: Remove once we clean up the GraphExecutor usage.
inline Stack evilDeprecatedBadCreateStackDoNotUse(
    const py::tuple& tuple,
    at::ArrayRef<Value*> inputs,
    size_t reserve_extra_space = 0) {
  if (tuple.size() != inputs.size()) {
    AT_ERROR(
        "expected " + std::to_string(inputs.size()) + " inputs, but got " +
        std::to_string(tuple.size()));
  }
  Stack result;
  result.reserve(tuple.size() + reserve_extra_space);
  for (const auto i : c10::irange(inputs.size())) {
    result.push_back(toIValue(std::move(tuple[i]), inputs[i]->type()));
  }
  return result;
}

// Run `callee`, potentially inserting a CallFunction/CallMethod node into the
// tracing graph.
inline py::object runAndInsertCall(
    Function& callee,
    const tuple_slice& args,
    const py::kwargs& kwargs,
    c10::optional<IValue> self,
    // Lambda that tells this function how to insert `callee` into the graph if
    // we're tracing.
    const std::function<Value*(Graph&, const MatchedSchema& match)>&
        callInserter) {
  auto stack =
      createStackForSchema(callee.getSchema(), args, kwargs, std::move(self));
  const auto& tracing_state = tracer::getTracingState();
  if (!tracing_state) {
    pybind11::gil_scoped_release no_gil_guard;
    // If we're not tracing, just run the callee as normal.
    callee.run(stack);
  } else {
    // If we are tracing, insert the appropriate CallFunction or CallMethod node
    // and then run the callee with tracing disabled.

    // Get the graph `Value`s that represent the input IValues
    auto inputs = last(stack, callee.num_inputs());
    auto input_values =
        fmap(inputs, [](const IValue& v) { return tracer::getValueTrace(v); });
    TORCH_INTERNAL_ASSERT(callee.getSchema().returns().size() == 1)
    auto return_type = callee.getSchema().returns().at(0).type();
    auto graph = tracing_state->graph;
    std::vector<NamedValue> named_values;
    named_values.reserve(input_values.size());
    for (Value* v : input_values) {
      named_values.emplace_back(v);
    }

    // Add a call node.
    MatchedSchema match = matchSchema(
        callee.getSchema(),
        tracer::getPythonInterpreterSourceRange(),
        *graph,
        named_values,
        {});
    auto output_value = callInserter(*graph, match);

    // Actually run the callee. Pause the tracer so that we don't double-add the
    // callee nodes.
    {
      pybind11::gil_scoped_release no_gil_guard;
      ResourceGuard guard(tracer::pauseTracing());
      callee.run(stack);
    }

    // Associate the output IValues with the output `Value`s in the graph
    tracer::setValueTrace(stack.back(), output_value);
  }

  TORCH_CHECK(
      stack.size() > 0,
      "Expected values in the stack after execution but found none");
  return toPyObject(std::move(stack.back()));
}

inline c10::optional<py::object> maybeTorchFunctionDispatch(
    const py::object& callee,
    const tuple_slice& args_no_self,
    const py::kwargs& kwargs,
    const c10::QualifiedName qualname) {
  std::vector<py::handle> args_vec;
  for (const auto& arg : args_no_self) {
    args_vec.push_back(arg);
  }
  py::tuple args = py::cast(args_vec);

  // Handle __torch_function__ dispatch
  std::vector<py::handle> overloaded_args;
  size_t total_arg_num = args.size() + kwargs.size();
  for (const auto& arg : args) {
    is_tensor_and_append_overloaded(arg.ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        arg.ptr(),
        &overloaded_args,
        static_cast<int>(total_arg_num),
        false /* throw_error */);
  }
  // NB: for kwargs, we cannot guarantee the order of appending
  // is the same as the argument order in operator's schema.
  // This is suboptimal, but should be fine. Later when we have
  // better schema matching and argument parsing, we could
  // match the operator in `operations` first, then the order will
  // be guaranteed.
  for (auto item : kwargs) {
    is_tensor_and_append_overloaded(item.second.ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        item.second.ptr(),
        &overloaded_args,
        total_arg_num,
        false /* throw_error */);
  }
  if (overloaded_args.size() > 0) {
    return pybind11::reinterpret_steal<py::object>(
        handle_torch_function_no_python_arg_parser(
            /*overloaded_args=*/overloaded_args,
            /*args=*/args.ptr(),
            /*kwargs=*/kwargs.ptr(),
            /*func_name=*/qualname.name().c_str(),
            /*torch_api_function=*/callee.ptr(),
            /*module_name=*/qualname.prefix().c_str()));
  }

  return c10::nullopt;
}

inline py::object invokeScriptFunctionFromPython(
    Function& callee,
    const tuple_slice& args,
    const py::kwargs& kwargs) {
  // TODO: we could add __torch_function__ dispatch here but I don't know
  // the implications of doing so

  return runAndInsertCall(
      callee,
      args,
      kwargs,
      /*self=*/c10::nullopt,
      [&](Graph& graph, const MatchedSchema& match) {
        return graph.insertFunctionCall(&callee, match);
      });
}

inline py::object invokeScriptMethodFromPython(
    Method& callee,
    const tuple_slice& args,
    const py::kwargs& kwargs) {
  auto self = callee.owner()._ivalue();

  if (auto torch_fn_result = maybeTorchFunctionDispatch(
          py::cast(callee), args, kwargs, callee.name())) {
    return *torch_fn_result;
  }

  return runAndInsertCall(
      callee.function(),
      args,
      kwargs,
      self,
      [&](Graph& graph, const MatchedSchema& match) {
        return graph.insertMethodCall(callee.name(), match);
      });
}

inline std::pair<std::shared_ptr<Operator>, Stack> getOpWithStack(
    const std::vector<std::shared_ptr<Operator>>& operations,
    py::args args,
    const py::kwargs& kwargs) {
  Stack stack;
  if (operations.size() == 1) {
    std::shared_ptr<Operator> op = operations.at(0);
    // Create a stack full of the arguments and keyword arguments.
    stack = createStackForSchema(
        op->schema(), std::move(args), kwargs, c10::nullopt);

    return std::make_pair(op, stack);
  } else {
    std::vector<schema_match_error> errors;
    std::shared_ptr<Operator> found_op = nullptr;
    for (const auto& op : operations) {
      try {
        stack = createStackForSchema(op->schema(), args, kwargs, c10::nullopt);
        found_op = op;
        break;
      } catch (schema_match_error& error) {
        errors.push_back(std::move(error));
      }
    }
    if (!found_op) {
      std::stringstream ss;
      ss << "Overloaded torch operator invoked from Python failed to many any schema:\n";
      for (const auto& err : errors) {
        ss << err.what() << "\n\n";
      }
      throw std::runtime_error(ss.str());
    }

    return std::make_pair(found_op, stack);
  }
}
inline py::object invokeOperatorFromPython(
    const std::vector<std::shared_ptr<Operator>>& operations,
    py::args args,
    const py::kwargs& kwargs,
    c10::optional<c10::DispatchKey> dk = c10::nullopt) {
  auto opWithStack = getOpWithStack(operations, args, kwargs);
  std::shared_ptr<Operator> found_op = std::get<0>(opWithStack);
  Stack stack = std::get<1>(opWithStack);
  {
    pybind11::gil_scoped_release no_gil_guard;
    if (dk) {
      found_op->getOperationForDispatchKey (*dk)(stack);
    } else {
      found_op->getOperation()(stack);
    }
  }

  return createPyObjectForStack(std::move(stack));
}

inline py::object _get_operation_for_overload_or_packet(
    const std::vector<std::shared_ptr<Operator>>& operations,
    Symbol symbol,
    py::args args,
    const py::kwargs& kwargs,
    bool is_overload,
    c10::optional<c10::DispatchKey> dk = c10::nullopt) {
  std::vector<py::handle> overloaded_args;
  size_t total_arg_num = args.size() + kwargs.size();
  for (const auto i : c10::irange(args.size())) {
    is_tensor_and_append_overloaded(args[i].ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        args[i].ptr(),
        &overloaded_args,
        static_cast<int>(total_arg_num),
        false /* throw_error */);
  }
  // NB: for kwargs, we cannot guarantee the order of appending
  // is the same as the argument order in operator's schema.
  // This is suboptimal, but should be fine. Later when we have
  // better schema matching and argument parsing, we could
  // match the operator in `operations` first, then the order will
  // be guaranteed.
  for (auto item : kwargs) {
    is_tensor_and_append_overloaded(item.second.ptr(), &overloaded_args);
    is_tensor_list_and_append_overloaded(
        item.second.ptr(),
        &overloaded_args,
        total_arg_num,
        false /* throw_error */);
  }
  if (overloaded_args.size() > 0 ||
      at::impl::PythonTorchFunctionTLS::get_mode()) {
    py::object ret;
    std::string ns = symbol.ns().toUnqualString();
    std::string method_name = symbol.toUnqualString();
    auto self_func = py::module::import("torch")
                         .attr("ops")
                         .attr(ns.c_str())
                         .attr(method_name.c_str());
    if (is_overload) {
      auto overload_name = operations[0]->schema().overload_name();
      if (overload_name == "") {
        self_func = self_func.attr("default");
      } else {
        self_func = self_func.attr(overload_name.c_str());
      }
    }
    std::string module_name("torch.ops");
    module_name.append(ns);
    return pybind11::reinterpret_steal<py::object>(
        handle_torch_function_no_python_arg_parser(
            overloaded_args,
            args.ptr(),
            kwargs.ptr(),
            method_name.c_str(),
            self_func.ptr(),
            module_name.c_str()));
  }
  return invokeOperatorFromPython(operations, args, kwargs, dk);
}

} // namespace jit
} // namespace torch
