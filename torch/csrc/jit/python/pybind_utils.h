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

namespace torch::jit {

using ResolutionCallback = std::function<py::object(std::string)>;

void clear_registered_instances(void* ptr);

TORCH_PYTHON_API IValue toIValue(
    py::handle obj,
    const TypePtr& type,
    c10::optional<int32_t> N = c10::nullopt);

TORCH_PYTHON_API py::object toPyObject(IValue ivalue);

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

// The PythonAwaitWrapper for ivalue::Await
//
// Expresses delayed function execution with Lazy semantic.
// i.e. Await[W] in eager mode can be used as W.
// When the attribute of W type is requested, Await[W] will return the
// attribute of W, transparently calling wait() beforehand.
// No Lazy semantic for script, explicit wait(Await[W]) -> W must be called to
// convert to type W.
//
// The Await object takes shared ownership of specified function and the
// arguments. After first call for wait() it owns the result. Deliberately no
// type inference for eager mode.
struct VISIBILITY_HIDDEN PythonAwaitWrapper
    : std::enable_shared_from_this<PythonAwaitWrapper> {
  explicit PythonAwaitWrapper(c10::intrusive_ptr<c10::ivalue::Await> aw)
      : aw_(std::move(aw)) {}
  explicit PythonAwaitWrapper(py::handle input) {
    args_ = py::tuple(1u);
    args_[0] = input;
    auto type = PyObjectType::get();
    aw_ = c10::make_intrusive<c10::ivalue::Await>(type);
    aw_->markCompleted(toIValue(input, type));
  }

  explicit PythonAwaitWrapper(py::function pf, py::tuple args) {
    pyfg_ = std::make_shared<torch::jit::PythonFunctionGuard>(std::move(pf));
    args_ = std::move(args);
    std::function<IValue()> f = [fg(pyfg_), &args(args_)]() {
      pybind11::gil_scoped_acquire ag;
      return toIValue(fg->func_(*args), PyObjectType::get());
    };
    aw_ = c10::make_intrusive<c10::ivalue::Await>(
        PyObjectType::get(), std::move(f));
  }

  explicit PythonAwaitWrapper(const PythonAwaitWrapper&) = delete;
  PythonAwaitWrapper& operator=(const PythonAwaitWrapper&) = delete;

  py::object wait() {
    py::gil_scoped_acquire acquire;
    return toPyObject(aw_->wait());
  }

  // Nowait semantic means trivial case when Await is constructed from the
  // result
  bool is_nowait() {
    return pyfg_ == nullptr;
  }

  const py::function fn() {
    TORCH_CHECK(
        pyfg_, "Await constructed as awaitable_nowait does not have fn");
    return pyfg_->func_;
  }

  const py::tuple args() {
    return args_;
  }

  TypePtr type() {
    return aw_->type();
  }

  c10::intrusive_ptr<c10::ivalue::Await> aw_;
  std::shared_ptr<torch::jit::PythonFunctionGuard> pyfg_;
  py::tuple args_;

 private:
  std::shared_ptr<PythonAwaitWrapper> getPtr() {
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

InferredType tryToInferContainerType(py::handle input, bool primitiveTypeOnly);

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

  if (input.is_none()) {
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
  } else if (THPGenerator_Check(input.ptr())) {
    return InferredType(GeneratorType::get());
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
    return InferredType(std::move(enum_type));
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
          return InferredType(std::move(class_type));
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

  auto await_type = py::module::import("torch._awaits").attr("_Await");
  py::bool_ is_await = py::isinstance(input, await_type);
  if (py::cast<bool>(is_await)) {
    auto awptr = input.cast<std::shared_ptr<PythonAwaitWrapper>>();
    return InferredType(AwaitType::create(awptr->aw_->elementType()));
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
  return tryToInferContainerType(input, false);
}

// This function is similar to tryToInferType, but it only tries to infer
// primitive types (int, float, bool, complex) or nested container of primitive
// types.
inline InferredType tryToInferPrimitiveType(py::handle input) {
  if (input.is_none()) {
    return InferredType(NoneType::get());
  }

  // Only primitive data type
  if (py::isinstance<py::bool_>(input)) {
    return InferredType(BoolType::get());
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (py::isinstance<py::int_>(input)) {
    return InferredType(IntType::get());
  } else if (py::isinstance<py::float_>(input)) {
    return InferredType(FloatType::get());
  } else if (PyComplex_CheckExact(input.ptr())) {
    return InferredType(ComplexType::get());
  }

  // Try container types
  return tryToInferContainerType(input, true);
}

inline InferredType tryToInferContainerType(
    py::handle input,
    bool primitiveTypeOnly = false) {
  if (six::isTuple(input)) {
    py::tuple tuple = py::cast<py::tuple>(input);
    std::vector<TypePtr> element_types;
    element_types.reserve(tuple.size());

    for (py::handle elem : tuple) {
      auto type_match = primitiveTypeOnly ? tryToInferPrimitiveType(elem)
                                          : tryToInferType(elem);
      if (type_match.success()) {
        element_types.push_back(type_match.type());
      } else {
        // Forward error message along
        return type_match.reason();
      }
    }
    return InferredType(TupleType::create(std::move(element_types)));
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
      auto entry_key_type_match = primitiveTypeOnly
          ? tryToInferPrimitiveType(entry.first)
          : tryToInferType(entry.first);
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
      auto entry_value_type_match = primitiveTypeOnly
          ? tryToInferPrimitiveType(entry.second)
          : tryToInferType(entry.second);
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
    return InferredType(
        DictType::create(std::move(key_type), std::move(value_type)));
  } else if (PyList_Check(input.ptr())) {
    auto list = py::cast<py::list>(input);
    size_t len = py::len(list);
    if (!len) {
      return InferredType("List trace inputs must have elements");
    }

    TypePtr element_type = nullptr;
    for (auto elem : list) {
      auto element_type_match = primitiveTypeOnly
          ? tryToInferPrimitiveType(elem)
          : tryToInferType(elem);
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
    if (primitiveTypeOnly) {
      return InferredType(c10::str(
          "Only tuple, list, or dict (possibly nested) of primitive types (bool, float, int, complex)",
          "are supported ",
          "as inputs or outputs of traced functions",
          ", but instead got value of type ",
          py::str(input.get_type().attr("__name__")),
          "."));
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
    auto object = py::cast<py::object>(input);
    if (auto mod = as_module(object)) {
      // if obj is already a ScriptModule, just return its ivalue
      auto ptr = mod.value()._ivalue();
      // explict copy semantics for strong ownership of the resource.
      return c10::intrusive_ptr<c10::ivalue::Object>::reclaim_copy(
          ptr.release());
    }

    // Check if the obj is a ScriptObject.
    if (auto script_obj = as_object(object)) {
      auto ptr = script_obj.value()._ivalue();
      return c10::intrusive_ptr<c10::ivalue::Object>::reclaim_copy(
          ptr.release());
    }
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

// Serialize the python dictionary into a traceable stack.
inline Stack toTraceableStack(const py::dict& inputs) {
  Stack res;
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    if (THPVariable_Check(it->second.ptr())) {
      res.push_back(toIValue(it->second, tryToInferType(it->second).type()));
    }
  }
  return res;
}

inline IValue createGenericList(py::handle obj, const TypePtr& elem_type) {
  auto elems = c10::impl::GenericList(elem_type);
  for (auto elem : obj) {
    elems.push_back(toIValue(elem, elem_type));
  }
  return IValue(elems);
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
  return IValue(elems);
}

template <class T>
inline void guardAgainstNamedTensor(const T& var) {
  TORCH_CHECK(
      !var.has_names(),
      "NYI: Named tensors are currently unsupported in TorchScript. As a  "
      "workaround please drop names via `tensor = tensor.rename(None)`.");
}

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
      !stack.empty(),
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
  std::vector<PyObject*> overloaded_args;
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
  if (!overloaded_args.empty()) {
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

TORCH_PYTHON_API std::pair<std::shared_ptr<Operator>, Stack> getOpWithStack(
    const std::vector<std::shared_ptr<Operator>>& operations,
    py::args args,
    const py::kwargs& kwargs);

TORCH_PYTHON_API py::object invokeOperatorFromPython(
    const std::vector<std::shared_ptr<Operator>>& operations,
    py::args args,
    const py::kwargs& kwargs,
    c10::optional<c10::DispatchKey> dk = c10::nullopt);

TORCH_PYTHON_API py::object _get_operation_for_overload_or_packet(
    const std::vector<std::shared_ptr<Operator>>& operations,
    Symbol symbol,
    py::args args,
    const py::kwargs& kwargs,
    bool is_overload,
    c10::optional<c10::DispatchKey> dk = c10::nullopt);

} // namespace torch::jit
