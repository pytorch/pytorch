#include <ATen/NamedTensorUtils.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/util/irange.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_torch_functions.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/csrc/autograd/utils/error_messages.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/Placement.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/pyobject_preservation.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>

#include <torch/csrc/utils/torch_dispatch_mode.h>

#include <ATen/ATen.h>

#include <c10/core/SymIntArrayRef.h>
#include <structmember.h>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace at;
using namespace torch;
using namespace torch::autograd;

namespace {
class OperatorArgsKwargsView {
 public:
  OperatorArgsKwargsView(
      const c10::OperatorHandle& op,
      const std::vector<c10::IValue>& arguments);
  using args_iterator = const c10::IValue*;

  args_iterator args_begin() const {
    return arguments_.data();
  }

  args_iterator args_end() const {
    return arguments_.data() + positional_default_start_;
  }

  auto num_positional_args() const {
    return positional_default_start_;
  }

  auto kwarg_start_index() const {
    return first_non_default_kwarg_;
  }

  struct kwargs_iterator {
    kwargs_iterator() = default;
    kwargs_iterator(const OperatorArgsKwargsView* parent, size_t current)
        : parent_(parent), current_(current) {}

    kwargs_iterator(const kwargs_iterator&) = default;
    kwargs_iterator& operator=(const kwargs_iterator&) = default;

    kwargs_iterator& operator++() {
      do {
        current_++;
      } while (current_ < parent_->arguments_.size() &&
               parent_->is_default(current_));
      return *this;
    }

    kwargs_iterator operator++(int) {
      auto copy = *this;
      ++(*this);
      return copy;
    }

    const c10::IValue& operator*() const {
      return parent_->arguments_[current_];
    }

    const c10::IValue* operator->() const {
      return &operator*();
    }

    int64_t underlying_index() const {
      return current_;
    }

    bool operator==(const kwargs_iterator& rhs) const {
      return parent_ == rhs.parent_ && current_ == rhs.current_;
    }

    bool operator!=(const kwargs_iterator& rhs) {
      return !(*this == rhs);
    }

   private:
    const OperatorArgsKwargsView* parent_ = nullptr;
    size_t current_ = 0;
  };

  kwargs_iterator kwargs_begin() const {
    return kwargs_iterator(this, first_non_default_kwarg_);
  }

  kwargs_iterator kwargs_end() const {
    return kwargs_iterator(this, arguments_.size());
  }

 private:
  bool is_default(size_t idx) const {
    const auto& arg = op_.schema().arguments()[idx];
    if (!arg.default_value().has_value()) {
      return false;
    }
    const auto& default_ivalue = *arg.default_value();
    const auto& ivalue = arguments_[idx];
    if (default_ivalue != ivalue) {
      return false;
    }
    return true;
  }

  const c10::OperatorHandle& op_;
  c10::ArrayRef<c10::IValue> arguments_;
  // About all the pointers:
  //
  // f(int x, int y = 0, *, int z = 0)
  //                                  ^- arguments.size()
  //                        ^- kwarg_only_start
  //          ^- positional_default_start
  //   ^- 0
  int64_t positional_default_start_;
  int64_t first_non_default_kwarg_;
};

OperatorArgsKwargsView::OperatorArgsKwargsView(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments)
    : op_(op), arguments_(arguments) {
  // Find the split point between kwarg-only and regular.  Since most functions
  // don't have kwarg-only arguments, it is more efficient to scan from the
  // right (but ideally, this would just be precomputed in FunctionSchema
  // itself).  (NB: minus one in the loop is because we're testing if the
  // *next* argument is kwarg-only before we advance the starting index)
  const int64_t signed_arguments_size = static_cast<int64_t>(arguments.size());
  int64_t kwarg_only_start = signed_arguments_size;
  for (; kwarg_only_start > 0; kwarg_only_start--) {
    const auto& arg = op.schema().arguments()[kwarg_only_start - 1];
    if (!arg.kwarg_only()) {
      break;
    }
  }

  // Find the first positional argument that isn't defaulted
  positional_default_start_ = kwarg_only_start;
  for (; positional_default_start_ > 0; positional_default_start_--) {
    if (!is_default(positional_default_start_ - 1)) {
      break;
    }
  }

  // kwargs_iterator will skip default kwargs when incremented, but we
  // need to skip any initial run of default kwargs ourselves.
  first_non_default_kwarg_ = kwarg_only_start;
  for (; first_non_default_kwarg_ < signed_arguments_size;
       ++first_non_default_kwarg_) {
    if (!is_default(first_non_default_kwarg_)) {
      break;
    }
  }
}
} // namespace

std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments) {
  TORCH_CHECK(
      PyGILState_Check(),
      "GIL must be held before you call parseIValuesToPyArgsKwargs");
  const auto& schema = op.schema();
  py::dict kwargs;

  OperatorArgsKwargsView args_kwargs(op, arguments);
  auto args = py::reinterpret_steal<py::object>(
      PyTuple_New(args_kwargs.num_positional_args()));

  auto schemaAwareToPyObject =
      [&schema](size_t idx, const c10::IValue& argument) -> py::object {
    const auto& arg = schema.arguments()[idx];
    auto match = [&](c10::TypeKind kind) {
      const auto& t = arg.real_type();
      if (t->kind() == kind)
        return true;
      if (auto opt_t = t->cast<c10::OptionalType>()) {
        if (opt_t->getElementType()->kind() == kind)
          return true;
      }
      return false;
    };
    if (argument.isNone()) {
      return py::none();
    } else if (match(c10::ScalarTypeType::Kind)) {
      auto* obj = getTHPDtype(static_cast<c10::ScalarType>(argument.toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::LayoutType::Kind)) {
      auto* obj = getTHPLayout(static_cast<c10::Layout>(argument.toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::MemoryFormatType::Kind)) {
      return py::cast(static_cast<c10::MemoryFormat>(argument.toInt()));
    } else {
      return torch::jit::toPyObject(argument);
    }
  };

  // Populate positional arguments
  size_t idx = 0;
  for (auto argument_it = args_kwargs.args_begin();
       argument_it != args_kwargs.args_end();
       ++argument_it) {
    PyTuple_SET_ITEM(
        args.ptr(),
        idx,
        schemaAwareToPyObject(idx, *argument_it).release().ptr());
    idx++;
  }

  // Populate keyword arguments
  for (auto argument_it = args_kwargs.kwargs_begin();
       argument_it != args_kwargs.kwargs_end();
       ++argument_it) {
    const auto& arg = schema.arguments()[argument_it.underlying_index()];
    kwargs[py::cast(arg.name())] =
        schemaAwareToPyObject(argument_it.underlying_index(), *argument_it);
  }
  return std::make_pair(std::move(args), std::move(kwargs));
}

void pushPyOutToStack(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    py::object out,
    const char* msg) {
  TORCH_CHECK(
      PyGILState_Check(), "GIL must be held before you call pushPyOutToStack");
  const auto& schema_returns = op.schema().returns();
  const auto num_returns = schema_returns.size();
  if (num_returns == 0) {
    // Check that we got a None return from Python. Anything else is an error.
    TORCH_CHECK(
        out.is_none(),
        "Expected ",
        msg,
        " for ",
        op.operator_name(),
        " to return None but it returned something else instead.");
  } else if (num_returns == 1) {
    torch::jit::push(
        stack, torch::jit::toIValue(out.ptr(), schema_returns[0].real_type()));
  } else {
    auto outs = py::cast<py::sequence>(out);
    for (const auto idx : c10::irange(outs.size())) {
      torch::jit::push(
          stack,
          torch::jit::toIValue(
              outs[idx].ptr(), schema_returns[idx].real_type()));
    }
  }
}

namespace {

c10::TensorImpl::SizesStridesPolicy parseSizesStridesPolicyArgument(
    std::string_view arg) {
  if (arg == "strides") {
    return c10::TensorImpl::SizesStridesPolicy::CustomStrides;
  }

  if (arg == "sizes") {
    return c10::TensorImpl::SizesStridesPolicy::CustomSizes;
  }

  TORCH_CHECK_VALUE(
      false,
      "Unknown sizes_strides_policy: ",
      arg,
      "; expected 'strides' or 'sizes'");
}
} // anonymous namespace

PyObject* THPVariableClass = nullptr;

PyObject* ParameterClass = nullptr;

static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    const at::TensorBase& _var,
    bool allow_preexisting_pyobj = false,
    std::optional<bool> has_torch_dispatch_if_known = std::nullopt);

// clang-tidy gets confused by static const
static constexpr const char* VOLATILE_WARNING =
    "volatile was removed and now has no effect. Use "
    "`with torch.no_grad():` instead.";

static bool check_has_torch_dispatch(PyObject* obj) {
  PyTypeObject* tp = Py_TYPE(obj);
  if (THPVariable_CheckTypeExact(tp)) {
    return false;
  }
  py::object attr = PyObject_FastGetAttrString(obj, "__torch_dispatch__");
  return (
      attr.ptr() != nullptr &&
      attr.ptr() != torch::disabled_torch_dispatch_impl());
}

// NOLINTNEXTLINE(*-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyObject* device_to_py_class_[static_cast<size_t>(
    c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)];

void registerPythonTensorClass(
    const std::string& device,
    PyObject* python_tensor_class) {
  c10::Device dev(device);

  TORCH_CHECK(
      dev.type() == kXLA, "Only the python class for XLA can be overridden");
  if (device_to_py_class_[static_cast<size_t>(dev.type())] != nullptr) {
    TORCH_WARN(
        "Overriding a previously registered python class for ", dev.str());
  }

  device_to_py_class_[static_cast<size_t>(dev.type())] = python_tensor_class;
}

static PyObject* getPythonTensorClass(c10::Device d) {
  return device_to_py_class_[static_cast<size_t>(d.type())];
}

void activateGPUTrace() {
  c10::impl::GPUTrace::set_trace(getPyInterpreter());
}

PyObject* THPVariable_Wrap(const at::TensorBase& var) {
  if (!var.defined()) {
    Py_RETURN_NONE;
  }

  if (c10::impl::HermeticPyObjectTLS::get_state()) {
    return THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, var);
  }

  std::optional<PyObject*> mb_obj =
      var.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          /*ignore_hermetic_tls=*/false);
  if (mb_obj.has_value()) {
    auto obj = *mb_obj;
    if (obj) {
      if (var.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj()) {
        // C++ owns the Python object; this implies there weren't any other
        // owning references to the Python object.  Since we're making the
        // object "live" again on Python side, let's flip back the ownership
        // (Python owns C++) as it would now be unsound to deallocate the C++
        // object if all C++ references go to zero
        var.unsafeGetTensorImpl()->pyobj_slot()->set_owns_pyobj(false);
        reinterpret_cast<THPVariable*>(obj)->cdata =
            MaybeOwned<Variable>::owned(Variable(var));
        // NB: incref is not necessary, because we are "stealing" the previous
        // ownership from the Variable to return it here for the wrap
        return obj;
      }
      Py_INCREF(obj);
      return obj;
    }
    // TODO: a better invariant is that if we tagged, we MUST have a valid
    // PyObject.  That's PyObject preservation
    // (https://github.com/pytorch/pytorch/pull/56017).  Prior to this PR
    // being a thing, the PyObject field will get cleared when all references
    // to the Python object are removed.
  }

  if (C10_LIKELY(var.device().type() != c10::kXLA)) {
    return THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, var);
  }

  if (auto clazz = getPythonTensorClass(var.device())) {
    return THPVariable_NewWithVar((PyTypeObject*)clazz, var);
  }

  return THPVariable_NewWithVar((PyTypeObject*)THPVariableClass, var);
}

static bool isResurrectable(THPVariable* self) {
  // We want to divide this check into 2 cases.

  // 1. C++ owns PyObject (in this case, self->cdata.unsafeIsBorrowed() is
  // true). You might think that in this case, it is impossible for tp_clear to
  // be called: surely the C++ reference to the PyObject is keeping it live? And
  // you'd be right! In fact, when C++ owns the PyObject, we have an invariant
  // that the refcount on the PyObject should be precisely one (because if you
  // take out another reference to the PyObject, we're supposed to flip the
  // ownership pointer back). In reality, you can violate this invariant
  // temporarily with weak references, so we don't test for it in asserts.

  // 2. PyObject owns C++ (in this case, self->cdata.unsafeIsBorrowed() is
  // false). In this case, tp_clear can get called if the PyObject is referenced
  // from a dead cycle, and nowhere else. But if resurrection did not occur,
  // then the reference to C++ from the PyObject must be the ONLY reference to
  // the C++ object.
  if (self->cdata.unsafeIsBorrowed()) {
    return false;
  }
  auto const& tensor = THPVariable_Unpack(self);
  if (!tensor.defined() || tensor.use_count() <= 1) {
    return false;
  }
  // Check if this is hermetic. If it is, no resurrection.
  if (tensor.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
          /*ignore_hermetic_tls=*/false) != (PyObject*)self) {
    return false;
  }
  return true;
}

// returns true if successfully rezzed; if so, cancel the
// rest of deallocation
static bool THPVariable_tryResurrect(THPVariable* self) {
  const auto& tensor = THPVariable_Unpack(self);

  if (!isResurrectable(self)) {
    return false;
  }

  // At this point, we are definitely going to resurrect the tensor. So, the
  // tensor better be defined :)
  TORCH_INTERNAL_ASSERT(tensor.defined());

  // There are other C++ owners of the tensor.  Flip ownership
  // so that C++ owns this Python object, and cancel deallocation.
  TORCH_INTERNAL_ASSERT(
      !tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj());

  c10::TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();
  auto maybe_pyobj = tensor_impl->pyobj_slot()->check_pyobj(
      /*ignore_hermetic_tls=*/false);

  TORCH_INTERNAL_ASSERT(
      maybe_pyobj.has_value(),
      "Trying to preserve a Python tensor whose PyObjectSlot does not have a PyObject");

  tensor_impl->pyobj_slot()->set_owns_pyobj(true);

  // Resurrect the Python object.  This is something CPython does
  // internally occasionally, see
  // https://github.com/python/cpython/blob/b98eba5bc2ffbe7a0ed49d540ebc4f756ae61985/Objects/object.c#L248-L259
  // so we just copy the pattern here.  Note that we don't have to worry
  // about saving and restoring the refcount (as the quoted code does)
  // because we actually DO need to reset the refcount to one here, we
  // can't assume that some other code has taken care of it.
  // NB: this will overreport _Py_RefTotal but based on inspection of object.c
  // there is no way to avoid this

  // When resurrecting, we MUST use _Py_NewReference and not Py_INCREF to
  // ensure the PyObject is in a valid state
  _Py_NewReference((PyObject*)self);

  // Flip THPVariable to be non-owning
  // (near use-after-free miss here: fresh MaybeOwned is created breaking
  // reference on Tensor in struct BEFORE we overwrite the old one)
  TORCH_INTERNAL_ASSERT(!c10::impl::HermeticPyObjectTLS::get_state());
  self->cdata = MaybeOwned<Variable>::borrowed(tensor);

  // NB: At this point, tensor *could* be dead (e.g., some other C++ thread
  // decrefed it.)  At this point, it is probably waiting on the GIL to
  // deallocate the Python object and will kill self, BUT NOT YET.

  return true;
}

static int THPFake_traverse(THPVariable* self, visitproc visit, void* arg) {
  TORCH_INTERNAL_ASSERT(
      false, "TensorBase tp_traverse function was not overridden properly");
  return 0;
}

static int THPFake_clear(THPVariable* self) {
  TORCH_INTERNAL_ASSERT(
      false, "TensorBase tp_clear function was not overridden properly");
  return 0;
}

static PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs);

static PyObject* THPVariable_fix_weakref(PyObject* self, PyObject* noargs) {
  const auto& var = THPVariable_Unpack(self);
  Py_DECREF(THPVariable_Wrap(var));
  Py_RETURN_NONE;
}

// Maps the given python callable over a vector of items, returning a vector
// of the same type of items.
template <typename T>
static std::vector<T> map_py_func(
    const py::function& func,
    const std::vector<T>& items) {
  std::vector<T> new_items;
  new_items.reserve(items.size());
  for (auto& item : items) {
    new_items.emplace_back(py::cast<T>(func(item)));
  }
  return new_items;
}

template <>
std::vector<at::Tensor> map_py_func(
    const py::function& func,
    const std::vector<at::Tensor>& items) {
  std::vector<at::Tensor> new_items;
  new_items.reserve(items.size());
  for (auto& item : items) {
    auto output = func(item);
    if (output.is(py::none())) {
      // treat None value as an undefined tensor
      new_items.emplace_back();
    } else {
      new_items.emplace_back(py::cast<at::Tensor>(output));
    }
  }
  return new_items;
}

static PyObject* view_func_impl(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs,
    bool check_has_same_meta) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(_self);

  static PythonArgParser parser({
      "_view_func(Tensor new_base, PyObject* symint_visitor_fn=None, PyObject* tensor_visitor_fn=None)",
  });
  ParsedArgs<3> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  auto new_base = r.tensor(0);
  PyObject* symint_visitor_fn = r.pyobject(1);
  PyObject* tensor_visitor_fn = r.pyobject(2);

  // Ensure that self is indeed a backward differentiable view
  // If not, we return an undefined Tensor (None) and let the user handle it.
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  at::Tensor out;
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    const auto& view_info = diff_view_meta->get_backward_view();
    // Ensure that the newly provided base is similar to the original base
    if (!check_has_same_meta ||
        torch::autograd::utils::has_same_meta(new_base, view_info.base_)) {
      // Do the actual view replay
      if (view_info.has_view_fn()) {
        auto& view_func = view_info.view_fn();

        // Determine new SymInt / tensor state as needed.
        std::optional<std::vector<c10::SymInt>> new_symints = std::nullopt;
        if (symint_visitor_fn != Py_None) {
          new_symints = map_py_func(
              py::cast<py::function>(symint_visitor_fn),
              view_func.get_symints());
        }

        std::optional<std::vector<at::Tensor>> new_tensors = std::nullopt;
        if (tensor_visitor_fn != Py_None) {
          new_tensors = map_py_func(
              py::cast<py::function>(tensor_visitor_fn),
              view_func.get_tensors());
        }

        // call view func
        if (new_symints.has_value() || new_tensors.has_value()) {
          out = (*view_func.clone_and_set(new_symints, new_tensors))(new_base);
        } else {
          out = view_func(new_base);
        }
      } else {
        out = new_base.as_strided(
            self.sizes(), self.strides(), self.storage_offset());
      }
    }
  }
  return THPVariable_Wrap(out);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_view_func(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  return view_func_impl(self_, args, kwargs, /*check_has_same_meta=*/true);
}

static PyObject* THPVariable_view_func_unsafe(
    PyObject* self_,
    PyObject* args,
    PyObject* kwargs) {
  return view_func_impl(self_, args, kwargs, /*check_has_same_meta=*/false);
}

static PyObject* rev_view_func_impl(PyObject* self_, PyObject* arg) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(self_);
  TORCH_CHECK(
      THPVariable_Check(arg),
      "_rev_view_func expect a single argument that is a Tensor");
  const auto& new_view = THPVariable_Unpack(arg);

  // Ensure that self is indeed a backward differentiable view
  // If not, we return an undefined Tensor (None) and let the user handle it.
  auto diff_view_meta = torch::autograd::impl::get_view_autograd_meta(self);
  at::Tensor out;
  if (diff_view_meta && diff_view_meta->has_bw_view()) {
    const auto& view_info = diff_view_meta->get_backward_view();
    // Do the actual view replay
    TORCH_CHECK(view_info.has_view_fn(), "No _rev_view_func() found");
    out = view_info.rev_view_fn()(new_view);
  }
  return THPVariable_Wrap(out);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_rev_view_func_unsafe(
    PyObject* self_,
    PyObject* arg) {
  return rev_view_func_impl(self_, arg);
}

// Instantiates a subclass of self with the same data.
static PyObject* THPVariable_as_subclass(
    PyObject* _self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  const auto& self = THPVariable_Unpack(_self);
  static PythonArgParser parser({
      "as_subclass(PyObject* cls)",
  });
  ParsedArgs<1> parsed_args{};
  auto r = parser.parse(_self, args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");
  // guard completely turns off torch dispatch modes, doesn't just pop off the
  // stack
  torch_dispatch_mode::StashTorchDispatchStackGuard td_g;
  c10::impl::DisablePythonDispatcher dpd_g;
  return THPVariable_NewWithVar((PyTypeObject*)cls, self.alias());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_make_subclass(
    PyObject* _ignored,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "_make_subclass(PyObject* cls, Tensor data, bool require_grad=False, *, std::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False, Device? device_for_backend_keys=None)",
  });
  ParsedArgs<7> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);
  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");
  // guard completely turns off torch dispatch modes, doesn't just pop off the
  // stack
  torch_dispatch_mode::StashTorchDispatchStackGuard td_g;
  c10::impl::DisablePythonDispatcher dpd_g;
  auto data =
      r.tensor(1).detach(); // creates a fresh Tensor (DEFINITELY_UNINITIALIZED)
  // We set `data`'s `allow_tensor_metadata_change` to true here, because we
  // want to allow the following use case for backward compatibility:
  //
  // ```python
  // rnn = torch.nn.RNN(100, 100, 2)
  // # The following calls `torch._cudnn_rnn_flatten_weight(rnn._flat_weights,
  // ...)`, # which changes storage of `rnn`'s weights in-place
  // rnn.flatten_parameters()
  // ```
  data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
  data.set_requires_grad(r.toBool(2));
  const auto sizes_strides_policy = r.stringViewOptional(3);
  if (sizes_strides_policy.has_value()) {
    data.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
        parseSizesStridesPolicyArgument(*sizes_strides_policy));
  }
  if (r.toBool(4)) {
    data.unsafeGetTensorImpl()->set_python_custom_device(true);
  }
  if (r.toBool(5)) {
    data.unsafeGetTensorImpl()->set_python_custom_layout(true);
  }
  if (!r.isNone(6)) {
    data.unsafeGetTensorImpl()->_change_backend_component_keys(r.device(6));
  }

  return THPVariable_NewWithVar((PyTypeObject*)cls, data);
  END_HANDLE_TH_ERRORS
}

// Shared code factored out of THPVariable_make_wrapper_subclass and
// THPVariable_dtensor__new__.
static Tensor make_tensor_for_subclass_helper(
    SymIntArrayRef sym_sizes,
    OptionalSymIntArrayRef sym_strides,
    const std::optional<c10::SymInt>& sym_storage_offset,
    const TensorOptions& options,
    const std::optional<c10::SymInt>& storage_size,
    std::optional<DispatchKeySet> extra_dispatch_keys) {
  AutoDispatchBelowADInplaceOrView guard{}; // TODO: Remove.
  tracer::impl::NoTracerDispatchMode tracer_guard{};

  c10::SymInt size_bytes;
  auto dtype_itemsize = static_cast<int64_t>(options.dtype().itemsize());

  if (storage_size.has_value()) {
    size_bytes = storage_size.value();
  } else if (sym_strides.has_value()) {
    size_bytes = at::detail::computeStorageNbytes(
        sym_sizes,
        sym_strides.value(),
        dtype_itemsize,
        sym_storage_offset.value_or(0));
  } else {
    size_bytes = at::detail::computeStorageNbytesContiguous(
        sym_sizes, dtype_itemsize, sym_storage_offset.value_or(0));
  }

  // We use storages **only** to track aliasing of subclasses during tracing.
  // The actual data pointers are not valid.
  Storage storage{
      Storage::use_byte_size_t{},
      size_bytes,
      at::DataPtr{nullptr, options.device()},
      /*allocator=*/c10::GetAllocator(c10::kMeta),
      /*resizable=*/true};

  auto keys = c10::DispatchKeySet({options.computeDispatchKey()});
  if (extra_dispatch_keys.has_value()) {
    keys = keys | *extra_dispatch_keys;
  }
  Tensor tensor = at::detail::make_tensor<TensorImpl>(
      std::move(storage), keys, options.dtype());

  TensorImpl* tensor_impl = tensor.unsafeGetTensorImpl();

  if (sym_strides.has_value()) {
    tensor_impl->set_sizes_and_strides(
        sym_sizes, sym_strides.value(), sym_storage_offset);
  } else {
    TORCH_CHECK(
        !sym_storage_offset.has_value(),
        "setting storage offset without stride not supported");
    tensor_impl->generic_set_sizes_contiguous(sym_sizes);
  }
  return tensor;
}

static PyObject* THPVariable_make_wrapper_subclass(
    PyObject* /*unused*/,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // NB: pin_memory doesn't actually do anything
  // TODO: strides variant?

  // cls: Python subclass type
  // size, strides, storage_offset, memory_format, dtype: self-explanatory
  // layout: memory layout, e.g. for types of Nested Tensors or other sparse
  //         tensors
  // pin_memory, requires_grad: self-explanatory
  // dispatch_sizes_strides_policy: string - which sizes/strides we should
  //                                dispatch to a custom python implementation.
  // dispatch_device: whether to dispatch to a custom python implementation
  //                  for device
  // dispatch_layout: whether to dispatch to a custom python implementation
  //                  for layout
  // _extra_dispatch_keys: additional dispatch keys to add to the tensor
  // storage_size: if provided, skip storage size calculation and just use the
  //               value provided. One use case is for Nested Tensor, where the
  //               storage size cannot be calculated from the sizes/strides
  //               (because they contain a NestedInt).
  static PythonArgParser parser({
      "_make_wrapper_subclass(PyObject* cls, SymIntArrayRef size, SymIntArrayRef? strides=None, "
      "SymInt? storage_offset=None, MemoryFormat? memory_format=None, ScalarType dtype=None, "
      "Layout layout=torch.strided, Device device=None, bool pin_memory=False, bool requires_grad=False, "
      "std::string_view? dispatch_sizes_strides_policy=None, bool dispatch_device=False, bool dispatch_layout=False, "
      "DispatchKeySet _extra_dispatch_keys=None, SymInt? storage_size=None)",
  });
  ParsedArgs<15> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);

  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");

  // This is an important safety check; without it, the default behavior will be
  // to continue on to the underlying CPU/CUDA kernel advertised by the dispatch
  // key, which will immediately segfault because the data pointer is null.  By
  // forcing users to define __torch_dispatch__ we ensure this does not happen
  // TODO: This check is not complete; because the user can disable torch
  // dispatch and then go again, triggering segfault.  TBH I'm thinking I want
  // to delete this function entirely
  py::object attr = PyObject_FastGetAttrString(cls, "__torch_dispatch__");
  TORCH_CHECK_TYPE(
      attr.ptr() != nullptr &&
          attr.ptr() != torch::disabled_torch_dispatch_impl(),
      ((PyTypeObject*)cls)->tp_name,
      " must define __torch_dispatch__");

  const auto options = TensorOptions()
                           .dtype(r.scalartype(5))
                           .device(r.device(7))
                           .layout(r.layoutOptional(6))
                           // NB: long standing issue, requires_grad is not
                           // respected here; you have to set it post facto, see
                           // https://github.com/pytorch/pytorch/issues/26428
                           // .requires_grad(r.toBool(7))
                           .pinned_memory(r.toBool(8));

  // don't bother releasing GIL here, as we are not allocating any nontrivial
  // data
  auto sym_sizes = r.symintlist(1);
  auto sym_strides_own = r.symintlistOptional(2);
  Tensor tensor = make_tensor_for_subclass_helper(
      /*sym_sizes=*/r.symintlist(1),
      /*sym_strides=*/r.symintlistOptional(2),
      /*sym_storage_offset=*/r.toSymIntOptional(3),
      options,
      /*storage_size=*/r.toSymIntOptional(14),
      r.toDispatchKeySetOptional(13));

  const auto sizes_strides_policy = r.stringViewOptional(10);
  if (sizes_strides_policy.has_value()) {
    tensor.unsafeGetTensorImpl()->set_python_custom_sizes_strides(
        parseSizesStridesPolicyArgument(*sizes_strides_policy));
  }

  tensor.set_requires_grad(r.toBool(9));

  if (r.toBool(11)) {
    tensor.unsafeGetTensorImpl()->set_python_custom_device(true);
  }
  if (r.toBool(12)) {
    tensor.unsafeGetTensorImpl()->set_python_custom_layout(true);
  }

  return THPVariable_NewWithVar(
      (PyTypeObject*)cls,
      tensor,
      // false is the default
      /*allow_preexisting_pyobj=*/false,
      // we checked __torch_dispatch__ above; avoid checking again.
      /*has_torch_dispatch_if_known=*/true);
  END_HANDLE_TH_ERRORS
}

static py::handle get_dtensor_spec_class() {
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch")
            .attr("distributed")
            .attr("tensor")
            .attr("_dtensor_spec")
            .attr("DTensorSpec");
      })
      .get_stored();
#else
  static py::handle dtensor_spec_class = py::object(py::module::import("torch")
                                                        .attr("distributed")
                                                        .attr("tensor")
                                                        .attr("_dtensor_spec")
                                                        .attr("DTensorSpec"))
                                             .release();
  return dtensor_spec_class;
#endif
}

static bool arg_type_tensor_or_tensor_list_like(py::handle arg) {
  const auto dtensor_spec_class = get_dtensor_spec_class();
  if (py::isinstance(arg, dtensor_spec_class)) {
    return true;
  }
  if (!PyList_Check(arg.ptr())) {
    return false;
  }
  py::list arg_list = py::reinterpret_borrow<py::list>(arg);
  for (const auto e : arg_list) {
    if (!e.is_none() && !py::isinstance(e, dtensor_spec_class)) {
      return false;
    }
  }
  return true;
}

#if IS_PYTHON_3_11_PLUS
#define MAYBE_FOR_EACH_PYTHON_3_10_MINUS_DTENSOR_INTERNED_STRING(_)
#else
#define MAYBE_FOR_EACH_PYTHON_3_10_MINUS_DTENSOR_INTERNED_STRING(_) _(__name__)
#endif

#define FOR_EACH_DTENSOR_INTERNED_STRING(_)                   \
  MAYBE_FOR_EACH_PYTHON_3_10_MINUS_DTENSOR_INTERNED_STRING(_) \
  _(_comparison_key)                                          \
  _(_local_tensor)                                            \
  _(_spec)                                                    \
  _(args_schema)                                              \
  _(kwargs_schema)                                            \
  _(op)                                                       \
  _(schema_info)                                              \
  _(shape)                                                    \
  _(size)                                                     \
  _(static_argnum)                                            \
  _(static_kwargkey)                                          \
  _(stride)                                                   \
  _(tensor_meta)

struct DTensorInternedStrings {
#define DECLARE_INTERNED_STRING_VARIABLE(s) PyObject* s;
  FOR_EACH_DTENSOR_INTERNED_STRING(DECLARE_INTERNED_STRING_VARIABLE)
#undef DECLARE_INTERNED_STRING_VARIABLE
};

static DTensorInternedStrings dtensor_interned_strings;

static bool intern_dtensor_strings() {
#define INTERN_DTENSOR_STRING(s)                                           \
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(dtensor_interned_strings.s == nullptr); \
  dtensor_interned_strings.s = PyUnicode_InternFromString(#s);             \
  if (dtensor_interned_strings.s == nullptr) {                             \
    return false;                                                          \
  }

  FOR_EACH_DTENSOR_INTERNED_STRING(INTERN_DTENSOR_STRING);
#undef INTERN_DTENSOR_STRING
  return true;
}

static bool checked_not(PyObject* obj) {
  int result = PyObject_Not(obj);
  if (result == -1) {
    throw py::error_already_set();
  }
  return result;
}

static c10::SymDimVector tuple_to_symintlist(PyObject* obj) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(PyTuple_Check(obj));
  c10::SymDimVector res;
  const auto size = PyTuple_GET_SIZE(obj);
  res.reserve(size);
  for (const auto idx : c10::irange(size)) {
    PyObject* item = PyTuple_GET_ITEM(obj, idx);
    if (THPUtils_checkLongExact(item)) {
      res.emplace_back(THPUtils_unpackLong(item));
    } else if (torch::is_symint(py::handle(item))) {
      res.push_back(py::handle(item).cast<c10::SymInt>());
    } else {
      // N.B. torch.Tensor.__index__ exists, so this should handle
      // scalar Tensors fine.
      res.emplace_back(THPUtils_unpackIndex(item));
    }
  }
  return res;
}

// DTensor-specific variant of make_wrapper_subclass to minimize DTensor
// overhead.
static PyObject* THPVariable_dtensor_new(
    PyObject* /*unused*/,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
      "_dtensor__new__(PyObject* cls, Tensor local_tensor, PyObject* spec, bool requires_grad)",
  });
  ParsedArgs<4> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  PyObject* cls = r.pyobject(0);

  TORCH_CHECK_TYPE(
      PyType_Check(cls),
      "cls must be a type (got ",
      Py_TYPE(cls)->tp_name,
      ")");

#ifndef NDEBUG
  // This is specifically for making a DTensor, which we know defines
  // __torch_dispatch__. Check anyway in debug builds in case somebody
  // removes it.
  py::object attr = PyObject_FastGetAttrString(cls, "__torch_dispatch__");
  TORCH_CHECK_TYPE(
      attr.ptr() != nullptr &&
          attr.ptr() != torch::disabled_torch_dispatch_impl(),
      ((PyTypeObject*)cls)->tp_name,
      " must define __torch_dispatch__");
#endif

  const auto& local_tensor = r.tensor(1);
  const bool requires_grad = r.toBool(3);
  if (local_tensor.requires_grad() && !requires_grad) {
    TORCH_WARN(
        "To construct DTensor from torch.Tensor, it's recommended to use "
        "local_tensor.detach() and make requires_grad consistent.");
  }
  const auto options = TensorOptions()
                           .dtype(local_tensor.dtype())
                           .device(local_tensor.device())
                           .layout(local_tensor.layout());

  DispatchKeySet extra_dispatch_keys;
  const auto tensor_keys = local_tensor.key_set();
  if (tensor_keys.has(c10::DispatchKey::Conjugate)) {
    extra_dispatch_keys = extra_dispatch_keys.add(c10::DispatchKey::Conjugate);
  }
  if (tensor_keys.has(c10::DispatchKey::Negative)) {
    extra_dispatch_keys = extra_dispatch_keys.add(c10::DispatchKey::Negative);
  }

  py::handle spec = py::handle(r.pyobject(2));
  const auto tensor_meta = spec.attr(dtensor_interned_strings.tensor_meta);
  TORCH_CHECK(!tensor_meta.is_none());
  const auto sizes = tensor_meta.attr(dtensor_interned_strings.shape);
  TORCH_CHECK(
      PyTuple_Check(sizes.ptr()), "spec.tensor_meta.shape must be a tuple");
  const auto stride = tensor_meta.attr(dtensor_interned_strings.stride);
  TORCH_CHECK(
      PyTuple_Check(stride.ptr()), "spec.tensor_meta.stride must be a tuple");

  Tensor tensor = make_tensor_for_subclass_helper(
      /*sym_sizes=*/tuple_to_symintlist(sizes.ptr()),
      /*sym_strides=*/tuple_to_symintlist(stride.ptr()),
      /*sym_storage_offset=*/std::nullopt,
      options,
      /*storage_size=*/std::nullopt,
      extra_dispatch_keys);
  tensor.set_requires_grad(requires_grad);
  py::object py_tensor =
      py::reinterpret_steal<py::object>(THPVariable_NewWithVar(
          (PyTypeObject*)cls,
          tensor,
          // false is the default
          /*allow_preexisting_pyobj=*/false,
          // we know DTensor has __torch_dispatch__; avoid checking again.
          /*has_torch_dispatch_if_known=*/true));
  py_tensor.attr(dtensor_interned_strings._spec) = spec;
  py_tensor.attr(dtensor_interned_strings._local_tensor) = local_tensor;
  return py_tensor.release().ptr();
  END_HANDLE_TH_ERRORS
}

static bool DTensor_OpSchema_recompute_comparison_key_impl(
    PyObject* self,
    const py::tuple& args_schema) {
  py::object static_kwargkey;
  size_t static_argnum = 0;
  const py::handle self_handle = py::handle(self);
  const py::handle schema_info =
      self_handle.attr(dtensor_interned_strings.schema_info);
  if (checked_not(schema_info.ptr())) {
    static_argnum = args_schema.size();
    static_kwargkey = py::none();
  } else {
    static_argnum = py::cast<size_t>(
        schema_info.attr(dtensor_interned_strings.static_argnum));
    static_kwargkey =
        schema_info.attr(dtensor_interned_strings.static_kwargkey);
  }
  c10::SmallVector<py::object, 8> args_to_hash;
  size_t idx = 0;
  for (const auto& e : args_schema) {
    if (idx >= static_argnum || arg_type_tensor_or_tensor_list_like(e)) {
      if (PyList_Check(e.ptr())) {
        args_to_hash.push_back(
            py::reinterpret_steal<py::object>(PyList_AsTuple(e.ptr())));
      } else {
        args_to_hash.push_back(py::reinterpret_borrow<py::object>(e));
      }
    }
    idx++;
  }
  py::tuple args_to_hash_tup(args_to_hash.size());
  for (const auto idx : c10::irange(args_to_hash.size())) {
    args_to_hash_tup[idx] = std::move(args_to_hash[idx]);
  }
  PyObject* comparison_key = nullptr;
  if (!static_kwargkey.is_none()) {
    if (!PyList_Check(static_kwargkey.ptr())) {
      PyErr_SetString(
          PyExc_TypeError, "self.schema_info.static_kwargkey must be a list!");
      return false;
    }
    py::list static_kwargkey_list =
        py::reinterpret_borrow<py::list>(static_kwargkey);
    auto raw_kwargs_schema =
        self_handle.attr(dtensor_interned_strings.kwargs_schema);
    if (!PyDict_Check(raw_kwargs_schema.ptr())) {
      PyErr_SetString(PyExc_TypeError, "self.kwargs_schema must be a dict!");
      return false;
    }
    py::tuple kwargs_to_hash(static_kwargkey_list.size());
    int idx = 0;
    auto kwargs_schema = py::reinterpret_borrow<py::dict>(raw_kwargs_schema);
    for (const auto& k : static_kwargkey_list) {
      PyObject* item = PyDict_GetItemWithError(kwargs_schema.ptr(), k.ptr());
      if (item) {
        kwargs_to_hash[idx++] = py::reinterpret_borrow<py::object>(item);
      } else if (PyErr_Occurred()) {
        return false;
      } else {
        kwargs_to_hash[idx++] = py::none();
      }
    }
    comparison_key = PyTuple_Pack(
        3,
        self_handle.attr(dtensor_interned_strings.op).ptr(),
        args_to_hash_tup.ptr(),
        kwargs_to_hash.ptr());
  } else {
    comparison_key = PyTuple_Pack(
        2,
        self_handle.attr(dtensor_interned_strings.op).ptr(),
        args_to_hash_tup.release().ptr());
  }
  if (!comparison_key) {
    return false;
  }
  self_handle.attr(dtensor_interned_strings._comparison_key) =
      py::reinterpret_steal<py::object>(comparison_key);

  return true;
}

static PyObject* DTensor_OpSchema_recompute_comparison_key(
    PyObject* mod,
    PyObject* self) {
  HANDLE_TH_ERRORS
  const py::handle self_handle = py::handle(self);
  const py::handle raw_args_schema =
      self_handle.attr(dtensor_interned_strings.args_schema);
  if (!PyTuple_Check(raw_args_schema.ptr())) {
    PyErr_SetString(PyExc_TypeError, "DTensor.args_schema must be a tuple!");
    return nullptr;
  }
  py::tuple args_schema = py::reinterpret_borrow<py::tuple>(raw_args_schema);
  if (!DTensor_OpSchema_recompute_comparison_key_impl(self, args_schema)) {
    return nullptr;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* DTensor_OpSchema_post_init(PyObject* mod, PyObject* self) {
  HANDLE_TH_ERRORS
  const py::handle self_handle = py::handle(self);
  const py::handle raw_args_schema =
      self_handle.attr(dtensor_interned_strings.args_schema);
  if (!PyTuple_Check(raw_args_schema.ptr())) {
    PyErr_SetString(
        PyExc_TypeError,
        "DTensor_OpSchema_post_init requires self.args_schema to be a tuple!");
    return nullptr;
  }
  py::tuple args_schema = py::reinterpret_borrow<py::tuple>(raw_args_schema);
  if (!DTensor_OpSchema_recompute_comparison_key_impl(self, args_schema)) {
    return nullptr;
  }

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static py::list symint_array_to_list(SymIntArrayRef arr) {
  py::list result(arr.size());
  for (const auto idx : c10::irange(arr.size())) {
    result[idx] = py::cast(arr[idx]);
  }
  return result;
}

static PyObject* DTensor_compute_global_tensor_info_impl(
    const Tensor& tensor,
    py::handle mesh,
    const py::sequence& placements) {
  Py_ssize_t idx = 0;
  c10::SymDimVector tensor_shape(
      tensor.sym_sizes().begin(), tensor.sym_sizes().end());
  c10::SymDimVector tensor_strides(
      tensor.sym_strides().begin(), tensor.sym_strides().end());
  // NOTE: if this is a py::handle then this code stops working;
  // apparently we can't rely on the bound method to stick around.
  py::object mesh_size;
  for (const auto& placement : placements) {
    // TODO: C++ify DeviceMesh somehow; profiling seems
    // to say that nearly all our remaining time spent is spent
    // calling back into Python.
    const auto& cpp_placement = placement.cast<const distributed::Placement&>();
    if (const auto* cpp_shard =
            dynamic_cast<const distributed::Shard*>(&cpp_placement)) {
      const auto shard_dim = cpp_shard->dim;
      TORCH_CHECK(
          shard_dim >= 0,
          "Shard placements should have negative dims normalized in the user-facing APIs: ",
          py::cast<std::string>(py::str(placement)));
      const auto tensor_ndim = tensor.dim();
      TORCH_CHECK(
          shard_dim < tensor_ndim,
          "Sharding dim ",
          shard_dim,
          " greater than tensor ndim ",
          tensor_ndim,
          " for placement number ",
          idx);

      if (!mesh_size) {
        mesh_size = mesh.attr(dtensor_interned_strings.size);
      }
      const auto mesh_dim_size = py::cast<int64_t>(mesh_size(idx));
      tensor_shape[shard_dim] *= mesh_dim_size;
      // recover tensor stride by modifying the strides that are
      // larger than the current stride on the shard_dim.
      for (const auto i : c10::irange(tensor_strides.size())) {
        if (static_cast<int64_t>(i) != shard_dim &&
            tensor_strides[i] >= tensor_strides[shard_dim]) {
          tensor_strides[i] *= mesh_dim_size;
        }
      }
    } else if (!cpp_placement.is_replicate() && !cpp_placement.is_partial()) {
#if IS_PYTHON_3_11_PLUS
      const auto placement_type_name =
          py::str(py::handle(PyType_GetName(Py_TYPE(placement.ptr()))));
#else
      const auto placement_type_name =
          py::str(py::handle((PyObject*)Py_TYPE(placement.ptr()))
                      .attr(dtensor_interned_strings.__name__));
#endif
      return PyErr_Format(
          PyExc_RuntimeError,
          "placement type %s not supported!",
          py::cast<std::string>(placement_type_name).c_str());
    }
    idx++;
  }
  return py::make_tuple(
             symint_array_to_list(tensor_shape),
             symint_array_to_list(tensor_strides))
      .release()
      .ptr();
}

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
static constexpr const char compute_global_tensor_info_doc[] =
    "Compute the global size and stride of a DTensor from the given local tensor.\n"
    "The local size is multiplied by `world_size` per Sharding dim.\n"
    "The local stride is multiplied by `world_size` per Sharding dim, as long as the\n"
    "dimension is outside sharding dim.\n"
    "\n"
    "For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).\n"
    "If the DTensor placements are [Shard(2)] and world_size is 2;\n"
    "then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).\n"
    "\n"
    "Args:\n"
    "    tensor (:class:`torch.Tensor`):\n"
    "        Local tensor which DTensor will be constructed from.\n"
    "    mesh (:class:`DeviceMesh`):\n"
    "        Object which describes the mesh topology\n"
    "        of devices for the DTensor.\n"
    "    placements (Sequence[:class:`Placement`]]):\n"
    "        The attribute of the DTensor that describes its layout\n"
    "        on the mesh topology.\n"
    "\n"
    "Return:\n"
    "    tensor_shape: A List of int which specifies the size of DTensor which build\n"
    "        on top of the local tensor.\n"
    "    tensor_stride: A List of int which specifies the stride of DTensor.\n";

static PyObject* DTensor_compute_global_tensor_info(
    PyObject* self,
    PyObject* const* args,
    Py_ssize_t nargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK_VALUE(
      nargs == 3,
      "compute_global_tensor_info expects 3 arguments, got ",
      nargs);
  TORCH_CHECK_TYPE(
      THPVariable_Check(args[0]),
      "compute_global_tensor_info 1st argument must be Tensor!");
  const auto& tensor = THPVariable_Unpack(args[0]);
  const py::handle mesh = args[1];
  TORCH_CHECK_TYPE(
      PySequence_Check(args[2]),
      "compute_global_tensor_info 3rd argument must be sequence!");
  const py::sequence placements = py::reinterpret_borrow<py::sequence>(args[2]);
  return DTensor_compute_global_tensor_info_impl(tensor, mesh, placements);
  END_HANDLE_TH_ERRORS
}

using getter = PyObject* (*)(PyObject*, void*);
using setter = int (*)(PyObject*, PyObject*, void*);

static PyObject* THPVariable_get_python_dispatch(
    THPVariable* self,
    void* unused) {
  HANDLE_TH_ERRORS
  const auto& var = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(
      var.unsafeGetTensorImpl()->is_python_dispatch());
  END_HANDLE_TH_ERRORS
}

// CRTP base class to implement the python bindings for a Tensor property in
// PyTorch A class that implements a property is expected to have:
// - static constexpr const char* name;
//   - This variable should hold the Python name of the property
// - static Tensor fn(const Tensor&);
//   - This function calls the relevant ATen on the tensor
template <typename T>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
struct GetterBase {
  static PyObject* getter(THPVariable* self, void* /*unused*/) {
    HANDLE_TH_ERRORS
    if (check_has_torch_function((PyObject*)self)) {
      return handle_torch_function_getter(self, T::name);
    }
    return THPVariable_Wrap(T::fn(THPVariable_Unpack(self)));
    END_HANDLE_TH_ERRORS
  }
};

struct PropertyT : GetterBase<PropertyT> {
  static constexpr const char* name = "T";
  static Tensor fn(const Tensor& t) {
    return t.numpy_T();
  }
};

struct PropertyH : GetterBase<PropertyH> {
  static constexpr const char* name = "H";
  static Tensor fn(const Tensor& t) {
    return t.matrix_H();
  }
};

struct PropertymT : GetterBase<PropertymT> {
  static constexpr const char* name = "mT";
  static Tensor fn(const Tensor& t) {
    return t.mT();
  }
};

struct PropertymH : GetterBase<PropertymH> {
  static constexpr const char* name = "mH";
  static Tensor fn(const Tensor& t) {
    return t.mH();
  }
};

struct PropertyData : GetterBase<PropertyData> {
  static constexpr const char* name = "data";
  static Tensor fn(const Tensor& t) {
    return t.variable_data();
  }
};

struct PropertyGrad : GetterBase<PropertyGrad> {
  static constexpr const char* name = "grad";
  static Tensor fn(const Tensor& t) {
    return t.grad();
  }
};

struct PropertyReal : GetterBase<PropertyReal> {
  static constexpr const char* name = "real";
  static Tensor fn(const Tensor& t) {
    return at::real(t);
  }
};

struct PropertyImag : GetterBase<PropertyImag> {
  static constexpr const char* name = "imag";
  static Tensor fn(const Tensor& t) {
    return at::imag(t);
  }
};

static PyObject* THPVariable_get_cdata(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_cdata");
  }
  const auto& var = THPVariable_Unpack(self);
  return PyLong_FromVoidPtr(var.unsafeGetTensorImpl());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_version(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_version");
  }
  const auto& var = THPVariable_Unpack(self);
  return THPUtils_packInt64(var._version());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_grad_fn(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "grad_fn");
  }
  const auto& var = THPVariable_Unpack(self);
  if (!var.grad_fn()) {
    Py_RETURN_NONE;
  }
  return functionToPyObject(var.grad_fn());
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_grad_fn(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "_grad_fn", obj);
  }
  TORCH_CHECK(obj, "Deletion of _grad_fn not allowed. Detach tensor instead!");
  TORCH_CHECK(obj == Py_None, "_grad_fn can be only set to None");
  THPVariable_Unpack(self).detach_();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_is_leaf(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_leaf");
  }
  return PyBool_FromLong(!THPVariable_Unpack(self).grad_fn());
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_data(
    THPVariable* self,
    PyObject* data,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "data", data);
  }
  TORCH_CHECK(
      data, "Deleting tensor data is not allowed. Delete tensor instead!");
  TORCH_CHECK_TYPE(
      THPVariable_Check(data),
      "Variable data has to be a tensor, but got ",
      Py_TYPE(data)->tp_name);

  THPVariable_Unpack(self).set_data(THPVariable_Unpack(data));
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static int THPVariable_set_grad(
    THPVariable* self,
    PyObject* py_grad,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "grad", py_grad);
  }
  const auto& var = THPVariable_Unpack(self);
  if (!py_grad || py_grad == Py_None) {
    var.mutable_grad().reset();
    return 0;
  }

  TORCH_CHECK_TYPE(
      THPVariable_Check(py_grad),
      "assigned grad expected to be a Tensor or None but got grad of type ",
      THPUtils_typename(py_grad));
  TORCH_CHECK(
      self != (THPVariable*)py_grad, "can't assign Variable as its own grad");

  const auto& grad = THPVariable_Unpack(py_grad);
  if (var.grad_dtype().has_value()) {
    TORCH_CHECK(
        grad.dtype() == var.grad_dtype().value(),
        "attempting to assign a gradient with dtype '",
        grad.dtype(),
        "' to a tensor with grad_dtype '",
        var.grad_dtype().value(),
        "'. The gradient must match the tensor's grad_dtype (defaults to the tensor's "
        "dtype). You can set the tensor's grad_dtype attribute with a specific dtype, or "
        "None to allow any dtype. Set grad_dtype with caution. Diverging the dtypes of "
        "a tensor and its gradient may break downstream systems that assume they match.");
  }
  TORCH_CHECK(
      var.device().type() == grad.device().type(),
      "attempting to assign a gradient with device type '",
      grad.device().type(),
      "' to a tensor with device type '",
      var.device().type(),
      "'. Please ensure that the gradient and the tensor are on the same device");
  if (grad.layout() != kSparse) {
    auto expected_options = var.options().dtype(
        var.grad_dtype().has_value() ? var.grad_dtype().value()
                                     : grad.scalar_type());
    TORCH_CHECK(
        grad.options().type_equal(expected_options),
        "attempting to assign a gradient to a tensor that has data of a different type");
  }
  TORCH_CHECK(
      grad.get_device() == var.get_device(),
      "attempting to assign a gradient located on device with index '",
      grad.get_device(),
      "' to a tensor located on device with index '",
      var.get_device(),
      "'. Please ensure that the gradient and the tensor are on the same device");
  TORCH_CHECK(
      grad.sym_sizes().equals(var.sym_sizes()),
      "attempting to assign a gradient of size '",
      grad.sym_sizes(),
      "' to a tensor of size '",
      var.sym_sizes(),
      "'. Please ensure that the gradient and the tensor are the same size");

  var.mutable_grad() = grad;
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_get_volatile(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "volatile");
  }
  const char* msg = "volatile was removed (Variable.volatile is always False)";
  auto r = PyErr_WarnEx(PyExc_UserWarning, msg, 1);
  if (r != 0)
    throw python_error();
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_volatile(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "volatile", obj);
  }
  auto r = PyErr_WarnEx(PyExc_UserWarning, VOLATILE_WARNING, 1);
  if (r != 0)
    throw python_error();
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_get_output_nr(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "output_nr");
  }
  const auto output_nr = THPVariable_Unpack(self).output_nr();
  return THPUtils_packInt64(output_nr);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_requires_grad(
    THPVariable* self,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "requires_grad");
  }
  if (THPVariable_Unpack(self).requires_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_retains_grad(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "retains_grad");
  }
  if (THPVariable_Unpack(self).retains_grad()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_ndim(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "ndim");
  }
  return THPUtils_packInt64(THPVariable_Unpack(self).dim());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_names(PyObject* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function_getter((THPVariable*)self, "names");
  }
  // The long-term plan is to return a list of (python) torch.Dimname.
  // However, for now, return a list of string.
  const auto& tensor = THPVariable_Unpack(self);
  auto size = tensor.dim();
  THPObjectPtr tuple(PyTuple_New(size));
  if (!tuple)
    throw python_error();

  const auto dimnames = tensor.names();
  for (const auto i : c10::irange(size)) {
    PyObject* str = nullptr;
    if (dimnames[i].type() == at::NameType::WILDCARD) {
      // PyTuple_SET_ITEM steals a reference to the object. When the tuple is
      // deallocated, it'll decrement the refcount on Py_None, which is bad.
      // To avoid this, we "create" a new reference to Py_None by increasing
      // the refcount.
      // Sources:
      // - https://docs.python.org/3/c-api/tuple.html#c.PyTuple_SetItem
      // -
      // https://stackoverflow.com/questions/16400600/how-to-return-a-tuple-containing-a-none-value-from-the-c-api
      Py_INCREF(Py_None);
      str = Py_None;
    } else {
      str = THPUtils_packString(dimnames[i].symbol().toUnqualString());
      if (!str)
        throw python_error();
    }
    PyTuple_SET_ITEM(tuple.get(), i, str);
  }
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_names(
    PyObject* self,
    PyObject* names,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function(self)) {
    return handle_torch_function_setter((THPVariable*)self, "names", names);
  }
  const auto& var = THPVariable_Unpack(self);
  if (names == Py_None) {
    at::internal_set_names_inplace(var, std::nullopt);
  } else {
    TORCH_CHECK(
        THPUtils_checkDimnameList(names),
        "names must either be None or a tuple of dim names");
    at::internal_set_names_inplace(var, torch::parseDimnameList(names));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static int THPVariable_set_requires_grad(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "requires_grad", obj);
  }
  TORCH_CHECK(obj && PyBool_Check(obj), "requires_grad must be a bool");
  const auto& var = THPVariable_Unpack(self);
  auto requires_grad = (obj == Py_True);
  if (!var.is_leaf()) {
    THPUtils_setError(
        autograd::utils::requires_grad_leaf_error(obj == Py_True).c_str());
    return -1;
  }
  if (requires_grad &&
      !isDifferentiableType(at::typeMetaToScalarType((var.dtype())))) {
    THPUtils_setError(
        "only Tensors of floating point and complex dtype can require gradients");
    return -1;
  }
  var.set_requires_grad(requires_grad);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_get_name(THPVariable* self, void* unused) {
  if (check_has_torch_function((PyObject*)self)) {
    HANDLE_TH_ERRORS
    return handle_torch_function_getter(self, "name");
    END_HANDLE_TH_ERRORS
  }
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.name().empty())
    Py_RETURN_NONE;
  return THPUtils_packString(tensor.name().c_str());
}

static PyObject* THPVariable_get_backwards_hooks(
    THPVariable* self,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_backward_hooks");
  }
  if (self->backward_hooks) {
    Py_INCREF(self->backward_hooks);
    return self->backward_hooks;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_backwards_hooks(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "_backward_hooks", obj);
  }
  TORCH_CHECK(obj, "Deletion of _backwards_hooks not allowed!");
  if (obj == Py_None) {
    obj = nullptr;
  }
  Py_XINCREF(obj);
  Py_XDECREF(self->backward_hooks);
  self->backward_hooks = obj;
  const auto& tensor = THPVariable_Unpack(self);
  torch::autograd::impl::clear_hooks(tensor);
  if (obj) {
    torch::autograd::impl::add_hook(
        tensor, std::make_unique<PyFunctionTensorPreHook>(obj, 0));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_get_post_accumulate_grad_hooks(
    THPVariable* self,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_post_accumulate_grad_hooks");
  }
  if (self->post_accumulate_grad_hooks) {
    Py_INCREF(self->post_accumulate_grad_hooks);
    return self->post_accumulate_grad_hooks;
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_post_accumulate_grad_hooks(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(
        self, "_post_accumulate_grad_hooks", obj);
  }
  TORCH_CHECK(obj, "Deletion of _post_accumulate_grad_hooks not allowed!");
  if (obj == Py_None) {
    obj = nullptr;
  }
  Py_XINCREF(obj);
  Py_CLEAR(self->post_accumulate_grad_hooks);
  self->post_accumulate_grad_hooks = obj;
  const auto& tensor = THPVariable_Unpack(self);
  if (obj) {
    torch::autograd::impl::set_post_acc_grad_hooks(
        tensor, std::make_unique<PyFunctionTensorPostAccGradHooks>(obj));
  }
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_get_base(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "_base");
  }
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.is_view()) {
    return THPVariable_Wrap(tensor._base());
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_shape(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "shape");
  }
  return THPSize_NewFromSymSizes(THPVariable_Unpack(self));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_cpu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_cpu");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_cpu());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_cuda(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_cuda");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_cuda());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_mtia(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_mtia");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_mtia());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_xla(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_xla");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_xla());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_ipu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_ipu");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_ipu());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_xpu(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_xpu");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_xpu());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_sparse(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_sparse");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_sparse());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_sparse_csr(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_sparse_csr");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_sparse_csr());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_mkldnn(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_mkldnn");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_mkldnn());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_mps(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_mps");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_mps());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_maia(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_maia");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_maia());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_vulkan(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_vulkan");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_vulkan());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_quantized(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_quantized");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_quantized());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_meta(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_meta");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_meta());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_complex(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_complex");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_complex());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_is_nested(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "is_nested");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.is_nested());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_has_symbolic_sizes_strides(
    THPVariable* self,
    void* unused) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(
      self_.unsafeGetTensorImpl()->has_symbolic_sizes_strides());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_dtype(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "dtype");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.scalar_type());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_layout(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "layout");
  }
  auto& self_ = THPVariable_Unpack(self);
  return torch::autograd::utils::wrap(self_.layout());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_device(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "device");
  }
  return THPDevice_New(THPVariable_Unpack(self).device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_nbytes(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "nbytes");
  }
  return PyLong_FromSize_t(THPVariable_Unpack(self).nbytes());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_get_grad_dtype(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "grad_dtype");
  }
  const auto& var = THPVariable_Unpack(self);
  TORCH_CHECK(
      !var.grad_fn(), "grad_dtype can only be accessed on leaf tensors.");
  if (!var.grad_dtype().has_value()) {
    Py_RETURN_NONE;
  } else {
    return torch::autograd::utils::wrap(var.grad_dtype().value());
  }
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_grad_dtype(
    THPVariable* self,
    PyObject* obj,
    void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_setter(self, "grad_dtype", obj);
  }
  const auto& var = THPVariable_Unpack(self);
  TORCH_CHECK(
      THPDtype_Check(obj) || obj == Py_None,
      "grad_dtype must be a torch.dtype or None, but got ",
      Py_TYPE(obj)->tp_name);
  if (var.grad().defined() && obj != Py_None) {
    auto new_dtype = reinterpret_cast<THPDtype*>(obj);
    TORCH_CHECK(
        var.grad().dtype() == new_dtype->scalar_type,
        "Cannot set grad_dtype to '",
        new_dtype->scalar_type,
        "' because there is already a gradient with dtype '",
        var.grad().dtype(),
        "'. Please clear the gradient (.grad = None) before changing grad_dtype, "
        "or ensure the new grad_dtype matches the existing gradient's dtype.");
  }
  std::optional<at::ScalarType> new_dtype;
  if (obj != Py_None) {
    auto* dtype = reinterpret_cast<THPDtype*>(obj);
    new_dtype = dtype->scalar_type;
  }
  var.set_grad_dtype(new_dtype);
  return 0;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable_get_itemsize(THPVariable* self, void* unused) {
  HANDLE_TH_ERRORS
  if (check_has_torch_function((PyObject*)self)) {
    return handle_torch_function_getter(self, "itemsize");
  }
  return PyLong_FromSize_t(THPVariable_Unpack(self).itemsize());
  END_HANDLE_TH_ERRORS
}

static int THPVariable_set_real(PyObject* self, PyObject* real, void* unused) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  auto self_real = at::real(self_);
  auto real_ = valueToTensor(self_real.options(), real, self_real.device());
  {
    pybind11::gil_scoped_release no_gil;
    self_real.copy_(real_);
    return 0;
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}

static int THPVariable_set_imag(PyObject* self, PyObject* imag, void* unused) {
  HANDLE_TH_ERRORS
  auto& self_ = THPVariable_Unpack(self);
  auto self_imag = at::imag(self_);
  auto imag_ = valueToTensor(self_imag.options(), imag, self_imag.device());
  {
    pybind11::gil_scoped_release no_gil;
    self_imag.copy_(imag_);
    return 0;
  }
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject* THPVariable__use_count(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  const auto& t = THPVariable_Unpack(self);
  return THPUtils_packUInt64(t.use_count());
  END_HANDLE_TH_ERRORS
}

// properties are registered here because we are currently only able to bind
// them manually. TODO: make declarable in native_functions
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPVariable_properties[] = {
    {"_python_dispatch",
     (getter)THPVariable_get_python_dispatch,
     nullptr,
     nullptr,
     nullptr},
    {"T", (getter)PropertyT::getter, nullptr, nullptr, nullptr},
    {"H", (getter)PropertyH::getter, nullptr, nullptr, nullptr},
    {"mT", (getter)PropertymT::getter, nullptr, nullptr, nullptr},
    {"mH", (getter)PropertymH::getter, nullptr, nullptr, nullptr},
    {"_cdata", (getter)THPVariable_get_cdata, nullptr, nullptr, nullptr},
    {"_version", (getter)THPVariable_get_version, nullptr, nullptr, nullptr},
    {"grad_fn", (getter)THPVariable_get_grad_fn, nullptr, nullptr, nullptr},
    {"_grad_fn",
     (getter)THPVariable_get_grad_fn,
     (setter)THPVariable_set_grad_fn,
     nullptr,
     nullptr},
    {"is_leaf", (getter)THPVariable_is_leaf, nullptr, nullptr, nullptr},
    {"retains_grad",
     (getter)THPVariable_retains_grad,
     nullptr,
     nullptr,
     nullptr},
    {"data",
     (getter)PropertyData::getter,
     (setter)THPVariable_set_data,
     nullptr,
     nullptr},
    {"_grad",
     (getter)PropertyGrad::getter,
     (setter)THPVariable_set_grad,
     nullptr,
     nullptr}, // Allows the python class to override .grad
    {"grad",
     (getter)PropertyGrad::getter,
     (setter)THPVariable_set_grad,
     nullptr,
     nullptr},
    {"_base", (getter)THPVariable_get_base, nullptr, nullptr, nullptr},
    {"volatile",
     (getter)THPVariable_get_volatile,
     (setter)THPVariable_set_volatile,
     nullptr,
     nullptr},
    {"output_nr", (getter)THPVariable_get_output_nr, nullptr, nullptr, nullptr},
    {"requires_grad",
     (getter)THPVariable_get_requires_grad,
     (setter)THPVariable_set_requires_grad,
     nullptr,
     nullptr},
    {"_backward_hooks",
     (getter)THPVariable_get_backwards_hooks,
     (setter)THPVariable_set_backwards_hooks,
     nullptr,
     nullptr},
    {"_post_accumulate_grad_hooks",
     (getter)THPVariable_get_post_accumulate_grad_hooks,
     (setter)THPVariable_set_post_accumulate_grad_hooks,
     nullptr,
     nullptr},
    {"name", (getter)THPVariable_get_name, nullptr, nullptr, nullptr},
    {"shape", (getter)THPVariable_get_shape, nullptr, nullptr, nullptr},
    {"is_cuda", (getter)THPVariable_is_cuda, nullptr, nullptr, nullptr},
    {"is_mtia", (getter)THPVariable_is_mtia, nullptr, nullptr, nullptr},
    {"is_cpu", (getter)THPVariable_is_cpu, nullptr, nullptr, nullptr},
    {"is_xla", (getter)THPVariable_is_xla, nullptr, nullptr, nullptr},
    {"is_xpu", (getter)THPVariable_is_xpu, nullptr, nullptr, nullptr},
    {"is_ipu", (getter)THPVariable_is_ipu, nullptr, nullptr, nullptr},
    {"is_sparse", (getter)THPVariable_is_sparse, nullptr, nullptr, nullptr},
    {"is_sparse_csr",
     (getter)THPVariable_is_sparse_csr,
     nullptr,
     nullptr,
     nullptr},
    {"is_mkldnn", (getter)THPVariable_is_mkldnn, nullptr, nullptr, nullptr},
    {"is_mps", (getter)THPVariable_is_mps, nullptr, nullptr, nullptr},
    {"is_maia", (getter)THPVariable_is_maia, nullptr, nullptr, nullptr},
    {"is_vulkan", (getter)THPVariable_is_vulkan, nullptr, nullptr, nullptr},
    {"is_complex", (getter)THPVariable_is_complex, nullptr, nullptr, nullptr},
    {"is_quantized",
     (getter)THPVariable_is_quantized,
     nullptr,
     nullptr,
     nullptr},
    {"is_meta", (getter)THPVariable_is_meta, nullptr, nullptr, nullptr},
    {"is_nested", (getter)THPVariable_is_nested, nullptr, nullptr, nullptr},
    {"_has_symbolic_sizes_strides",
     (getter)THPVariable_has_symbolic_sizes_strides,
     nullptr,
     nullptr,
     nullptr},
    {"dtype", (getter)THPVariable_dtype, nullptr, nullptr, nullptr},
    {"layout", (getter)THPVariable_layout, nullptr, nullptr, nullptr},
    {"device", (getter)THPVariable_device, nullptr, nullptr, nullptr},
    {"ndim", (getter)THPVariable_get_ndim, nullptr, nullptr, nullptr},
    {"nbytes", (getter)THPVariable_get_nbytes, nullptr, nullptr, nullptr},
    {"itemsize", (getter)THPVariable_get_itemsize, nullptr, nullptr, nullptr},
    {"names",
     (getter)THPVariable_get_names,
     (setter)THPVariable_set_names,
     nullptr,
     nullptr},
    {"real",
     (getter)PropertyReal::getter,
     (setter)THPVariable_set_real,
     nullptr,
     nullptr},
    {"imag",
     (getter)PropertyImag::getter,
     (setter)THPVariable_set_imag,
     nullptr,
     nullptr},
    {"grad_dtype",
     (getter)THPVariable_get_grad_dtype,
     (setter)THPVariable_set_grad_dtype,
     nullptr,
     nullptr},
    {nullptr}};

static PyMappingMethods THPVariable_as_mapping = {
    THPVariable_length,
    THPVariable_getitem,
    THPVariable_setitem,
};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef extra_methods[] = {
    {"as_subclass",
     castPyCFunctionWithKeywords(THPVariable_as_subclass),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_make_subclass",
     castPyCFunctionWithKeywords(THPVariable_make_subclass),
     METH_STATIC | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_make_wrapper_subclass",
     castPyCFunctionWithKeywords(THPVariable_make_wrapper_subclass),
     METH_STATIC | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_dtensor__new__",
     castPyCFunctionWithKeywords(THPVariable_dtensor_new),
     METH_STATIC | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_fix_weakref", THPVariable_fix_weakref, METH_NOARGS, nullptr},
    {"_view_func",
     castPyCFunctionWithKeywords(THPVariable_view_func),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_view_func_unsafe",
     castPyCFunctionWithKeywords(THPVariable_view_func_unsafe),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_rev_view_func_unsafe",
     THPVariable_rev_view_func_unsafe,
     METH_O,
     nullptr},
    {"_use_count", THPVariable__use_count, METH_NOARGS, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef extra_functions[] = {
    {"_DTensor_OpSchema_post_init",
     DTensor_OpSchema_post_init,
     METH_O,
     nullptr},
    {"_DTensor_OpSchema_recompute_comparison_key",
     DTensor_OpSchema_recompute_comparison_key,
     METH_O,
     nullptr},
    {"_DTensor_compute_global_tensor_info",
     castPyCFunctionFast(DTensor_compute_global_tensor_info),
     METH_FASTCALL,
     compute_global_tensor_info_doc},
    {nullptr}};

struct THPVariableMeta {
  PyHeapTypeObject base;
};

static int THPVariableMetaType_init(
    PyObject* cls,
    PyObject* args,
    PyObject* kwargs);

static PyTypeObject THPVariableMetaType = {
    PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
    "torch._C._TensorMeta", /* tp_name */
    sizeof(THPVariableMeta), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    DEFERRED_ADDRESS(&PyType_Type), /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    THPVariableMetaType_init, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

static PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(&THPVariableMetaType, 0)
    "torch._C.TensorBase", /* tp_name */
    sizeof(THPVariable), /* tp_basicsize */
    0, /* tp_itemsize */
    // This is unspecified, because it is illegal to create a THPVariableType
    // directly.  Subclasses will have their tp_dealloc set appropriately
    // by the metaclass
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &THPVariable_as_mapping, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    // Also set by metaclass
    (traverseproc)THPFake_traverse, /* tp_traverse */
    (inquiry)THPFake_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    THPVariable_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    // Although new is provided here, it is illegal to call this with cls ==
    // THPVariableMeta.  Instead, subclass it first and then construct it
    THPVariable_pynew, /* tp_new */
};

PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      type != &THPVariableType,
      "Cannot directly construct TensorBase; subclass it and then construct that");
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto tensor = torch::utils::base_tensor_ctor(args, kwargs);
  // WARNING: tensor is NOT guaranteed to be a fresh tensor; e.g., if it was
  // given a raw pointer that will refcount bump
  // NB: base_tensor_ctor can call into dispatched ATen functions (e.g.,
  // alias(), lift_fresh()) which can return Tensor subclasses.  We allow
  // these to be passed on directly.
  return THPVariable_NewWithVar(
      type,
      tensor,
      /*allow_preexisting_pyobj=*/true);
  END_HANDLE_TH_ERRORS
}

static int THPVariable_subclass_clear(THPVariable* self) {
  // Is it OK for an object to still be live after running
  // tp_clear? Yes. When Python is breaking reference cycles, it can't assume
  // that an object will dealloc after it's cleared.  The source code explicitly
  // handles this case:
  // https://github.com/python/cpython/blob/4e661cd69164318c1f871faa476c68a04092ddc4/Modules/gcmodule.c#L1010-L1025

  // Note that we don't need to actually resurrect here. There are 2 cases:
  // 1. The PyObject is not part of a reference cycle. In this case, we don't
  // need to do anything. The GC will move on to try and break the reference
  // cycle on another object, which will eventually trigger tp_dealloc (and thus
  // resurrection).

  // 2. The PyObject is part of a reference cycle. This case should not actually
  // be possible, due to the logic in our tp_traverse
  // (THPVariable_subclass_traverse).

  // In fact, resurrecting here breaks the invariant that "C++ owns Python only
  // when PyObject's refcount would otherwise be 0". Most immediately, as we're
  // merely breaking reference cycles here, there can be other references to the
  // PyObject. *However*, if other objects in the refcycle resurrect, then we
  // will be in a state where the PyObject has multiple Python references, yet
  // C++ owns the PyObject.

  // See https://github.com/pytorch/pytorch/pull/75933 for more discussion.
  if (isResurrectable(self)) {
    return 0;
  }

  // First clear Tensor specific things

  Py_CLEAR(self->backward_hooks);
  Py_CLEAR(self->post_accumulate_grad_hooks);
  const auto& tensor = THPVariable_Unpack(self);
  if (tensor.defined()) {
    // Two situations to consider:
    //    PyObject -owns-> Tensor
    //        unsafeIsBorrowed() is FALSE.  We're obligated to look through
    //        Tensor to break references.  Clearing cdata must induce the
    //        destruction of the C++ Tensor.  If there were other references
    //        to C++ tensor, the Python object would have been resurrected
    //        by flipping the ownership.
    //    Tensor -owns-> PyObject
    //        unsafeIsBorrowed() is TRUE.  We're deallocating the PyObject
    //        because Tensor asked us to (it's already destructing).

    if (!self->cdata.unsafeIsBorrowed() &&
        tensor.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
            /*ignore_hermetic_tls=*/false) == (PyObject*)self) {
      // TODO: empirically, on OS X this assert appears to be untrue
      // In test_py_tensors_multi_async_call - ProcessGroupRpcTestWithSpawn
      // distributed/rpc/test_process_group_agent.py
      //
      //  libc++abi.dylib: terminating with uncaught exception of type
      //  c10::Error:
      //  !tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj()INTERNAL
      //  ASSERT FAILED at "../torch/csrc/autograd/python_variable.cpp":171,
      //  please report a bug to PyTorch. Exception raised from
      //  THPVariable_subclass_clear at
      //  ../torch/csrc/autograd/python_variable.cpp:171 (most recent call
      //  first): frame #0: c10::Error::Error(c10::SourceLocation,
      //  std::__1::basic_string<char, std::__1::char_traits<char>,
      //  std::__1::allocator<char> >) + 98 (0x1158a0442 in libc10.dylib) frame
      //  #1: c10::detail::torchCheckFail(char const*, char const*, unsigned
      //  int, char const*) + 205 (0x11589ed3d in libc10.dylib) frame #2:
      //  c10::detail::torchInternalAssertFail(char const*, char const*,
      //  unsigned int, char const*, c10::detail::CompileTimeEmptyString) + 9
      //  (0x1141e3f89 in libtorch_python.dylib) frame #3:
      //  THPVariable_subclass_clear(THPVariable*) + 412 (0x1148a547c in
      //  libtorch_python.dylib) frame #4:
      //  THPVariable_subclass_dealloc(_object*) + 453 (0x1148a5035 in
      //  libtorch_python.dylib) frame #5: (anonymous
      //  namespace)::concrete_decref_fn(c10::impl::PyInterpreter const*,
      //  _object*) + 53 (0x1148a5ea5 in libtorch_python.dylib) frame #6:
      //  c10::TensorImpl::release_resources() + 182 (0x11588c4a6 in
      //  libc10.dylib) frame #7:
      //  c10::MaybeOwned<at::Tensor>::operator=(c10::MaybeOwned<at::Tensor>&&)
      //  + 91 (0x11488c11b in libtorch_python.dylib) frame #8:
      //  THPVariable_subclass_dealloc(_object*) + 607 (0x1148a50cf in
      //  libtorch_python.dylib) <omitting python frames> frame #47: start + 1
      //  (0x7fff6ffc7cc9 in libdyld.dylib) frame #48: 0x0 + 4 (0x4 in ???)
      // TORCH_INTERNAL_ASSERT(!tensor.unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj());
      if (auto grad_acc =
              torch::autograd::impl::try_get_grad_accumulator(tensor)) {
        grad_acc->pre_hooks().clear();
        grad_acc->tensor_pre_hooks().clear();
        grad_acc->retains_grad_hooks().clear();
      }
    }
  }
  TORCH_INTERNAL_ASSERT(!isResurrectable(self));
  {
    // MapAllocator can take significant time to release large tensors;
    // release the GIL here to avoid impacting main thread perf.
    pybind11::gil_scoped_release no_gil;
    self->cdata = MaybeOwned<Variable>();
  }
  // Since we override the basic subtype_clear from CPython, we need a crappy
  // version here just like for traverse and dealloc

  // Clear all slots until we get to the base Tensor class
  PyTypeObject* type = Py_TYPE((PyObject*)self);
  PyTypeObject* base = type;
  while (base != &THPVariableType) {
    if (Py_SIZE(base))
      clear_slots(base, (PyObject*)self);
    base = base->tp_base;
    TORCH_INTERNAL_ASSERT(base);
  }

  // Assume we never have managed dict for Tensors as we don't set the flag on
  // the base class
  if (C10_LIKELY(type->tp_dictoffset)) {
    PyObject** dictptr = _PyObject_GetDictPtr((PyObject*)self);
    if (dictptr && *dictptr)
      Py_CLEAR(*dictptr);
  }

  return 0;
}

// NB: this is not the tp_dealloc on THPVariable; instead, its the dealloc
// on subclasses.  It's never valid to construct a THPVariable so it's not
// necessary to implement the dealloc for that case
static void THPVariable_subclass_dealloc(PyObject* self) {
  if (THPVariable_tryResurrect((THPVariable*)self))
    return;

  // This is like a crappy version of subtype_dealloc.
  // Unfortunately, we cannot directly delegate to
  // subtype_dealloc as it will start walking the parent
  // chain *starting with* the type of self, which will cause
  // us to go back to our custom dealloc.
  //
  // We have to replicate the subtype_dealloc logic to ensure
  // that finalizers are handled correctly
  PyTypeObject* type = Py_TYPE(self);
  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  TORCH_INTERNAL_ASSERT(PyType_IS_GC(type), "GC types not implemented");

  PyObject_GC_UnTrack(self);
  // TODO: consider using trash can

  bool has_finalizer = type->tp_finalize || type->tp_del;

  if (type->tp_finalize) {
    PyObject_GC_Track(self);
    if (PyObject_CallFinalizerFromDealloc(self) < 0) {
      /* Resurrected */
      return;
    }
    PyObject_GC_UnTrack(self);
  }

  // base test is unnecessary as THPVariable does not set this
  if (type->tp_weaklistoffset) {
    PyObject_ClearWeakRefs(self);
  }

  if (type->tp_del) {
    PyObject_GC_Track(self);
    type->tp_del(self);
    if (Py_REFCNT(self) > 0) {
      /* Resurrected */
      return;
    }
    PyObject_GC_UnTrack(self);
  }

  if (has_finalizer) {
    /* New weakrefs could be created during the finalizer call.
       If this occurs, clear them out without calling their
       finalizers since they might rely on part of the object
       being finalized that has already been destroyed. */
    if (type->tp_weaklistoffset) {
      /* Modeled after GET_WEAKREFS_LISTPTR() */
      PyWeakReference** list =
          (PyWeakReference**)PyObject_GET_WEAKREFS_LISTPTR(self);
      while (*list)
        _PyWeakref_ClearRef(*list);
    }
  }

  // Clear all slots until we get to base class THPVariableType
  {
    PyTypeObject* base = type;
    while (base != &THPVariableType) {
      if (Py_SIZE(base)) {
        clear_slots(base, self);
      }
      base = base->tp_base;
      TORCH_INTERNAL_ASSERT(base);
    }
  }

  // All Python defined classes have __dict__
  if (C10_LIKELY(type->tp_dictoffset)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr != nullptr) {
      PyObject* dict = *dictptr;
      if (dict != nullptr) {
        Py_DECREF(dict);
        *dictptr = nullptr;
      }
    }
  }

  // subtype_dealloc allows for this but we don't
  TORCH_INTERNAL_ASSERT(Py_TYPE(self) == type);

  // Finally clear out the base THPVariable
  THPVariable_subclass_clear((THPVariable*)self);
  ((THPVariable*)self)->cdata.~MaybeOwned<Variable>();
  Py_TYPE(self)->tp_free(self);

  // Python defined subclasses should always be on the heap
  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  Py_DECREF(type);
}

// Creates a new Python object for a Variable.
static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    const at::TensorBase& _var,
    bool allow_preexisting_pyobj,
    std::optional<bool> has_torch_dispatch_if_known) {
  // Make sure that the reinterpret into a THPVariable* will be valid
  TORCH_CHECK(
      type == &THPVariableType || PyType_IsSubtype(type, &THPVariableType),
      "Creating a Tensor subclass from a class ",
      "that does not inherit from Tensor is not possible. Make sure your class inherits from Tensor.");

  // This function overwrite the Tensor's pyobj field without extra checks
  // Make sure it is not set otherwise we would leak memory
  auto mb_obj = _var.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
      /*ignore_hermetic_tls=*/false);

  // Under some circumstances, we may attempt to create a new Python
  // object for a variable that already has a Python object.  The most common
  // situation this can occur is if you have a TorchDispatchMode active that
  // is returning a subclass from lift_fresh (which is invoked to
  // appropriately "wrap" a constant tensor into whatever ambient modes are
  // active.)
  //
  // In general, it is impossible to handle this case compositionally.
  // Suppose you have a user call ATensor([1, 2, 3]) when a mode is active
  // that is transforming all ops (including the internal lift_fresh call that
  // transforms [1, 2, 3] into a torch.tensor([1., 2., 3.])) to output
  // BTensor, where ATensor and BTensor are completely unrelated subclasses
  // and there is no way to compose them.  There is no way to satisfy the user
  // request here: in particular, you can't just try to re-invoke the ATensor
  // constructor on the returned BTensor, because (1) this could cause an
  // infinite loop--we are already in ATensor.__new__ and (2) there isn't any
  // guarantee that ATensor.__new__ supports a single element constructor
  // anyway.
  //
  // However, a more common case is a user just called torch.Tensor([1, 2, 3]),
  // and a fake tensor mode is active.  Really, all you want is to get back
  // a FakeTensor, in the same way torch.tensor([1, 2, 3]) or torch.arange(3)
  // would have returned a fake tensor (concretely, the way this happens
  // is we create a *real* tensor torch.tensor([1., 2., 3.]), and then it
  // turns into a FakeTensor when we call lift_fresh on this real tensor).
  // This case is compositional because FakeTensor is a subclass of Tensor, so
  // it's valid for us to return it in place of a Tensor.  So this is what we
  // do.

  if (mb_obj.has_value() && mb_obj.value()) {
    TORCH_CHECK(
        allow_preexisting_pyobj,
        "Creating a new Tensor subclass ",
        type->tp_name,
        " but the raw Tensor object is already associated to a python object ",
        "of type ",
        mb_obj.value()->ob_type->tp_name);
    // Even if we allow pre-existing PyObject, we don't allow completely
    // ignoring the requested type.  Check that we fulfilled a subtype
    // relation here.  In the common case the requested type is Tensor and
    // this always succeeds.
    PyObject* obj = *mb_obj;
    // Check if it's OK to just directly return the Python object without
    // allocating a new variable.  We just check that the existing Python
    // object is a subclass of the requested type.
    PyTypeObject* obj_type = Py_TYPE(obj);
    TORCH_CHECK(
        obj_type == type || PyType_IsSubtype(obj_type, type),
        "Creating a new Tensor subclass ",
        type->tp_name,
        " but the raw Tensor object is already associated to a python object ",
        "of type ",
        mb_obj.value()->ob_type->tp_name,
        " which is not a subclass of the "
        "requested type");
    // We may (in fact, we typically will) need to resurrect this
    return THPVariable_Wrap(_var);
  }

  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*)obj;
    // TODO: named constructor to avoid default initialization
    new (&v->cdata) MaybeOwned<Variable>();
    if (c10::impl::HermeticPyObjectTLS::get_state()) {
      // Do NOT initialize pyobj field on the tensor, you own the C++
      v->cdata = MaybeOwned<Variable>::owned(Variable(_var));
      TORCH_INTERNAL_ASSERT(
          !check_has_torch_dispatch(obj),
          "While HermeticPyObject was enabled, we attempted to create a tensor "
          "subclass with __torch_dispatch__.  This violates the invariant that "
          "operations in HermeticPyObject have equivalent C++ implementations. "
          "If your operator registered from Python operator registration isn't "
          "doing anything strange, there may be an internal PyTorch bug involving "
          "not appropriately disabling TorchDispatchMode before executing "
          "Python op registration.");
    } else {
      // Normal codepath
      v->cdata = MaybeOwned<Variable>::owned(Variable(_var));
      const auto& var = THPVariable_Unpack(v);
      var.unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(obj);
      if (has_torch_dispatch_if_known.has_value()
              ? *has_torch_dispatch_if_known
              : check_has_torch_dispatch(obj)) {
        var.unsafeGetTensorImpl()->set_python_dispatch(true);
      }
    }
  }
  return obj;
}

/// NOTE [ PyObject Traversal ]
///
/// PyObjects that are wrapping c++ objects can lead to non-trivial traverse
/// logic and it can be tricky to know what to traverse and when. This note
/// tries to clarify what is the danger here and a simple algorithm to choose
/// how to write the tp_traverse and tp_clear functions. If you're not already
/// familiar with how the CPython GC works, you should read this in-depth
/// description: https://devguide.python.org/garbage_collector/
///
/// The complexity for us comes from the fact that some c++ shared_ptr objects
/// own references to python objects and are also owned both by other python
/// objects and c++ objects. This means that to allow the GC to collect all
/// cycles, we need to properly implement the traverse/clear methods that take
/// into account these C++ ownership links.
///
/// The main danger here comes from the fact that, while all python-related code
/// is thread safe wrt the GC execution (thanks to the GIL), other threads might
/// be using our C++ objects arbitrarily which can lead to shared_ptr ref count
/// going up or down in between the different traverse/clear invocations. The
/// one constraint we add here that is not explicitly mentioned in the GC
/// description above is that for a given GC run (meaning while the GIL is
/// held), the traverse/clear pair should never report different ownership
/// relations: if traverse visited a given PyObject, then the clear within that
/// same GC run must still be the sole owner and clear that PyObject.
///
/// A more mechanical algorithm to know what to traverse/clear is as follows:
///   - Any field on this PyObject that contains a strong reference to another
///   PyObject
///     must be visited and cleared. An example of that is the "backward_hooks"
///     field of the THPVariable.
///   - Any field that contains a C++ object that is uniquely owned by this
///   PyObject (either
///     a unique_ptr or a shared_ptr with use_count==1) should have all the
///     PyObject it owns visited and cleared. An example would be here the
///     tensor hooks.
///   - If that uniquely owned C++ object also uniquely owns other C++ objects,
///   these should be
///     visited and cleared as well if they contain any PyObject.
///
/// Caveat: to avoid slow runtime, we limit the depth of this exploration of C++
/// objects in practice and we do not, for example, go through the whole
/// autograd graph, even if it is uniquely owned. This is a known place where
/// users can create noncollectable cycles as described in:
/// https://github.com/pytorch/pytorch/issues/7343
///

static int traverse_slots(
    PyTypeObject* type,
    PyObject* self,
    visitproc visit,
    void* arg) {
  auto n = Py_SIZE(type);
  auto mp = type->tp_members;
  for (Py_ssize_t i = 0; i < n; i++, mp++) {
    if (mp->type == T_OBJECT_EX) {
      char* addr = (char*)self + mp->offset;
      PyObject* obj = *(PyObject**)addr;
      if (obj != nullptr) {
        int err = visit(obj, arg);
        if (err)
          return err;
      }
    }
  }
  return 0;
}

static int THPVariable_subclass_traverse(
    PyObject* self,
    visitproc visit,
    void* arg) {
  // If the tensor is eligible to be resurrected, don't traverse it; instead
  // treat all of its references as a root (as they WOULD be a root since we
  // can treat the inbound C++ references as root owners).
  //
  // This works because unlike conventional GCs, Python's GC operates in two
  // phases: first it uses traverse to discover roots, and then it uses traverse
  // to do reachability.  Bypassing traverse during root discovery forces Python
  // to treat self as a root for everything it refers to.  For a full
  // explanation of the algorithm see
  // https://devguide.python.org/garbage_collector/
  //
  // NB: if we don't hold an owning reference to the underlying Tensor, it is
  // possible that the underlying Tensor has already gone dead.  In that case,
  // it's not safe to access it.  But it's also safe to traverse, because if
  // the underlying Tensor *is* live, then root discovery will determine that
  // self is live, and nothing will get GC'ed anyway (resurrection cannot happen
  // if the C++ objects owns the PyObject)
  THPVariable* var = reinterpret_cast<THPVariable*>(self);
  if (isResurrectable(var)) {
    return 0;
  }

  // Crappy version of subtype_traverse; same deal as
  // THPVariable_subclass_dealloc

  PyTypeObject* type = Py_TYPE(self);
  // Traverse slots until we get to base class THPVariableType
  {
    PyTypeObject* base = type;
    while (base != &THPVariableType) {
      if (Py_SIZE(base)) {
        int err = traverse_slots(base, self, visit, arg);
        if (err)
          return err;
      }
      base = base->tp_base;
      TORCH_INTERNAL_ASSERT(base);
    }
  }

  // All Python defined classes have __dict__
  if (C10_LIKELY(type->tp_dictoffset)) {
    PyObject** dictptr = _PyObject_GetDictPtr(self);
    if (dictptr && *dictptr)
      Py_VISIT(*dictptr);
  }

  TORCH_INTERNAL_ASSERT(type->tp_flags & Py_TPFLAGS_HEAPTYPE);
  Py_VISIT(type);

  // Finally traverse THPVariable special stuff
  Py_VISIT(var->backward_hooks);
  Py_VISIT(var->post_accumulate_grad_hooks);
  if (!var->cdata.unsafeIsBorrowed()) {
    const auto& tensor = THPVariable_Unpack(var);
    if (tensor.defined()) {
      // WARNING: The grad_fn traversal logic is very subtle, if you change
      // this, be very careful not to re-introduce this bug:
      // https://gist.github.com/zou3519/7ac92b84dd7d206dcc6eae55fee8372c

      // We ensure that we follow NOTE [ PyObject Traversal ] he by checking
      // that this python object is the sole owner of the underlying Tensor and
      // that this Tensor is the sole owner of its grad_fn. In this case, the
      // only way to get a new reference to the grad_fn is by using this python
      // object, which requires the GIL to be accessed. Note that this is only
      // valid as long as user don't share non-owning references across
      // different threads (which is crazy and should never be done).
      auto autograd_meta = torch::autograd::impl::get_autograd_meta(tensor);
      if (tensor.use_count() == 1) {
        if (autograd_meta) {
          // Do NOT call grad_fn() here as that might trigger a recompute
          const auto& grad_fn = autograd_meta->grad_fn_;
          if (grad_fn && grad_fn.use_count() == 1) {
            // All Node can have a pyobj (stored in "pyobj_")
            Py_VISIT(grad_fn->pyobj());
            // PyNode are special as they also have an "obj" field
            if (auto py_node_fn = dynamic_cast<PyNode*>(grad_fn.get())) {
              Py_VISIT(py_node_fn->obj);
            }
          }
        }
      }
      if (autograd_meta) {
        for (const auto& hook : torch::autograd::impl::hooks(tensor)) {
          if (auto pyhook =
                  dynamic_cast<PyFunctionTensorPreHook*>(hook.get())) {
            Py_VISIT(pyhook->dict);
          }
        }
      }
    }
  }

  return 0;
}

int THPVariableMetaType_init(PyObject* cls, PyObject* args, PyObject* kwargs) {
  if (PyType_Type.tp_init(cls, args, kwargs) < 0) {
    return -1;
  }
  // It is important for all three of these to be overridden correctly for the
  // resurrection checks to properly happen. In particular, an older version
  // was not overriding tp_clear here. This lead to the default subtype_clear
  // running on the Tensor object (as only TensorBase tp_clear was custom),
  // clearing the __dict__ field, before the TensorBase custom clear was called
  // and would properly detect the resurrect.
  // See https://github.com/pytorch/pytorch/issues/136358 for the exact behavior
  ((PyTypeObject*)cls)->tp_dealloc = (destructor)THPVariable_subclass_dealloc;
  ((PyTypeObject*)cls)->tp_traverse =
      (traverseproc)THPVariable_subclass_traverse;
  ((PyTypeObject*)cls)->tp_clear = (inquiry)THPVariable_subclass_clear;

  // Don't do anything for the base Tensor class
  if (!THPVariableClass) {
    return 0;
  }

  // Forbid subclassing _TensorBase directly
  py::tuple mro =
      py::reinterpret_borrow<py::tuple>(((PyTypeObject*)cls)->tp_mro);
  bool is_subclass_of_thpvariable = false;
  for (py::handle h : mro) {
    if (h.ptr() == THPVariableClass) {
      is_subclass_of_thpvariable = true;
      break;
    }
  }
  if (!is_subclass_of_thpvariable) {
    PyErr_SetString(PyExc_RuntimeError, "Cannot subclass _TensorBase directly");
    return -1;
  }

  // If the user provided a torch_dispatch implementation, disable
  // torch_function.
  py::object torch_dispatch_impl = py::reinterpret_steal<py::object>(
      PyObject_GetAttrString(cls, "__torch_dispatch__"));
  py::object torch_dispatch_default = py::reinterpret_steal<py::object>(
      PyObject_GetAttrString(THPVariableClass, "__torch_dispatch__"));
  if (torch_dispatch_impl.ptr() != torch_dispatch_default.ptr()) {
    py::object torch_function_impl = py::reinterpret_steal<py::object>(
        PyObject_GetAttrString(cls, "__torch_function__"));
    py::object torch_function_default_bound = py::reinterpret_steal<py::object>(
        PyObject_GetAttrString(THPVariableClass, "__torch_function__"));

    // Since our __torch_function__ is a classmethod, we need to "unbound" the
    // method to get the raw function
    py::object torch_function_default = py::reinterpret_steal<py::object>(
        PyObject_GetAttrString(torch_function_default_bound.ptr(), "__func__"));

    // User-defined __torch_function__ might not be a classmethod
    if (PyObject_HasAttrString(torch_function_impl.ptr(), "__func__")) {
      torch_function_impl = py::reinterpret_steal<py::object>(
          PyObject_GetAttrString(torch_function_impl.ptr(), "__func__"));
    }
    if (torch_function_impl.ptr() == torch_function_default.ptr()) {
      PyObject_SetAttrString(
          cls, "__torch_function__", torch::disabled_torch_function_impl());
    }
  }

  return 0;
}

namespace torch::autograd {

// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
extern PyMethodDef variable_methods[];

static void initTensorImplConversion(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_wrap_tensor_impl", [](void* ptr) {
    auto p = c10::intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>::
        unsafe_reclaim_from_nonowning(static_cast<c10::TensorImpl*>(ptr));
    TORCH_CHECK(p.defined(), "Can't wrap undefined tensor");
    auto tensor = at::Tensor::wrap_tensor_impl(std::move(p));
    return py::cast(std::move(tensor));
  });
  // set on the module level to avoid mixing pybind and plain CPython extensions
  m.def("_tensor_impl_raw_handle", [](torch::autograd::Variable* t) -> void* {
    // We return a raw non-owning pointer here, we rely on surrounding
    // code to keep the original tensor alive
    return t->getIntrusivePtr().get();
  });
}
} // namespace torch::autograd

bool THPVariable_initModule(PyObject* module) {
  THPVariableMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPVariableMetaType) < 0)
    return false;
  Py_INCREF(&THPVariableMetaType);
  PyModule_AddObject(module, "_TensorMeta", (PyObject*)&THPVariableMetaType);

  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "TensorBase", (PyObject*)&THPVariableType);
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  torch::utils::validate_numpy_for_dlpack_deleter_bug();

  if (!intern_dtensor_strings()) {
    return false;
  }
  PyModule_AddFunctions(module, extra_functions);
  return true;
}
