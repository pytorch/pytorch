#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/tensor_types.h>

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/detail/XPUHooksInterface.h>

#include <structmember.h>
#include <utility>

using namespace at;
using namespace torch;

PyObject* THPGeneratorClass = nullptr;

PyObject* THPGenerator_initDefaultGenerator(const at::Generator& cdata) {
  auto type = (PyTypeObject*)THPGeneratorClass;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPGenerator*>(self.get());
  self_->cdata = cdata;
  return self.release();
}

static void THPGenerator_dealloc(PyObject* _self) {
  auto self = reinterpret_cast<THPGenerator*>(_self);
  if (self->cdata.defined()) {
    self->cdata.set_pyobj(nullptr);
    self->cdata.~Generator();
  }
  Py_TYPE(_self)->tp_free(_self);
}

static PyObject* THPGenerator_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({"Generator(Device device=None)"});
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto device = r.deviceWithDefault(0, at::Device(at::kCPU));

  THPGeneratorPtr self((THPGenerator*)type->tp_alloc(type, 0));

  c10::DeviceType device_type = device.type();
  if (device_type == at::kCPU) {
    self->cdata = make_generator<CPUGeneratorImpl>();
  } else {
    self->cdata = globalContext()
                      .getAcceleratorHooksInterface(device_type)
                      .getNewGenerator(device.index());
  }

  return (PyObject*)self.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_getState(PyObject* _self, PyObject* noargs) {
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  auto& gen = ((THPGenerator*)_self)->cdata;

  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(gen.mutex());
  auto state_tensor = gen.get_state();

  return THPVariable_Wrap(state_tensor);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_setState(PyObject* _self, PyObject* _new_state) {
  using namespace torch::autograd;

  HANDLE_TH_ERRORS
  if (!THPVariable_Check(_new_state)) {
    TORCH_CHECK_TYPE(
        false,
        fmt::format(
            "expected a torch.ByteTensor, but got {}",
            Py_TYPE(_new_state)->tp_name));
  }
  auto self = (THPGenerator*)_self;
  auto& gen = self->cdata;
  const auto& new_state_tensor = THPVariable_Unpack(_new_state);

  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(gen.mutex());
  gen.set_state(new_state_tensor);

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static uint64_t unpack_uint64(PyObject* pyobj) {
  uint64_t unsigned_obj = 0;
  try {
    // First try to interpret as unsigned long
    unsigned_obj = THPUtils_unpackUInt64(pyobj);
  } catch (...) {
    if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
      // If an overflow happened, then the pyobj could be negative,
      // so try to interpret it as signed long
      PyErr_Clear();
      int64_t obj = THPUtils_unpackLong(pyobj);
      unsigned_obj = *(reinterpret_cast<uint64_t*>(&obj));
    } else {
      // If any other type of exception happened, rethrow it
      throw;
    }
  }
  return unsigned_obj;
}

static PyObject* THPGenerator_graphSafeGetState(
    PyObject* _self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto& gen = ((THPGenerator*)_self)->cdata;

  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(gen.mutex());

  return THPGenerator_Wrap(gen.graphsafe_get_state());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_graphSafeSetState(
    PyObject* _self,
    PyObject* _state) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  auto& gen = self->cdata;

  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(gen.mutex());
  gen.graphsafe_set_state(THPGenerator_Unwrap(_state));

  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_cloneState(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto& gen = ((THPGenerator*)_self)->cdata;

  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(gen.mutex());

  return THPGenerator_Wrap(gen.clone());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_manualSeed(PyObject* _self, PyObject* seed) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  auto generator = self->cdata;
  TORCH_CHECK(
      THPUtils_checkLong(seed),
      "manual_seed expected a long, "
      "but got ",
      THPUtils_typename(seed));
  uint64_t unsigned_seed = unpack_uint64(seed);
  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(generator.mutex());
  generator.set_current_seed(unsigned_seed);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_setOffset(PyObject* _self, PyObject* offset) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  auto generator = self->cdata;
  TORCH_CHECK(
      THPUtils_checkLong(offset),
      "manual_offset expected a long, "
      "but got ",
      THPUtils_typename(offset));
  uint64_t unsigned_offset = unpack_uint64(offset);
  // See Note [Acquire lock when using random generators]
  std::scoped_lock<std::mutex> lock(generator.mutex());
  generator.set_offset(unsigned_offset);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_seed(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  // See Note [Acquire lock when using random generators]
  auto self = (THPGenerator*)_self;
  std::scoped_lock<std::mutex> lock(self->cdata.mutex());
  uint64_t seed_val = self->cdata.seed();
  return THPUtils_packUInt64(seed_val);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_initialSeed(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  return THPUtils_packUInt64(self->cdata.current_seed());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_getOffset(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  return THPUtils_packUInt64(self->cdata.get_offset());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_get_device(THPGenerator* self, void* unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(self->cdata.device());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_reduce(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THPGenerator*)_self;
  auto& gen = self->cdata;

  auto ret = THPObjectPtr{PyTuple_New(3)};
  if (!ret)
    throw python_error();

  py::object torch_module = py::module::import("torch");
  py::object torch_generator = torch_module.attr("Generator");
  PyTuple_SET_ITEM(ret.get(), 0, torch_generator.release().ptr());

  auto args = THPObjectPtr{PyTuple_New(1)};
  if (!args)
    throw python_error();

  PyTuple_SET_ITEM(args.get(), 0, THPGenerator_get_device(self, nullptr));
  PyTuple_SET_ITEM(ret.get(), 1, args.release());

  auto state = THPObjectPtr{PyTuple_New(3)};
  if (!state)
    throw python_error();

  c10::DeviceType device_type = gen.device().type();
  PyTuple_SET_ITEM(state.get(), 0, THPGenerator_initialSeed(_self, nullptr));
  PyTuple_SET_ITEM(
      state.get(),
      1,
      device_type != at::kCPU ? THPGenerator_getOffset(_self, nullptr)
                              : Py_None);
  PyTuple_SET_ITEM(state.get(), 2, THPGenerator_getState(_self, nullptr));
  PyTuple_SET_ITEM(ret.get(), 2, state.release());

  return ret.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THPGenerator_pickleSetState(PyObject* _self, PyObject* state) {
  HANDLE_TH_ERRORS
  THPGenerator_manualSeed(_self, PyTuple_GET_ITEM(state, 0));
  auto& offset = PyTuple_GET_ITEM(state, 1);
  if (offset != Py_None) {
    THPGenerator_setOffset(_self, offset);
  }
  THPGenerator_setState(_self, PyTuple_GET_ITEM(state, 2));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPGenerator_properties[] = {
    {"device", (getter)THPGenerator_get_device, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THPGenerator_methods[] = {
    {"__reduce__", THPGenerator_reduce, METH_NOARGS, nullptr},
    {"__setstate__", THPGenerator_pickleSetState, METH_O, nullptr},
    {"get_state", THPGenerator_getState, METH_NOARGS, nullptr},
    {"set_state", THPGenerator_setState, METH_O, nullptr},
    {"clone_state", THPGenerator_cloneState, METH_NOARGS, nullptr},
    {"graphsafe_get_state",
     THPGenerator_graphSafeGetState,
     METH_NOARGS,
     nullptr},
    {"graphsafe_set_state", THPGenerator_graphSafeSetState, METH_O, nullptr},
    {"set_offset", THPGenerator_setOffset, METH_O, nullptr},
    {"manual_seed", THPGenerator_manualSeed, METH_O, nullptr},
    {"seed", THPGenerator_seed, METH_NOARGS, nullptr},
    {"initial_seed", THPGenerator_initialSeed, METH_NOARGS, nullptr},
    {"get_offset", THPGenerator_getOffset, METH_NOARGS, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMemberDef THPGenerator_members[] = {
    {"_cdata", T_ULONGLONG, offsetof(THPGenerator, cdata), READONLY, nullptr},
    {nullptr}};

static PyTypeObject THPGeneratorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C.Generator", /* tp_name */
    sizeof(THPGenerator), /* tp_basicsize */
    0, /* tp_itemsize */
    THPGenerator_dealloc, /* tp_dealloc */
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
    THPGenerator_methods, /* tp_methods */
    THPGenerator_members, /* tp_members */
    THPGenerator_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THPGenerator_pynew, /* tp_new */
};

bool THPGenerator_init(PyObject* module) {
  THPGeneratorClass = (PyObject*)&THPGeneratorType;
  if (PyType_Ready(&THPGeneratorType) < 0)
    return false;
  Py_INCREF(&THPGeneratorType);
  PyModule_AddObject(module, "Generator", (PyObject*)&THPGeneratorType);
  return true;
}

static void set_pyobj(const Generator& self, PyObject* pyobj) {
  TORCH_CHECK(self.defined(), "cannot call set_pyobj() on undefined generator");
  self.set_pyobj(pyobj);
}

static PyObject* pyobj(const Generator& self) {
  TORCH_CHECK(self.defined(), "cannot call pyobj() on undefined generator");
  return self.pyobj();
}

PyObject* THPGenerator_Wrap(const Generator& gen) {
  if (!gen.defined()) {
    Py_RETURN_NONE;
  }

  if (auto obj = pyobj(gen)) {
    Py_INCREF(obj);
    return obj;
  }

  return THPGenerator_NewWithVar((PyTypeObject*)THPGeneratorClass, gen);
}

at::Generator THPGenerator_Unwrap(PyObject* state) {
  if (!Py_IS_TYPE(state, &THPGeneratorType)) {
    TORCH_CHECK_TYPE(
        false,
        fmt::format(
            "expected a Generator, but got {}", Py_TYPE(state)->tp_name));
  }
  return reinterpret_cast<THPGenerator*>(state)->cdata;
}

// Creates a new Python object for a Generator. The Generator must not already
// have a PyObject* associated with it.
PyObject* THPGenerator_NewWithVar(PyTypeObject* type, Generator gen) {
  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto g = (THPGenerator*)obj;
    new (&g->cdata) Generator(std::move(gen));
    set_pyobj(g->cdata, obj);
  }
  return obj;
}
