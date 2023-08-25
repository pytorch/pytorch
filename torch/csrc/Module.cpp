#include <c10/util/Optional.h>
#include <fmt/core.h>
#include <sys/types.h>
#include <torch/csrc/python_headers.h>

#ifndef _MSC_VER
#include <sys/socket.h>
#endif

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/LegacyVmapMode.h>
#include <ATen/LinalgBackend.h>
#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <ATen/core/Vitals.h>
#include <ATen/dlpack.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/util/Backtrace.h>
#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <libshm.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/THConcat.h>
#include <torch/csrc/utils/pybind.h>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include <ATen/ThreadLocalPythonObjects.h>
#include <torch/csrc/DataLoader.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/TypeInfo.h>
#include <torch/csrc/api/include/torch/python/init.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_enum_tag.h>
#include <torch/csrc/autograd/python_fft_functions.h>
#include <torch/csrc/autograd/python_function.h>
#include <torch/csrc/autograd/python_legacy_variable.h>
#include <torch/csrc/autograd/python_linalg_functions.h>
#include <torch/csrc/autograd/python_nested_functions.h>
#include <torch/csrc/autograd/python_nn_functions.h>
#include <torch/csrc/autograd/python_return_types.h>
#include <torch/csrc/autograd/python_sparse_functions.h>
#include <torch/csrc/autograd/python_special_functions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/cpu/Module.h>
#include <torch/csrc/dynamo/init.h>
#include <torch/csrc/functorch/init.h>
#include <torch/csrc/jit/python/init.h>
#include <torch/csrc/jit/python/python_ir.h>
#include <torch/csrc/jit/python/python_tracer.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/lazy/python/init.h>
#include <torch/csrc/monitor/python_init.h>
#include <torch/csrc/mps/Module.h>
#include <torch/csrc/multiprocessing/init.h>
#include <torch/csrc/onnx/init.h>
#include <torch/csrc/profiler/python/init.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/init.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/tensor_layouts.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>
#include <torch/csrc/utils/tensor_qschemes.h>
#include <torch/csrc/utils/verbose.h>

#ifdef USE_DISTRIBUTED
#ifdef USE_C10D
#include <torch/csrc/distributed/autograd/python_autograd.h>
#include <torch/csrc/distributed/c10d/c10d.h>
#include <torch/csrc/distributed/rpc/rpc.h>
#include <torch/csrc/distributed/rpc/testing/testing.h>
#endif
#endif

#if defined(USE_VALGRIND)
#include <callgrind.h>
#endif

namespace py = pybind11;

PyObject* module;

THPGenerator* THPDefaultCPUGenerator = nullptr;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static PyObject* THPModule_initNames(PyObject* self, PyObject* arg) {
  static std::vector<std::string> names;

  THPObjectPtr types(PySequence_Fast(arg, "expected a sequence"));
  if (!types)
    return nullptr;

  // NOLINTNEXTLINE(bugprone-branch-clone)
  auto num_classes = PySequence_Fast_GET_SIZE(types.get());
  names.reserve(names.size() + num_classes);
  for (Py_ssize_t i = 0; i < num_classes; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(types.get(), i);
    THPUtils_assert(PyType_Check(obj), "expected a PyTypeObject");
    PyTypeObject* type = (PyTypeObject*)obj;

    THPObjectPtr module_name(PyObject_GetAttrString(obj, "__module__"));
    if (!module_name)
      return nullptr;
    THPUtils_assert(
        THPUtils_checkString(module_name.get()),
        "expected __module__ to be a string");
    std::string name = THPUtils_unpackString(module_name.get());
    names.emplace_back(name + "." + type->tp_name);
    type->tp_name = names.back().c_str();
  }
  Py_RETURN_NONE;
}
//
// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THPModule_initExtension(
    PyObject* _unused,
    PyObject* shm_manager_path) {
  HANDLE_TH_ERRORS
  if (!THPUtils_checkString(shm_manager_path)) {
    THPUtils_setError(
        "initialization error - expected bytes/string object as shm_manager_path!");
    return nullptr;
  }
  torch::utils::initializeLayouts();
  torch::utils::initializeMemoryFormats();
  torch::utils::initializeQSchemes();
  torch::utils::initializeDtypes();
  torch::tensors::initialize_python_bindings();
  std::string path = THPUtils_unpackString(shm_manager_path);
  libshm_init(path.c_str());

  auto module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!module)
    throw python_error();

  THPStorage_postInit(module);
  THPAutograd_initFunctions();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// The idea behind these two functions is to make it easy to test if we are
// built with ASAN: they're designed not to crash if ASAN is not enabled, but
// to trigger ASAN if it is enabled.  This lets us run a "canary" tests which
// checks if our build environment is misconfigured.

static PyObject* THPModule_crashIfCsrcASAN(PyObject* module, PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "crash_if_csrc_asan expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
  volatile char x[3];
  x[THPUtils_unpackInt(arg)] = 0;
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  return THPUtils_packInt32(x[0]);
}

static PyObject* THPModule_crashIfCsrcUBSAN(PyObject* module, PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "crash_if_csrc_ubsan expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  int32_t x = THPUtils_unpackInt(arg);
  double y = 1.0 / x;
  return THPUtils_packInt32((int)y);
}

static PyObject* THPModule_crashIfvptrUBSAN(PyObject* module, PyObject* noarg) {
  // This code should work perfectly fine, as vtables are identical for Foo and
  // Baz unless rtti and ubsan are enabled
  struct Foo {
    virtual int bar() = 0;
    virtual ~Foo() = default;
  };
  struct Baz {
    virtual int bar() {
      return 17;
    }
    virtual ~Baz() = default;
  };
  Baz x{};
  auto y = static_cast<Foo*>(static_cast<void*>(&x));
  auto rc = y->bar();
  return THPUtils_packInt32(rc);
}

static PyObject* THPModule_crashIfATenASAN(PyObject* module, PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "crash_if_aten_asan expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  return THPUtils_packInt32(at::_crash_if_asan(THPUtils_unpackInt(arg)));
}

static PyObject* THPModule_crashIfDebugAssertsFail(
    PyObject* module,
    PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "crash_if_debug_asserts_fail expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      THPUtils_unpackInt(arg) != 424242,
      "Expect anything but 424242 as an input for debug builds");
  return THPUtils_packInt32(0);
}

static PyObject* THPModule_getNumThreads(PyObject* module, PyObject* noargs) {
  return THPUtils_packInt32(at::get_num_threads());
}

static PyObject* THPModule_setNumThreads(PyObject* module, PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "set_num_threads expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  int nthreads = (int)THPUtils_unpackLong(arg);
  THPUtils_assert(nthreads > 0, "set_num_threads expects a positive integer");
  at::set_num_threads(nthreads);
  Py_RETURN_NONE;
}

static PyObject* THPModule_getNumInteropThreads(
    PyObject* module,
    PyObject* noargs) {
  return THPUtils_packInt32(at::get_num_interop_threads());
}

static PyObject* THPModule_setNumInteropThreads(
    PyObject* module,
    PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "set_num_interop_threads expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  int nthreads = (int)THPUtils_unpackLong(arg);
  THPUtils_assert(
      nthreads > 0, "set_num_interop_threads expects a positive integer");
  at::set_num_interop_threads(nthreads);
  Py_RETURN_NONE;
}

PyObject* THPModule_setDefaultTensorType(PyObject* _unused, PyObject* type) {
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_tensor_type(type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setDefaultDtype(PyObject* _unused, PyObject* dtype) {
  HANDLE_TH_ERRORS
  torch::tensors::py_set_default_dtype(dtype);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_addDocStr(PyObject* _unused, PyObject* args) {
  // adds a __doc__ string to a function, similar to numpy's arr_add_docstring
  static std::vector<std::string> all_docs;
  PyObject* obj = nullptr;
  PyObject* doc_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &obj, &doc_obj)) {
    return nullptr;
  }

  const char* doc_str = "<invalid string>";
  if (THPUtils_checkString(doc_obj)) {
    all_docs.push_back(THPUtils_unpackString(doc_obj));
    doc_str = all_docs.back().c_str();
  }

  if (Py_TYPE(obj) == &PyCFunction_Type) {
    PyCFunctionObject* f = (PyCFunctionObject*)obj;
    if (f->m_ml->ml_doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "function '%s' already has a docstring",
          f->m_ml->ml_name);
    }
    f->m_ml->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "method_descriptor") == 0) {
    PyMethodDescrObject* m = (PyMethodDescrObject*)obj;
    if (m->d_method->ml_doc) {
      return PyErr_Format(
          PyExc_RuntimeError,
          "method '%s' already has a docstring",
          m->d_method->ml_name);
    }
    m->d_method->ml_doc = doc_str;
  } else if (strcmp(Py_TYPE(obj)->tp_name, "getset_descriptor") == 0) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    PyGetSetDescrObject* m = (PyGetSetDescrObject*)obj;
    if (m->d_getset->doc) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      return PyErr_Format(
          PyExc_RuntimeError,
          "attribute '%s' already has a docstring",
          m->d_getset->name);
    }
    // This field is not const for python < 3.7 yet the content is
    // never modified.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    m->d_getset->doc = const_cast<char*>(doc_str);
  } else if (Py_TYPE(obj) == &PyType_Type) {
    PyTypeObject* t = (PyTypeObject*)obj;
    if (t->tp_doc) {
      return PyErr_Format(
          PyExc_RuntimeError, "Type '%s' already has a docstring", t->tp_name);
    }
    t->tp_doc = doc_str;
  } else {
    return PyErr_Format(
        PyExc_TypeError,
        "don't know how to add docstring to type '%s'",
        Py_TYPE(obj)->tp_name);
  }

  Py_INCREF(obj);
  return obj;
}

PyObject* THPModule_inferSize(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  Py_ssize_t num_args = args ? (Py_ssize_t)PyTuple_Size(args) : 0;
  THPUtils_assert(num_args == 2, "expected exactly 2 arguments");
  PyObject* arg1 = PyTuple_GET_ITEM(args, 0);
  THPUtils_assert(THPSize_Check(arg1), "expected a torch.Size as argument 1");
  PyObject* arg2 = PyTuple_GET_ITEM(args, 1);
  THPUtils_assert(THPSize_Check(arg2), "expected a torch.Size as argument 2");

  auto size1 = THPUtils_unpackLongs(arg1);
  auto size2 = THPUtils_unpackLongs(arg2);
  auto sizes = at::infer_size(size1, size2);
  return THPSize_NewFromSizes(static_cast<int64_t>(sizes.size()), sizes.data());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_setBackcompatBroadcastWarn(
    PyObject* module,
    PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_backcompat_broadcast_warn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  setBackCompatBroadcastWarn(arg == Py_True);
  Py_RETURN_NONE;
}

static PyObject* THPModule_getBackcompatBroadcastWarn(
    PyObject* module,
    PyObject* noargs) {
  if (getBackCompatBroadcastWarn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyObject* THPModule_setBackcompatKeepdimWarn(
    PyObject* module,
    PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_backcompat_keepdim_warn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  setBackCompatKeepdimWarn(arg == Py_True);
  Py_RETURN_NONE;
}

static PyObject* THPModule_getBackcompatKeepdimWarn(
    PyObject* module,
    PyObject* noargs) {
  if (getBackCompatKeepdimWarn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_hasDistributed(PyObject* _unused, PyObject* noargs) {
#ifdef USE_DISTRIBUTED
  Py_RETURN_TRUE;
#else
  Py_RETURN_FALSE;
#endif
}

static PyObject* THPModule_showConfig(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::show_config());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_cxxFlags(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_cxx_flags());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_parallelInfo(PyObject* module, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_parallel_info());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_getCpuCapability(
    PyObject* module,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(at::get_cpu_capability());
  END_HANDLE_TH_ERRORS
}

void DLPack_Capsule_Destructor(PyObject* data) {
  if (C10_LIKELY(!PyCapsule_IsValid(data, "dltensor"))) {
    // early out, see DLPack spec: if a consuming library sets the capsule
    // name to something else, they own it and we don't need to do anything
    return;
  }
  HANDLE_TH_ERRORS
  // Causes overheads for validity checks again, but this case is rare
  // since consuming libraries should rename the capsule according to spec.
  // Note that this cannot set a python error (we checked validity above),
  // so we don't need to handle python error state here.
  DLManagedTensor* dlMTensor =
      (DLManagedTensor*)PyCapsule_GetPointer(data, "dltensor");
  // the dlMTensor has not been consumed, call deleter ourselves.
  // DLPack spec mentions that deleter may be NULL, but deleter from
  // `at::toDLPack` is never NULL, so no need for an additional check here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  dlMTensor->deleter(const_cast<DLManagedTensor*>(dlMTensor));
  END_HANDLE_TH_ERRORS_RET()
}

PyObject* THPModule_toDLPack(PyObject* _unused, PyObject* data) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPVariable_Check(data), "data must be a Tensor");
  DLManagedTensor* dlMTensor = at::toDLPack(THPVariable_Unpack(data));
  return PyCapsule_New(dlMTensor, "dltensor", DLPack_Capsule_Destructor);
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_fromDLPack(PyObject* _unused, PyObject* data) {
  using namespace torch::autograd;
  HANDLE_TH_ERRORS
  auto tensor = torch::utils::tensor_fromDLPack(data);
  return THPVariable_Wrap(tensor);
  END_HANDLE_TH_ERRORS
}

PyObject* THModule_getCppBacktrace(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  size_t frames_to_skip = 0;
  size_t maximum_number_of_frames = 0;
  if (!PyArg_ParseTuple(
          args, "LL", &frames_to_skip, &maximum_number_of_frames)) {
    return nullptr;
  }
  return THPUtils_packString(
      c10::get_backtrace(frames_to_skip, maximum_number_of_frames, true));
  END_HANDLE_TH_ERRORS
}

static PyObject* THModule_rename_privateuse1_backend(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkString(arg),
      "_rename_privateuse1_backend expects a str, "
      "but got %s",
      THPUtils_typename(arg));
  const std::string backend_name = THPUtils_unpackString(arg);
  c10::register_privateuse1_backend(backend_name);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THModule_get_privateuse1_backend_name(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::get_privateuse1_backend());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setAllowTF32CuDNN(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_allow_tf32_cublas expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setAllowTF32CuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_allowTF32CuDNN(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().allowTF32CuDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_setFloat32MatmulPrecision(
    PyObject* _unused,
    PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkString(arg),
      "set_float32_matmul_precision expects a str, "
      "but got %s",
      THPUtils_typename(arg));
  std::string s = THPUtils_unpackString(arg);
  at::globalContext().setFloat32MatmulPrecision(s);
  Py_RETURN_NONE;
}

PyObject* THPModule_float32MatmulPrecision(
    PyObject* _unused,
    PyObject* noargs) {
  std::string s = "highest";
  auto p = at::globalContext().float32MatmulPrecision();
  if (p == at::Float32MatmulPrecision::HIGH) {
    s = "high";
  } else if (p == at::Float32MatmulPrecision::MEDIUM) {
    s = "medium";
  }
  return THPUtils_packString(s);
}
PyObject* THPModule_setSDPUseFlash(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseFlash(arg == Py_True);
  Py_RETURN_NONE;
}
PyObject* THPModule_userEnabledFlashSDP(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().userEnabledFlashSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
PyObject* THPModule_setSDPUseMemEfficient(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseMemEfficient(arg == Py_True);
  Py_RETURN_NONE;
}
PyObject* userEnabledMemEfficientSDP(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().userEnabledMemEfficientSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
PyObject* THPModule_setSDPUseMath(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_sdp_use_math expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setSDPUseMath(arg == Py_True);
  Py_RETURN_NONE;
}
PyObject* THPModule_userEnabledMathSDP(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().userEnabledMathSDP())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}
PyObject* THPModule_setUserEnabledCuDNN(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_enabled_cudnn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setUserEnabledCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_userEnabledCuDNN(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().userEnabledCuDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_setUserEnabledMkldnn(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_enabled_mkldnn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setUserEnabledMkldnn(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_userEnabledMkldnn(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().userEnabledMkldnn())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_setDeterministicCuDNN(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      PyBool_Check(arg),
      "set_deterministic_cudnn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setDeterministicCuDNN(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_deterministicCuDNN(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().deterministicCuDNN())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_setDeterministicAlgorithms(
    PyObject* _unused,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"_set_deterministic_algorithms(bool mode, *, bool warn_only=False)"});
  torch::ParsedArgs<2> parsed_args{};
  auto r = parser.parse(args, kwargs, parsed_args);
  bool mode = r.toBool(0);
  bool warn_only = r.toBool(1);
  at::globalContext().setDeterministicAlgorithms(mode, warn_only);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_deterministicAlgorithms(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().deterministicAlgorithms()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_deterministicAlgorithmsWarnOnly(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().deterministicAlgorithmsWarnOnly()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_setWarnAlways(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "setWarnOnlyOnce expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  c10::WarningUtils::set_warnAlways(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_warnAlways(PyObject* _unused, PyObject* noargs) {
  if (c10::WarningUtils::get_warnAlways()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

// Used only for testing C++ to Python warning translations.
PyObject* THPModule_warn(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_WARN("Test message for TORCH_WARN");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// Used only for testing C++ to Python warning translations.
PyObject* THPModule_warnDeprecation(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_WARN_DEPRECATION("Test message for TORCH_WARN_DEPRECATION");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setBenchmarkCuDNN(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_benchmark_cudnn expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setBenchmarkCuDNN(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_benchmarkCuDNN(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().benchmarkCuDNN()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_setAllowTF32CuBLAS(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_allow_tf32_cublas expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setAllowTF32CuBLAS(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_allowTF32CuBLAS(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().allowTF32CuBLAS()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_setAllowFP16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_allow_fp16_reduction_cublas expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setAllowFP16ReductionCuBLAS(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_allowFP16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().allowFP16ReductionCuBLAS()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_setAllowBF16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_allow_bf16_reduction_cublas expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setAllowBF16ReductionCuBLAS(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_allowBF16ReductionCuBLAS(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().allowBF16ReductionCuBLAS()) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_setFlushDenormal(PyObject* _unused, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "flush_denormal expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  if (!at::globalContext().setFlushDenormal(arg == Py_True)) {
    Py_RETURN_FALSE;
  };
  Py_RETURN_TRUE;
}

PyObject* THPModule_getDefaultDtype(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto scalar_type = torch::tensors::get_default_scalar_type();
  auto dtype = (PyObject*)torch::getTHPDtype(scalar_type);
  Py_INCREF(dtype);
  return dtype;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getDefaultDevice(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::DeviceTypeName(
      dispatchKeyToDeviceType(torch::tensors::get_default_dispatch_key()),
      /*lower_case=*/true));
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setQEngine(PyObject* /* unused */, PyObject* arg) {
  THPUtils_assert(
      THPUtils_checkLong(arg),
      "set_qengine expects an int, "
      "but got %s",
      THPUtils_typename(arg));
  HANDLE_TH_ERRORS
  auto qengine = THPUtils_unpackLong(arg);
  at::globalContext().setQEngine(static_cast<at::QEngine>(qengine));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_qEngine(PyObject* _unused, PyObject* noargs) {
  return THPUtils_packInt64(
      static_cast<int64_t>(at::globalContext().qEngine()));
}

PyObject* THPModule_supportedQEngines(PyObject* _unused, PyObject* noargs) {
  auto qengines = at::globalContext().supportedQEngines();
  auto list =
      THPObjectPtr(PyList_New(static_cast<Py_ssize_t>(qengines.size())));
  if (!list)
    return nullptr;
  for (const auto i : c10::irange(qengines.size())) {
    PyObject* i64 = THPUtils_packInt64(static_cast<int64_t>(qengines[i]));
    if (!i64)
      return nullptr;
    PyList_SET_ITEM(list.get(), i, i64);
  }
  return list.release();
}

PyObject* THPModule_isEnabledXNNPACK(PyObject* _unused, PyObject* noargs) {
  if (at::globalContext().isXNNPACKAvailable())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_setCheckSparseTensorInvariants(
    PyObject* _unused,
    PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_check_sparse_tensor_invariants expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setCheckSparseTensorInvariants(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModule_checkSparseTensorInvariants(
    PyObject* _unused,
    PyObject* noargs) {
  if (at::globalContext().checkSparseTensorInvariants())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

PyObject* THPModule_willEngineExecuteNode(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  bool isTHPFunction = THPFunction_Check(arg);
  bool isTHPCppFunction = torch::autograd::THPCppFunction_Check(arg);
  THPUtils_assert(
      isTHPFunction || isTHPCppFunction,
      "_will_engine_execute_node expects an grad_fn, "
      "but got %s",
      THPUtils_typename(arg));
  const auto exec_info = torch::autograd::get_current_graph_task_exec_info();
  THPUtils_assert(
      exec_info,
      "_get_should_execute_nodes should only be called during the backward pass");
  torch::autograd::Node* node = nullptr;
  std::shared_ptr<torch::autograd::Node> node_sp;
  if (isTHPFunction) {
    node_sp = ((THPFunction*)arg)->cdata.lock();
    node = node_sp.get();
  } else {
    node = ((torch::autograd::THPCppFunction*)arg)->cdata.get();
  }
  const auto nodes_in_graph =
      torch::autograd::get_current_graph_task_nodes_in_graph();
  bool ret = nodes_in_graph->find(node) != nodes_in_graph->end();
  if (ret && !exec_info->empty()) {
    auto it = exec_info->find(node);
    if (it == exec_info->end() || !it->second.should_execute()) {
      ret = false;
    } else {
      TORCH_CHECK(
          !(node->topological_nr() == 0 && it->second.captures_),
          "A leaf node was passed to _will_engine_execute_node but we are "
          "currently running autograd.grad(). This is currently not supported.");
    }
  }
  if (ret) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getCurrentGraphTaskExecutionOrder(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  std::vector<torch::autograd::Node*> nodes =
      torch::autograd::get_current_graph_task_execution_order();
  TORCH_CHECK(
      !nodes.empty(),
      "_current_graph_task_execution_order should only be called during the backward pass");
  auto list = THPObjectPtr(PyList_New(static_cast<Py_ssize_t>(nodes.size())));
  if (!list)
    return nullptr;
  for (const auto i : c10::irange(nodes.size())) {
    // This node is guaranteed to be alive since the backward is still running
    PyObject* pyobj_node =
        torch::autograd::functionToPyObject(nodes[i]->getptr());
    PyList_SET_ITEM(list.get(), i, pyobj_node);
  }
  return list.release();
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getCurrentGraphTaskId(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(torch::autograd::get_current_graph_task_id());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_getCurrentNode(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return torch::autograd::functionToPyObject(
      torch::autograd::get_current_node());
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_setDefaultMobileCPUAllocator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::globalContext().setDefaultMobileCPUAllocator();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_unsetDefaultMobileCPUAllocator(
    PyObject* _unused,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::globalContext().unsetDefaultMobileCPUAllocator();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_vmapmode_increment_nesting(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::increment_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_vmapmode_decrement_nesting(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(at::impl::VmapMode::decrement_nesting());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_set_display_vmap_fallback_warnings_mode(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      PyBool_Check(arg),
      "enabled must be a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::globalContext().setDisplayVmapFallbackWarnings(arg == Py_True);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPModule_are_vmap_fallback_warnings_enabled(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  if (at::globalContext().areVmapFallbackWarningsEnabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, cppcoreguidelines-avoid-non-const-global-variables, modernize-avoid-c-arrays)
static PyMethodDef TorchMethods[] = {
    {"_initExtension", THPModule_initExtension, METH_O, nullptr},
    {"_autograd_init", THPAutograd_initExtension, METH_NOARGS, nullptr},
    {"_add_docstr", THPModule_addDocStr, METH_VARARGS, nullptr},
    {"_init_names", THPModule_initNames, METH_O, nullptr},
    {"_has_distributed", THPModule_hasDistributed, METH_NOARGS, nullptr},
    {"_set_default_tensor_type",
     THPModule_setDefaultTensorType,
     METH_O,
     nullptr},
    {"_set_default_dtype", THPModule_setDefaultDtype, METH_O, nullptr},
    {"_infer_size", THPModule_inferSize, METH_VARARGS, nullptr},
    {"_crash_if_csrc_asan", THPModule_crashIfCsrcASAN, METH_O, nullptr},
    {"_crash_if_csrc_ubsan", THPModule_crashIfCsrcUBSAN, METH_O, nullptr},
    {"_crash_if_vptr_ubsan", THPModule_crashIfvptrUBSAN, METH_NOARGS, nullptr},
    {"_crash_if_aten_asan", THPModule_crashIfATenASAN, METH_O, nullptr},
    {"_crash_if_debug_asserts_fail",
     THPModule_crashIfDebugAssertsFail,
     METH_O,
     nullptr},
    {"_show_config", THPModule_showConfig, METH_NOARGS, nullptr},
    {"_cxx_flags", THPModule_cxxFlags, METH_NOARGS, nullptr},
    {"_parallel_info", THPModule_parallelInfo, METH_NOARGS, nullptr},
    {"_get_cpu_capability", THPModule_getCpuCapability, METH_NOARGS, nullptr},
    {"_set_backcompat_broadcast_warn",
     THPModule_setBackcompatBroadcastWarn,
     METH_O,
     nullptr},
    {"_get_backcompat_broadcast_warn",
     THPModule_getBackcompatBroadcastWarn,
     METH_NOARGS,
     nullptr},
    {"_set_backcompat_keepdim_warn",
     THPModule_setBackcompatKeepdimWarn,
     METH_O,
     nullptr},
    {"_get_backcompat_keepdim_warn",
     THPModule_getBackcompatKeepdimWarn,
     METH_NOARGS,
     nullptr},
    {"get_num_threads", THPModule_getNumThreads, METH_NOARGS, nullptr},
    {"set_num_threads", THPModule_setNumThreads, METH_O, nullptr},
    {"get_num_interop_threads",
     THPModule_getNumInteropThreads,
     METH_NOARGS,
     nullptr},
    {"set_num_interop_threads",
     THPModule_setNumInteropThreads,
     METH_O,
     nullptr},
    {"_get_flash_sdp_enabled",
     THPModule_userEnabledFlashSDP,
     METH_NOARGS,
     nullptr},
    {"_set_sdp_use_flash", THPModule_setSDPUseFlash, METH_O, nullptr},
    {"_get_mem_efficient_sdp_enabled",
     userEnabledMemEfficientSDP,
     METH_NOARGS,
     nullptr},
    {"_set_sdp_use_mem_efficient",
     THPModule_setSDPUseMemEfficient,
     METH_O,
     nullptr},
    {"_get_math_sdp_enabled",
     THPModule_userEnabledMathSDP,
     METH_NOARGS,
     nullptr},
    {"_set_sdp_use_math", THPModule_setSDPUseMath, METH_O, nullptr},
    {"_get_cudnn_enabled", THPModule_userEnabledCuDNN, METH_NOARGS, nullptr},
    {"_set_cudnn_enabled", THPModule_setUserEnabledCuDNN, METH_O, nullptr},
    {"_get_mkldnn_enabled", THPModule_userEnabledMkldnn, METH_NOARGS, nullptr},
    {"_set_mkldnn_enabled", THPModule_setUserEnabledMkldnn, METH_O, nullptr},
    {"_get_cudnn_allow_tf32", THPModule_allowTF32CuDNN, METH_NOARGS, nullptr},
    {"_set_cudnn_allow_tf32", THPModule_setAllowTF32CuDNN, METH_O, nullptr},
    {"_get_cudnn_benchmark", THPModule_benchmarkCuDNN, METH_NOARGS, nullptr},
    {"_set_cudnn_benchmark", THPModule_setBenchmarkCuDNN, METH_O, nullptr},
    {"_get_cudnn_deterministic",
     THPModule_deterministicCuDNN,
     METH_NOARGS,
     nullptr},
    {"_set_cudnn_deterministic",
     THPModule_setDeterministicCuDNN,
     METH_O,
     nullptr},
    {"_get_deterministic_algorithms",
     THPModule_deterministicAlgorithms,
     METH_NOARGS,
     nullptr},
    {"_get_deterministic_algorithms_warn_only",
     THPModule_deterministicAlgorithmsWarnOnly,
     METH_NOARGS,
     nullptr},
    {"_set_deterministic_algorithms",
     castPyCFunctionWithKeywords(THPModule_setDeterministicAlgorithms),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_get_warnAlways", THPModule_warnAlways, METH_NOARGS, nullptr},
    {"_set_warnAlways", THPModule_setWarnAlways, METH_O, nullptr},
    {"_warn", THPModule_warn, METH_NOARGS, nullptr},
    {"_warn_deprecation", THPModule_warnDeprecation, METH_NOARGS, nullptr},
    {"_get_cublas_allow_tf32", THPModule_allowTF32CuBLAS, METH_NOARGS, nullptr},
    {"_set_cublas_allow_tf32", THPModule_setAllowTF32CuBLAS, METH_O, nullptr},
    {"_get_float32_matmul_precision",
     THPModule_float32MatmulPrecision,
     METH_NOARGS,
     nullptr},
    {"_set_float32_matmul_precision",
     THPModule_setFloat32MatmulPrecision,
     METH_O,
     nullptr},
    {"_get_cublas_allow_fp16_reduced_precision_reduction",
     THPModule_allowFP16ReductionCuBLAS,
     METH_NOARGS,
     nullptr},
    {"_set_cublas_allow_fp16_reduced_precision_reduction",
     THPModule_setAllowFP16ReductionCuBLAS,
     METH_O,
     nullptr},
    {"_get_cublas_allow_bf16_reduced_precision_reduction",
     THPModule_allowBF16ReductionCuBLAS,
     METH_NOARGS,
     nullptr},
    {"_set_cublas_allow_bf16_reduced_precision_reduction",
     THPModule_setAllowBF16ReductionCuBLAS,
     METH_O,
     nullptr},
    {"_vmapmode_increment_nesting",
     THPModule_vmapmode_increment_nesting,
     METH_NOARGS,
     nullptr},
    {"_vmapmode_decrement_nesting",
     THPModule_vmapmode_decrement_nesting,
     METH_NOARGS,
     nullptr},
    {"_debug_only_display_vmap_fallback_warnings",
     THPModule_set_display_vmap_fallback_warnings_mode,
     METH_O,
     nullptr},
    {"_debug_only_are_vmap_fallback_warnings_enabled",
     THPModule_are_vmap_fallback_warnings_enabled,
     METH_NOARGS,
     nullptr},
    {"_to_dlpack", THPModule_toDLPack, METH_O, nullptr},
    {"_from_dlpack", THPModule_fromDLPack, METH_O, nullptr},
    {"_get_cpp_backtrace", THModule_getCppBacktrace, METH_VARARGS, nullptr},
    {"_rename_privateuse1_backend",
     THModule_rename_privateuse1_backend,
     METH_O,
     nullptr},
    {"_get_privateuse1_backend_name",
     THModule_get_privateuse1_backend_name,
     METH_NOARGS,
     nullptr},
    {"set_flush_denormal", THPModule_setFlushDenormal, METH_O, nullptr},
    {"get_default_dtype", THPModule_getDefaultDtype, METH_NOARGS, nullptr},
    {"_get_default_device", THPModule_getDefaultDevice, METH_NOARGS, nullptr},
    {"_get_qengine", THPModule_qEngine, METH_NOARGS, nullptr},
    {"_set_qengine", THPModule_setQEngine, METH_O, nullptr},
    {"_supported_qengines", THPModule_supportedQEngines, METH_NOARGS, nullptr},
    {"_is_xnnpack_enabled", THPModule_isEnabledXNNPACK, METH_NOARGS, nullptr},
    {"_set_check_sparse_tensor_invariants",
     THPModule_setCheckSparseTensorInvariants,
     METH_O,
     nullptr},
    {"_check_sparse_tensor_invariants",
     THPModule_checkSparseTensorInvariants,
     METH_NOARGS,
     nullptr},
    {"_will_engine_execute_node",
     THPModule_willEngineExecuteNode,
     METH_O,
     nullptr},
    {"_current_graph_task_execution_order",
     THPModule_getCurrentGraphTaskExecutionOrder,
     METH_NOARGS,
     nullptr},
    {"_current_graph_task_id",
     THPModule_getCurrentGraphTaskId,
     METH_NOARGS,
     nullptr},
    {"_current_autograd_node", THPModule_getCurrentNode, METH_NOARGS, nullptr},
    {"_set_default_mobile_cpu_allocator",
     THPModule_setDefaultMobileCPUAllocator,
     METH_NOARGS,
     nullptr},
    {"_unset_default_mobile_cpu_allocator",
     THPModule_unsetDefaultMobileCPUAllocator,
     METH_NOARGS,
     nullptr},
    {"_is_torch_function_enabled",
     THPModule_isEnabledTorchFunction,
     METH_NOARGS,
     nullptr},
    {"_disabled_torch_function_impl",
     THPModule_disable_torch_function,
     METH_VARARGS,
     nullptr},
    {"_disabled_torch_dispatch_impl",
     THPModule_disable_torch_dispatch,
     METH_VARARGS,
     nullptr},
    {"_has_torch_function", THPModule_has_torch_function, METH_O, nullptr},
    {"_has_torch_function_unary",
     THPModule_has_torch_function_unary,
     METH_O,
     nullptr},
    {"_has_torch_function_variadic",
     (PyCFunction)(void (*)())THPModule_has_torch_function_variadic,
     METH_FASTCALL,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

void THCPStream_init(PyObject* module);
void THCPEvent_init(PyObject* module);
void THCPGraph_init(PyObject* module);

#ifdef USE_CUDA
PyMethodDef* THCPModule_methods();
namespace torch {
namespace cuda {

void initModule(PyObject* module);

}
} // namespace torch
#endif

#ifdef USE_ITT
namespace torch {
namespace profiler {
void initIttBindings(PyObject* module);
} // namespace profiler
} // namespace torch
#endif

static std::vector<PyMethodDef> methods;

// In Python we can't use the trick of C10_LOG_API_USAGE_ONCE
// Guaranteed to be invoked from Python under GIL, no locking on map needed
static void LogAPIUsageOnceFromPython(const std::string& event) {
  static std::unordered_set<std::string> seen;
  if (!seen.count(event)) {
    seen.insert(event);
    c10::LogAPIUsage(event);
  }
}

static void LogAPIUsageMetadataFromPython(
    const std::string& event,
    const std::map<std::string, std::string>& metadata_map) {
  c10::LogAPIUsageMetadata(event, metadata_map);
}

// Weak reference to tensor, used to test a tensor isn't leaked
class WeakTensorRef {
  c10::weak_intrusive_ptr<c10::TensorImpl> weakref_;

 public:
  WeakTensorRef(const at::Tensor& t) : weakref_(t.getIntrusivePtr()) {}

  bool expired() {
    return weakref_.expired();
  }
};

extern "C" C10_EXPORT PyObject* initModule();
// separate decl and defn for msvc error C2491
PyObject* initModule() {
  HANDLE_TH_ERRORS

  c10::initLogging();

  at::internal::lazy_init_num_threads();

  C10_LOG_API_USAGE_ONCE("torch.python.import");

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ASSERT_TRUE(cmd) \
  if (!(cmd))            \
  return nullptr

  THPUtils_addPyMethodDefs(methods, TorchMethods);
  THPUtils_addPyMethodDefs(methods, DataLoaderMethods);
  THPUtils_addPyMethodDefs(methods, torch::autograd::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::multiprocessing::python_functions());
  THPUtils_addPyMethodDefs(methods, torch::mps::python_functions());
#ifdef USE_CUDA
  THPUtils_addPyMethodDefs(methods, THCPModule_methods());
#endif
#if defined(USE_DISTRIBUTED) && defined(USE_C10D)
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::c10d::python_functions());
#ifndef _WIN32
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::rpc::python_functions());
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::autograd::python_functions());
  THPUtils_addPyMethodDefs(
      methods, torch::distributed::rpc::testing::python_functions());
#endif
#endif

  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torchmodule);
  ASSERT_TRUE(module);
  ASSERT_TRUE(THPGenerator_init(module));
  ASSERT_TRUE(THPException_init(module));
  THPSize_init(module);
  THPDtype_init(module);
  THPDTypeInfo_init(module);
  THPLayout_init(module);
  THPMemoryFormat_init(module);
  THPQScheme_init(module);
  THPDevice_init(module);
  THPStream_init(module);
  ASSERT_TRUE(THPVariable_initModule(module));
  ASSERT_TRUE(THPFunction_initModule(module));
  ASSERT_TRUE(THPEngine_initModule(module));
  // NOTE: We need to be able to access OperatorExportTypes from ONNX for use in
  // the export side of JIT, so this ONNX init needs to appear before the JIT
  // init.
  torch::onnx::initONNXBindings(module);
  torch::autograd::initEnumTag(module);
  torch::jit::initJITBindings(module);
  torch::monitor::initMonitorBindings(module);
  torch::impl::dispatch::initDispatchBindings(module);
  torch::dynamo::initDynamoBindings(module);
  torch::functorch::impl::initFuncTorchBindings(module);
  torch::throughput_benchmark::initThroughputBenchmarkBindings(module);
  torch::autograd::initReturnTypes(module);
  torch::autograd::initNNFunctions(module);
  torch::autograd::initFFTFunctions(module);
  torch::autograd::initLinalgFunctions(module);
  torch::autograd::initNestedFunctions(module);
  torch::autograd::initSparseFunctions(module);
  torch::autograd::initSpecialFunctions(module);
  torch::autograd::init_legacy_variable(module);
  torch::profiler::initPythonBindings(module);
  torch::python::init_bindings(module);
  torch::lazy::initLazyBindings(module);
#ifdef USE_ITT
  torch::profiler::initIttBindings(module);
#endif
#ifdef USE_CUDA
  torch::cuda::initModule(module);
#endif
  torch::cpu::initModule(module);
  torch::initVerboseBindings(module);
  ASSERT_TRUE(THPStorage_init(module));

#ifdef USE_CUDA
  // This will only initialise base classes and attach them to library namespace
  // They won't be ready for real usage until importing cuda module, that will
  // complete the process (but it defines Python classes before calling back
  // into C, so these lines have to execute first)..
  THCPStream_init(module);
  THCPEvent_init(module);
  THCPGraph_init(module);
#endif

  auto set_module_attr =
      [&](const char* name, PyObject* v, bool incref = true) {
        // PyModule_AddObject steals reference
        if (incref) {
          Py_INCREF(v);
        }
        return PyModule_AddObject(module, name, v) == 0;
      };

#if defined(USE_CUDNN) || defined(USE_ROCM)
  PyObject* has_cudnn = Py_True;
#else
  PyObject* has_cudnn = Py_False;
#endif
  ASSERT_TRUE(set_module_attr("_has_cudnn", has_cudnn));

#if AT_MKL_ENABLED() || AT_POCKETFFT_ENABLED()
  PyObject* has_spectral = Py_True;
#else
  PyObject* has_spectral = Py_False;
#endif
  ASSERT_TRUE(set_module_attr("has_spectral", has_spectral));

  // force ATen to initialize because it handles
  // setting up TH Errors so that they throw C++ exceptions
  at::init();

  // Automatically translate errors thrown from pybind11 functions
  py::register_exception_translator([](std::exception_ptr e) { // NOLINT
    try {
      if (e) {
        std::rethrow_exception(e);
      }
    }
    CATCH_TH_ERRORS()
  });

  auto py_module = py::reinterpret_borrow<py::module>(module);
  py_module.def("_demangle", &c10::demangle);
  py_module.def("_log_api_usage_once", &LogAPIUsageOnceFromPython);
  py_module.def("_log_api_usage_metadata", &LogAPIUsageMetadataFromPython);

  py_module.def("vitals_enabled", &at::vitals::torchVitalEnabled);
  py_module.def(
      "set_vital",
      [](const std::string& vital,
         const std::string& attr,
         const std::string& value) {
        return at::vitals::VitalsAPI.setVital(vital, attr, value);
      });
  py_module.def(
      "read_vitals", []() { return at::vitals::VitalsAPI.readVitals(); });

  py_module.def(
      "init_num_threads",
      torch::wrap_pybind_function(at::init_num_threads),
      R"(
init_num_threads()

Initializes the number of parallel threads used on the current thread.

Call this whenever a new thread is created in order to propagate values from
:func:`torch.set_num_threads` onto the new thread.
)");

  ASSERT_TRUE(
      set_module_attr("has_openmp", at::hasOpenMP() ? Py_True : Py_False));
  ASSERT_TRUE(set_module_attr("has_mkl", at::hasMKL() ? Py_True : Py_False));
  ASSERT_TRUE(
      set_module_attr("has_lapack", at::hasLAPACK() ? Py_True : Py_False));

  py_module.def("_valgrind_supported_platform", []() {
#if defined(USE_VALGRIND)
    return true;
#else
      return false;
#endif
  });

  py_module.def("_valgrind_toggle", []() {
#if defined(USE_VALGRIND)
    CALLGRIND_TOGGLE_COLLECT;
#else
      TORCH_CHECK(false, "Valgrind is not supported.");
#endif
  });

  py_module.def("_valgrind_toggle_and_dump_stats", []() {
#if defined(USE_VALGRIND)
    // NB: If we don't toggle collect around dump stats, callgrind_annotate
    //     won't process the results correctly. Specifically,
    //     `callgrind_annotate --inclusive=no` will be almost completely empty.
    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_DUMP_STATS;
#else
      TORCH_CHECK(false, "Valgrind is not supported.");
#endif
  });

  py::class_<WeakTensorRef>(py_module, "_WeakTensorRef")
      .def(py::init([](py::object tensor) {
        return WeakTensorRef(THPVariable_Unpack(tensor.ptr()));
      }))
      .def("expired", &WeakTensorRef::expired);

  py::enum_<at::native::ConvBackend>(py_module, "_ConvBackend")
      .value("CudaDepthwise2d", at::native::ConvBackend::CudaDepthwise2d)
      .value("CudaDepthwise3d", at::native::ConvBackend::CudaDepthwise3d)
      .value("Cudnn", at::native::ConvBackend::Cudnn)
      .value("CudnnTranspose", at::native::ConvBackend::CudnnTranspose)
      .value("Empty", at::native::ConvBackend::Empty)
      .value("Miopen", at::native::ConvBackend::Miopen)
      .value("MiopenDepthwise", at::native::ConvBackend::MiopenDepthwise)
      .value("MiopenTranspose", at::native::ConvBackend::MiopenTranspose)
      .value("Mkldnn", at::native::ConvBackend::Mkldnn)
      .value("MkldnnEmpty", at::native::ConvBackend::MkldnnEmpty)
      .value("NnpackSpatial", at::native::ConvBackend::NnpackSpatial)
      .value("Overrideable", at::native::ConvBackend::Overrideable)
      .value("Slow2d", at::native::ConvBackend::Slow2d)
      .value("Slow3d", at::native::ConvBackend::Slow3d)
      .value("SlowDilated2d", at::native::ConvBackend::SlowDilated2d)
      .value("SlowDilated3d", at::native::ConvBackend::SlowDilated3d)
      .value("SlowTranspose2d", at::native::ConvBackend::SlowTranspose2d)
      .value("SlowTranspose3d", at::native::ConvBackend::SlowTranspose3d)
      .value(
          "Winograd3x3Depthwise", at::native::ConvBackend::Winograd3x3Depthwise)
      .value("Xnnpack2d", at::native::ConvBackend::Xnnpack2d)
      .value("Mps", at::native::ConvBackend::Mps)
      .value("MpsTranspose,", at::native::ConvBackend::MpsTranspose);

  py_module.def(
      "_select_conv_backend",
      [](const at::Tensor& input,
         const at::Tensor& weight,
         const c10::optional<at::Tensor>& bias_opt,
         at::IntArrayRef stride_,
         at::SymIntArrayRef padding_,
         at::IntArrayRef dilation_,
         bool transposed_,
         at::SymIntArrayRef output_padding_,
         int64_t groups_) {
        return at::native::select_conv_backend(
            input,
            weight,
            bias_opt,
            stride_,
            padding_,
            dilation_,
            transposed_,
            output_padding_,
            groups_,
            c10::nullopt);
      },
      py::arg("input"),
      py::arg("weight"),
      py::arg("bias"),
      py::arg("stride"),
      py::arg("padding"),
      py::arg("dilation"),
      py::arg("transposed"),
      py::arg("output_padding"),
      py::arg("groups"));

  // overload for bias_sizes_opt/backward TODO: figure out default value
  py_module.def(
      "_select_conv_backend",
      [](const at::Tensor& input,
         const at::Tensor& weight,
         const c10::optional<at::Tensor>& bias,
         at::IntArrayRef stride_,
         at::SymIntArrayRef padding_,
         at::IntArrayRef dilation_,
         bool transposed_,
         at::SymIntArrayRef output_padding_,
         int64_t groups_,
         c10::optional<std::vector<c10::SymInt>> bias_sizes_opt) {
        c10::OptionalArrayRef<c10::SymInt> ref = c10::nullopt;
        if (bias_sizes_opt) {
          ref = (*bias_sizes_opt);
        }
        return at::native::select_conv_backend(
            input,
            weight,
            bias,
            stride_,
            padding_,
            dilation_,
            transposed_,
            output_padding_,
            groups_,
            ref);
      },
      py::arg("input"),
      py::arg("weight"),
      py::arg("bias"),
      py::arg("stride"),
      py::arg("padding"),
      py::arg("dilation"),
      py::arg("transposed"),
      py::arg("output_padding"),
      py::arg("groups"),
      py::arg("bias_sizes"));

  py_module.def(
      "_conv_determine_backend_memory_format",
      at::native::_determine_backend_memory_format);

  py::enum_<at::LinalgBackend>(py_module, "_LinalgBackend")
      .value("Default", at::LinalgBackend::Default)
      .value("Cusolver", at::LinalgBackend::Cusolver)
      .value("Magma", at::LinalgBackend::Magma);

  py_module.def("_set_linalg_preferred_backend", [](at::LinalgBackend b) {
    at::globalContext().setLinalgPreferredBackend(b);
  });
  py_module.def("_get_linalg_preferred_backend", []() {
    return at::globalContext().linalgPreferredBackend();
  });

  py_module.def(
      "_construct_storage_from_data_pointer",
      [](int64_t data_ptr, c10::Device device, size_t size_bytes) {
        return c10::Storage(
            c10::Storage::use_byte_size_t(),
            size_bytes,
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            at::DataPtr(reinterpret_cast<void*>(data_ptr), device));
      });

  py_module.def(
      "_stash_obj_in_tls", [](const std::string& key, py::handle arg) {
        at::impl::ThreadLocalPythonObjects::get_state().set(
            key,
            std::make_shared<c10::SafePyObject>(arg.ptr(), getPyInterpreter()));
      });

  py_module.def("_get_obj_in_tls", [](const std::string& key) -> py::handle {
    auto safe_pyobject =
        at::impl::ThreadLocalPythonObjects::get_state().get(key);
    auto obj = safe_pyobject->ptr(getPyInterpreter());
    return py::handle(obj);
  });

  py_module.def("_is_key_in_tls", [](const std::string& key) -> bool {
    return at::impl::ThreadLocalPythonObjects::get_state().contains(key);
  });

#ifdef USE_CUDA
  PyObject* has_cuda = Py_True;
#else
  PyObject* has_cuda = Py_False;
#endif

#ifdef USE_MPS
  PyObject* has_mps = Py_True;
#else
  PyObject* has_mps = Py_False;
#endif

  ASSERT_TRUE(set_module_attr("_has_cuda", has_cuda));
  ASSERT_TRUE(set_module_attr("_has_mps", has_mps));
  ASSERT_TRUE(
      set_module_attr("_has_mkldnn", at::hasMKLDNN() ? Py_True : Py_False));

#ifdef _GLIBCXX_USE_CXX11_ABI
  ASSERT_TRUE(set_module_attr(
      "_GLIBCXX_USE_CXX11_ABI", _GLIBCXX_USE_CXX11_ABI ? Py_True : Py_False));
#else
  ASSERT_TRUE(set_module_attr("_GLIBCXX_USE_CXX11_ABI", Py_False));
#endif

// See note [Pybind11 ABI constants]
#define SET_STR_DEFINE(name) \
  ASSERT_TRUE(set_module_attr("_" #name, THPUtils_packString(name)))

#ifdef PYBIND11_COMPILER_TYPE
  SET_STR_DEFINE(PYBIND11_COMPILER_TYPE);
#else
  ASSERT_TRUE(
      set_module_attr("_" C10_STRINGIZE(PYBIND11_COMPILER_TYPE), Py_None));
#endif

#ifdef PYBIND11_STDLIB
  SET_STR_DEFINE(PYBIND11_STDLIB);
#else
  ASSERT_TRUE(set_module_attr("_" C10_STRINGIZE(PYBIND11_STDLIB), Py_None));
#endif

#ifdef PYBIND11_BUILD_ABI
  SET_STR_DEFINE(PYBIND11_BUILD_ABI);
#else
  ASSERT_TRUE(set_module_attr("_" C10_STRINGIZE(PYBIND11_BUILD_ABI), Py_None));
#endif
#undef SET_STR_DEFINE

  py_module.def(
      "_set_conj", [](const at::Tensor& x, bool conj) { x._set_conj(conj); });
  py_module.def(
      "_set_neg", [](const at::Tensor& x, bool neg) { x._set_neg(neg); });
  py_module.def("_get_tensor_metadata", &torch::jit::getTensorMetadata);
  py_module.def(
      "_set_tensor_metadata",
      static_cast<void (*)(
          const at::Tensor&, std::unordered_map<std::string, bool>)>(
          torch::jit::setTensorMetadata));
  py_module.def("_dispatch_key_set", [](const at::Tensor& x) {
    return toString(x.key_set());
  });
  py_module.def(
      "_has_storage", [](const at::Tensor& x) { return x.has_storage(); });

  py_module.def("_set_meta_in_tls_dispatch_include", [](bool meta_in_tls) {
    auto local_keyset = c10::impl::tls_local_dispatch_key_set();
    c10::DispatchKeySet key_set({at::DispatchKey::Meta});
    if (meta_in_tls) {
      local_keyset.included_ = local_keyset.included_ | key_set;
    } else {
      local_keyset.included_ =
          local_keyset.included_.remove_backend(c10::BackendComponent::MetaBit);
    }
    c10::impl::_force_tls_local_dispatch_key_set(local_keyset);
  });

  py_module.def("_meta_in_tls_dispatch_include", []() {
    auto local_keyset = c10::impl::tls_local_dispatch_key_set();
    return local_keyset.included_.has_backend(c10::BackendComponent::MetaBit);
  });

  py_module.def("_dump_local_tls_set", []() {
    auto local_keyset = c10::impl::tls_local_dispatch_key_set();
    std::cout << "Included: " << toString(local_keyset.included_) << "\n";
    std::cout << "Excluded: " << toString(local_keyset.excluded_) << "\n";
  });

  py_module.def(
      "_should_allow_numbers_as_tensors", [](const std::string& name) {
        return torch::should_allow_numbers_as_tensors(name);
      });

  // FIXME(crcrpar): Better to have `at::ScalarType` get mapped to `torch.dtype`
  // Currently I see the second item of the key is displayed as
  // e.g. `torch._C._te.ScalarType at 0x7fcf318adab0`
  // I thought adding an appropriate type_caster of `at::ScalarType` to
  // torch/csrc/pybind.h` would solve this but it caused segmentation fault in
  // my environment.
  using _DeviceDtypeKey = std::pair<at::Device, std::string>;
  // Custom hasher is necessary to make unordered_map compilable for Windows
  // debug targets. As `at::native::ParamsHash` only works on structs with
  // standard layout, but std::string isn't one in Visual C++ debug builds,
  // which one can easily verify by running something like:
  //   #define _DEBUG
  //   #include <type_traits>
  //   #include <string>
  //   static_assert(std::is_standard_layout_v<std::string>, "Oh noes");
  // If above condition is not met, VC++ raises a very cryptic compilation
  // error. See
  // https://github.com/pytorch/pytorch/pull/100007#discussion_r1227116292 for
  // more detail
  struct _DeviceDtypeHasher {
    std::size_t operator()(const _DeviceDtypeKey& k) const noexcept {
      static at::native::ParamsHash<at::Device> device_hasher;
      static std::hash<std::string> string_hasher;
      return device_hasher(k.first) ^ string_hasher(k.second);
    }
  };
  using _FlatMap = std::unordered_map<
      _DeviceDtypeKey,
      at::native::TensorsAndIndicesT,
      _DeviceDtypeHasher>;
  py_module.def(
      "_group_tensors_by_device_and_dtype",
      [](const std::vector<std::vector<c10::optional<at::Tensor>>>&
             nested_tensorlist,
         const bool with_indices) {
        _FlatMap map;
        for (const auto& iter :
             at::native::_group_tensors_by_first_tensors_device_and_dtype(
                 nested_tensorlist, with_indices)) {
          const auto scalar_type_name =
              torch::utils::getDtypeNames(iter.first.second).first;
          map.insert({{iter.first.first, scalar_type_name}, iter.second});
        }
        return map;
      });

  const auto& defaultGenerator = at::detail::getDefaultCPUGenerator();
  THPDefaultCPUGenerator =
      (THPGenerator*)THPGenerator_initDefaultGenerator(defaultGenerator);
  // This reference is meant to be given away, so no need to incref here.
  ASSERT_TRUE(set_module_attr(
      "default_generator",
      (PyObject*)THPDefaultCPUGenerator,
      /* incref= */ false));
  ASSERT_TRUE(set_module_attr(
      "DisableTorchFunctionSubclass",
      (PyObject*)THPModule_DisableTorchFunctionSubclassType(),
      /* incref= */ false));
  ASSERT_TRUE(set_module_attr(
      "DisableTorchFunction",
      (PyObject*)THPModule_DisableTorchFunctionType(),
      /* incref= */ false));
  torch::set_disabled_torch_function_impl(
      PyObject_GetAttrString(module, "_disabled_torch_function_impl"));
  ASSERT_TRUE(torch::disabled_torch_function_impl() != nullptr);
  torch::set_disabled_torch_dispatch_impl(
      PyObject_GetAttrString(module, "_disabled_torch_dispatch_impl"));
  ASSERT_TRUE(torch::disabled_torch_dispatch_impl() != nullptr);
  return module;
  END_HANDLE_TH_ERRORS
}

// Checks that the _C shared library isn't initialized multiple times. This
// can happen if the same csrc files are compiled into multiple shared
// libraries.
inline void pytorch_duplicate_guard() {
  static int initialized = 0;
  if (initialized) {
    fmt::print(stderr, "pytorch: _C shared library re-initialized\n");
    abort();
  }
  initialized = 1;
  ;
}

struct call_duplicate_guard {
  call_duplicate_guard() {
    pytorch_duplicate_guard();
  }
};

static call_duplicate_guard _call_duplicate_guard;
