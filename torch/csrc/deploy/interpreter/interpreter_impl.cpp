#include <dlfcn.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <torch/csrc/deploy/interpreter/interpreter_impl.h>

#include <assert.h>
#include <pybind11/embed.h>
#include <stdio.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <iostream>
#include <map>
#include <thread>

#include <fmt/format.h>

namespace py = pybind11;
using namespace py::literals;

// TODO this should come from cmake
#define DEBUG 1

#if (DEBUG == 1)
#define PYOBJ_ASSERT(obj) \
  if (NULL == obj) {      \
    PyErr_Print();        \
  }                       \
  assert(NULL != obj);
#elif (DEBUG == 0)
#define PYOBJ_ASSERT(obj) assert(NULL != obj);
#endif

static wchar_t* program;

#define FOREACH_LIBRARY(_) \
  _(array)                 \
  _(_asyncio)              \
  _(audioop)               \
  _(binascii)              \
  _(_bisect)               \
  _(_blake2)               \
  _(_bz2)                  \
  _(cmath)                 \
  _(_codecs_cn)            \
  _(_codecs_hk)            \
  _(_codecs_iso2022)       \
  _(_codecs_jp)            \
  _(_codecs_kr)            \
  _(_codecs_tw)            \
  _(_contextvars)          \
  _(_crypt)                \
  _(_csv)                  \
  _(_ctypes)               \
  _(_ctypes_test)          \
  _(_curses)               \
  _(_curses_panel)         \
  _(_datetime)             \
  _(_decimal)              \
  _(_elementtree)          \
  _(fcntl)                 \
  _(grp)                   \
  _(_hashlib)              \
  _(_heapq)                \
  _(_json)                 \
  _(_lsprof)               \
  _(_lzma)                 \
  _(math)                  \
  _(_md5)                  \
  _(mmap)                  \
  _(_multibytecodec)       \
  _(_multiprocessing)      \
  _(nis)                   \
  _(_opcode)               \
  _(ossaudiodev)           \
  _(parser)                \
  _(_pickle)               \
  _(_posixsubprocess)      \
  _(pyexpat)               \
  _(_queue)                \
  _(_random)               \
  _(readline)              \
  _(resource)              \
  _(select)                \
  _(_sha1)                 \
  _(_sha256)               \
  _(_sha3)                 \
  _(_sha512)               \
  _(_socket)               \
  _(spwd)                  \
  _(_ssl)                  \
  _(_struct)               \
  _(syslog)                \
  _(termios)               \
  _(_testbuffer)           \
  _(_testcapi)             \
  _(_testimportmultiple)   \
  _(_testmultiphase)       \
  _(unicodedata)           \
  _(xxlimited)             \
  _(_xxtestfuzz)           \
  _(zlib)

#define DECLARE_LIBRARY_INIT(name) extern "C" PyObject* PyInit_##name(void);
FOREACH_LIBRARY(DECLARE_LIBRARY_INIT)
#undef DECLARE_LIBRARY_INIT

extern "C" PyObject* initModule(void);
extern "C" PyObject* PyInit__C(void);
extern "C" struct _frozen _PyImport_FrozenModules[];
extern "C" struct _frozen _PyImport_FrozenModules_torch[];

// We need to register a custom finder because we are registering `torch._C` as
// a built-in module, and it will otherwise get skipped by the default importer.
const char* startup = R"RAW(
import sys

class F:
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch._C':
            return sys.meta_path[1].find_spec('torch._C', None, None)
        elif fullname == 'maskrcnn_benchmark._C':
            return sys.meta_path[1].find_spec('maskrcnn_benchmark._C', None, None)
        return None
sys.meta_path.insert(0, F())
# make loader importable

import sys

import importlib.machinery
import importlib.util
spec = importlib.machinery.ModuleSpec('maskrcnn_benchmark', None, is_package=True)  # type: ignore
r = importlib.util.module_from_spec(spec)
sys.modules['maskrcnn_benchmark'] = r

# print("exec_prefix:", sys.base_exec_prefix)
# print("_base_executable:", sys._base_executable)
# print("base_prefix:", sys.base_prefix)
# print("exec_prefix:", sys.exec_prefix)
# print("executable:", sys.executable)
# print("path:", sys.path)
# print("prefix:", sys.prefix)
import torch # has to be done serially otherwise things will segfault
import torch.version # for some reason torch doesn't import this and cuda fails?
if torch.cuda.is_available():
  torch.zeros(1).cuda() # force cuda init...
import warnings
warnings.simplefilter("ignore")
)RAW";

// These numbers of modules should not change as long as the cpython version
// embedded in the build remains fixed
static const size_t NUM_FROZEN_PY_BUILTIN_MODULES = 6;
static const size_t NUM_FROZEN_PY_STDLIB_MODULES = 680;

// We need to preserve the existing FrozenModules list, since it includes
// important importlib machinery. This code is adapted from the similar
// `PyImport_ExtendInittab`.
int extendFrozenModules(
    struct _frozen* frozenpython,
    struct _frozen* frozentorch) {
  struct _frozen* p = nullptr;
  size_t a = 0, b = 0, c = 0;
  int res = 0;

  /* Count the number of entries in both tables */
  for (a = 0; frozenpython[a].name != nullptr; a++) {
    // std::cout << "frozenpython[" << a << "]: " << frozenpython[a].name <<
    // std::endl;
  }
  for (b = 0; frozentorch[b].name != nullptr; b++) {
    // std::cout << "frozentorch[" << b << "]: " << frozentorch[b].name <<
    // std::endl;
  }
  for (c = 0; PyImport_FrozenModules[c].name != nullptr; c++) {
    // std::cout << "oldfrozen[" << c << "]: " << PyImport_FrozenModules[c].name
    // << std::endl;
  }

  // Num frozen builtins shouldn't change (unless modifying the underlying
  // cpython version)
  TORCH_INTERNAL_ASSERT(
      c == NUM_FROZEN_PY_BUILTIN_MODULES,
      "Missing python builtin frozen modules");
  // Check a+b together since in OSS a is empty and b contains stdlib+torch,
  // while in fbcode they are separated due to thirdparty2 frozenpython. No
  // fixed number of torch modules to check for, but there should be at least
  // one.
  TORCH_INTERNAL_ASSERT(
      a + b > NUM_FROZEN_PY_STDLIB_MODULES + 1,
      "Missing frozen python stdlib or torch modules");

  /* Allocate new memory for the combined table */
  if (a + b + c <= SIZE_MAX / sizeof(struct _frozen) - 1) {
    size_t size = sizeof(struct _frozen) * (a + b + c + 1);
    p = (_frozen*)PyMem_Realloc(p, size);
  }
  if (p == nullptr) {
    return -1;
  }

  /* Copy the tables into the new memory */
  memcpy(p, PyImport_FrozenModules, (c + 1) * sizeof(struct _frozen));
  memcpy(p + c, frozenpython, (a + 1) * sizeof(struct _frozen));
  memcpy(p + a + c, frozentorch, (b + 1) * sizeof(struct _frozen));
  PyImport_FrozenModules = p;
  return res;
}

static py::object global_impl(const char* module, const char* name) {
  return py::module::import(module).attr(name);
}

using at::IValue;
using torch::PickledObject;
using torch::PythonObject;

struct ScopedAcquire {
  ScopedAcquire() {
    PyGILState_Ensure();
  }
  ~ScopedAcquire() {
    PyEval_SaveThread();
  }
};

struct InitLockAcquire {
  InitLockAcquire(std::mutex& init_lock) : init_lock_(init_lock) {
    // to avoid deadlock, we need to ensure a consistent lock order:
    // init_lock -> GIL. Otherwise, the GIL can be released by the python
    // interpreter during initalization tasks, and then re-acquired. If another
    // thread grabs the GIL to do non-initialization tasks, then it might start
    // initializing (GIL -> init_lock). To avoid this, releasethe GIL before
    // trying to get the init_lock and then reacquire it afterward.
    PyEval_SaveThread();
    init_lock.lock();
    PyGILState_Ensure();
  }
  ~InitLockAcquire() {
    init_lock_.unlock();
  }

 private:
  std::mutex& init_lock_;
};

struct ConcreteInterpreterImpl : public torch::InterpreterImpl {
  ConcreteInterpreterImpl() {
    // some dependency in mkl requires this...
    void* result = dlopen("libz.so", RTLD_GLOBAL | RTLD_LAZY);
    assert(result);

#define APPEND_INIT(name) PyImport_AppendInittab(#name, PyInit_##name);
    FOREACH_LIBRARY(APPEND_INIT)
#undef APPEND_INIT
    PyImport_AppendInittab("torch._C", initModule);
    // PyImport_AppendInittab("maskrcnn_benchmark._C", PyInit__C);

    int ret = extendFrozenModules(
        _PyImport_FrozenModules, _PyImport_FrozenModules_torch);
    TORCH_INTERNAL_ASSERT(ret == 0);

    PyPreConfig preconfig;
    PyPreConfig_InitIsolatedConfig(&preconfig);
    PyStatus status = Py_PreInitialize(&preconfig);
    TORCH_INTERNAL_ASSERT(!PyStatus_Exception(status))

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);

    // Completely blank out the path configuration. This ensures we have
    // complete control of how our embedded Python searches for modules, and we
    // will never consult the external filesystem. See:
    // https://docs.python.org/3/c-api/init_config.html#path-configuration
    config.site_import = 0;
    status = PyConfig_SetString(&config, &config.base_exec_prefix, L"");
    status =
        PyConfig_SetString(&config, &config.base_executable, L"torch_deploy");
    status = PyConfig_SetString(&config, &config.base_prefix, L"");
    status = PyConfig_SetString(&config, &config.exec_prefix, L"");
    status = PyConfig_SetString(&config, &config.executable, L"torch_deploy");
    status = PyConfig_SetString(&config, &config.prefix, L"");
    config.module_search_paths_set = 1;
    wchar_t* module_search_paths[0] = {};
    status = PyConfig_SetWideStringList(
        &config, &config.module_search_paths, 0, module_search_paths);

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    TORCH_INTERNAL_ASSERT(!PyStatus_Exception(status))

    int r = PyRun_SimpleString(startup);
    TORCH_INTERNAL_ASSERT(r == 0);

    // we cache these so we don't have to repeat the conversion of strings into
    // Python and hash table lookups to get to these object
    save_storage = global_impl("torch._deploy", "_save_storages");
    load_storage = global_impl("torch._deploy", "_load_storages");
    get_package = global_impl("torch._deploy", "_get_package");
    objects = global_impl("torch._deploy", "_deploy_objects");
    // Release the GIL that PyInitialize acquires
    PyEval_SaveThread();
  }
  ~ConcreteInterpreterImpl() override {
    PyGILState_Ensure();
    // make sure pybind11 doesn't try to decref after we have destroyed python
    // note: this leads the referneces to these objects, but we are about to
    // deinit python anyway so it doesn't matter
    objects.release();
    save_storage.release();
    load_storage.release();
    get_package.release();
    if(Py_FinalizeEx() != 0) {
      exit(1); // can't use TORCH_INTERNAL_ASSERT because we are in a non-throwing destructor.
    }
    PyMem_RawFree(program);
  }
  torch::InterpreterSessionImpl* acquire_session() override;
  py::object save_storage;
  py::object load_storage;
  py::object get_package;
  py::dict objects;
  std::mutex init_lock_;
};

struct ConcreteInterpreterSessionImpl : public torch::InterpreterSessionImpl {
  ConcreteInterpreterSessionImpl(ConcreteInterpreterImpl* interp)
      : interp_(interp) {}
  PythonObject global(const char* module, const char* name) override {
    return wrap(global_impl(module, name));
  }

  PythonObject from_ivalue(IValue value) override {
    return wrap(torch::jit::toPyObject(value));
  }
  PythonObject create_or_get_package_importer_from_container_file(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          container_file_) override {
    InitLockAcquire guard(interp_->init_lock_);
    return wrap(interp_->get_package(container_file_));
  }

  PickledObject pickle(PythonObject container, PythonObject obj) override {
    py::tuple result = interp_->save_storage(unwrap(container), unwrap(obj));
    py::bytes bytes = py::cast<py::bytes>(result[0]);
    py::list storages = py::cast<py::list>(result[1]);
    py::list dtypes = py::cast<py::list>(result[2]);
    auto container_file =
        py::cast<std::shared_ptr<caffe2::serialize::PyTorchStreamReader>>(
            result[3]);

    std::vector<at::Storage> storages_c;
    std::vector<at::ScalarType> dtypes_c;
    for (size_t i = 0, N = storages.size(); i < N; ++i) {
      storages_c.push_back(torch::createStorage(storages[i].ptr()));
      dtypes_c.push_back(
          reinterpret_cast<THPDtype*>(dtypes[i].ptr())->scalar_type);
    }
    return PickledObject{
        bytes,
        std::move(storages_c),
        std::move(dtypes_c),
        std::move(container_file)};
  }
  PythonObject unpickle_or_get(int64_t id, const PickledObject& obj) override {
    py::dict objects = interp_->objects;
    py::object id_p = py::cast(id);
    if (objects.contains(id_p)) {
      return wrap(objects[id_p]);
    }

    InitLockAcquire guard(interp_->init_lock_);
    // re-check if something else loaded this before we acquired the
    // init_lock_
    if (objects.contains(id_p)) {
      return wrap(objects[id_p]);
    }

    py::tuple storages(obj.storages_.size());
    for (size_t i = 0, N = obj.storages_.size(); i < N; ++i) {
      py::object new_storage =
          py::reinterpret_steal<py::object>(torch::createPyObject(
              obj.storages_[i], scalarTypeToTypeMeta(obj.types_[i])));
      storages[i] = std::move(new_storage);
    }
    py::object result = interp_->load_storage(
        id, obj.container_file_, py::bytes(obj.data_), storages);
    return wrap(result);
  }
  void unload(int64_t id) override {
    py::dict objects = interp_->objects;
    py::object id_p = py::cast(id);
    if (objects.contains(id_p)) {
      objects.attr("__delitem__")(id_p);
    }
  }

  IValue toIValue(PythonObject obj) const override {
    return torch::jit::toTypeInferredIValue(unwrap(obj));
  }

  PythonObject call(PythonObject obj, at::ArrayRef<PythonObject> args)
      override {
    py::tuple m_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      m_args[i] = unwrap(args[i]);
    }
    return wrap(call(unwrap(obj), m_args));
  }

  PythonObject call(PythonObject obj, at::ArrayRef<IValue> args) override {
    py::tuple m_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      m_args[i] = torch::jit::toPyObject(args[i]);
    }
    return wrap(call(unwrap(obj), m_args));
  }

  PythonObject attr(PythonObject obj, const char* attr) override {
    return wrap(unwrap(obj).attr(attr));
  }

  static py::object call(py::handle object, py::handle args) {
    PyObject* result = PyObject_CallObject(object.ptr(), args.ptr());
    if (!result) {
      throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
  }

  py::handle unwrap(PythonObject obj) const {
    return objects_.at(ID(obj));
  }
  PythonObject wrap(py::object obj) {
    objects_.emplace_back(std::move(obj));
    return PythonObject(this, objects_.size() - 1);
  }
  ~ConcreteInterpreterSessionImpl() override {
    objects_.clear();
  }
  ConcreteInterpreterImpl* interp_;
  ScopedAcquire acquire_;
  std::vector<py::object> objects_;
};

torch::InterpreterSessionImpl* ConcreteInterpreterImpl::acquire_session() {
  return new ConcreteInterpreterSessionImpl(this);
}

extern "C" __attribute__((visibility("default"))) torch::InterpreterImpl*
new_interpreter_impl(void) {
  return new ConcreteInterpreterImpl();
}
