#include <torch/csrc/deploy/interpreter/interpreter_impl.h>

#include <dlfcn.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/deploy/ArrayRef.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <thread>

#include <fmt/format.h>
#include <torch/csrc/deploy/interpreter/builtin_registry.h>

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

const char* start = R"PYTHON(
import _ssl # must come before _hashlib otherwise ssl's locks will be set to a Python that might no longer exist...
import sys
import importlib.abc
import linecache

class RegisterModuleImporter(importlib.abc.InspectLoader):
    def __init__(self, find_module_source):
        self.find_module_source = find_module_source

    def create_module(self, spec):
        return None

    def get_source(self, name):
        return self.find_module_source(name)

    def exec_module(self, module):
        filename = f"_deploy_internal.{module.__name__}"
        linecache.lazycache(filename, module.__dict__)
        code = compile(self.get_source(module.__name__), filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    def find_spec(self, fullname, path, target=None):
        r = self.find_module_source(fullname)
        if r is not None:
            return importlib.util.spec_from_loader(fullname, self)
        return None

# print("exec_prefix:", sys.base_exec_prefix)
# print("_base_executable:", sys._base_executable)
# print("base_prefix:", sys.base_prefix)
# print("exec_prefix:", sys.exec_prefix)
# print("executable:", sys.executable)
# print("path:", sys.path)
# print("prefix:", sys.prefix)
import torch # has to be done serially otherwise things will segfault
try:
  import torch.version # for some reason torch doesn't import this and cuda fails?
except ModuleNotFoundError:
  # fbcode built doesn't have version.py, workaround by faking its info...
  from types import ModuleType
  _v = torch.version = sys.modules['torch.version'] = ModuleType('torch.version')
  _v.__version__ = '1.8.0a0+fake'
  _v.debug = False
  _v.cuda = '10.1'
  _v.git_version = 'fake'
  _v.hip = None


if torch.cuda.is_available():
  torch.zeros(1).cuda() # force cuda init...
import warnings
warnings.simplefilter("ignore")
)PYTHON";

extern "C" __attribute__((__weak__)) PyObject* PyInit_tensorrt(void);
extern "C"
    __attribute__((__weak__)) struct _frozen _PyImport_FrozenModules_tensorrt[];

using torch::deploy::BuiltinRegistry;
// TODO(shunting) move this to the tensorrt code
REGISTER_TORCH_DEPLOY_BUILTIN(
    tensorrt,
    _PyImport_FrozenModules_tensorrt,
    "tensorrt.tensorrt",
    PyInit_tensorrt);

static py::object global_impl(const char* module, const char* name) {
  return py::module::import(module).attr(name);
}

using at::IValue;
using torch::deploy::Obj;
using torch::deploy::PickledObject;

// Ensure GIL is held while this object is live,
// note: we are not use py::gil_scoped_acquire here because
// InitLockAcquire used below has to temporarily release the GIL
// within this scope to ensure locking order.  Having the source
// for these objects together makes it easier to see what is happening.
struct ScopedAcquire {
  ScopedAcquire() {
    gstate = PyGILState_Ensure();
  }
  ~ScopedAcquire() {
    PyGILState_Release(gstate);
  }
  PyGILState_STATE gstate;
};

struct InitLockAcquire {
  InitLockAcquire(std::mutex& init_lock) : init_lock_(init_lock) {
    // to avoid deadlock, we need to ensure a consistent lock order:
    // init_lock -> GIL. Otherwise, the GIL can be released by the python
    // interpreter during initalization tasks, and then re-acquired. If another
    // thread grabs the GIL to do non-initialization tasks, then it might start
    // initializing (GIL -> init_lock). To avoid this, release the GIL before
    // trying to get the init_lock and then reacquire it afterward.
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyThreadState* _save;
    _save = PyEval_SaveThread();
    init_lock.lock();
    PyEval_RestoreThread(_save);
  }
  ~InitLockAcquire() {
    init_lock_.unlock();
  }

 private:
  std::mutex& init_lock_;
};

struct __attribute__((visibility("hidden"))) ConcreteInterpreterImpl
    : public torch::deploy::InterpreterImpl {
  explicit ConcreteInterpreterImpl(
      const std::vector<std::string>& extra_python_paths) {
    BuiltinRegistry::runPreInitialization();
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
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    wchar_t* module_search_paths[0] = {};
    status = PyConfig_SetWideStringList(
        &config, &config.module_search_paths, 0, module_search_paths);

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    TORCH_INTERNAL_ASSERT(!PyStatus_Exception(status))
#ifdef FBCODE_CAFFE2
    auto sys_path = global_impl("sys", "path");
    for (const auto& entry : extra_python_paths) {
      sys_path.attr("insert")(0, entry);
    }
#endif
    BuiltinRegistry::runPostInitialization();

    int r = PyRun_SimpleString(start);
    TORCH_INTERNAL_ASSERT(r == 0);

    // we cache these so we don't have to repeat the conversion of strings into
    // Python and hash table lookups to get to these object
    saveStorage = global_impl("torch._deploy", "_save_storages");
    loadStorage = global_impl("torch._deploy", "_load_storages");
    getPackage = global_impl("torch._deploy", "_get_package");
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
    saveStorage.release();
    loadStorage.release();
    getPackage.release();
    if (Py_FinalizeEx() != 0) {
      exit(1); // can't use TORCH_INTERNAL_ASSERT because we are in a
               // non-throwing destructor.
    }
  }

  void setFindModule(
      std::function<multipy::optional<std::string>(const std::string&)>
          find_module) override {
    std::function<py::object(const std::string&)> wrapped_find_module =
        [=](const std::string& name) -> py::object {
      auto r = find_module(name);
      return r ? py::cast(*r) : py::none();
    };
    py::object register_module_importer =
        py::module::import("__main__")
            .attr("RegisterModuleImporter")(wrapped_find_module);
    py::module::import("sys")
        .attr("meta_path")
        .attr("append")(register_module_importer);
  }

  torch::deploy::InterpreterSessionImpl* acquireSession() override;
  py::object saveStorage;
  py::object loadStorage;
  py::object getPackage;
  py::dict objects;
  std::mutex init_lock_;
};

struct __attribute__((visibility("hidden"))) ConcreteInterpreterSessionImpl
    : public torch::deploy::InterpreterSessionImpl {
  ConcreteInterpreterSessionImpl(ConcreteInterpreterImpl* interp)
      : interp_(interp) {}
  Obj global(const char* module, const char* name) override {
    return wrap(global_impl(module, name));
  }

  Obj fromIValue(IValue value) override {
    return wrap(torch::jit::toPyObject(value));
  }
  Obj createOrGetPackageImporterFromContainerFile(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          containerFile_) override {
    InitLockAcquire guard(interp_->init_lock_);
    return wrap(interp_->getPackage(containerFile_));
  }

  PickledObject pickle(Obj container, Obj obj) override {
    py::tuple result = interp_->saveStorage(unwrap(container), unwrap(obj));
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
  Obj unpickleOrGet(int64_t id, const PickledObject& obj) override {
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
      py::object new_storage = py::reinterpret_steal<py::object>(
          torch::createPyObject(obj.storages_[i]));
      storages[i] = std::move(new_storage);
    }
    py::tuple dtypes(obj.types_.size());
    for (size_t i = 0, N = obj.types_.size(); i < N; ++i) {
      auto dtype = (PyObject*)torch::getTHPDtype(obj.types_[i]);
      Py_INCREF(dtype);
      dtypes[i] = dtype;
    }
    py::object result = interp_->loadStorage(
        id, obj.containerFile_, py::bytes(obj.data_), storages, dtypes);
    return wrap(result);
  }
  void unload(int64_t id) override {
    py::dict objects = interp_->objects;
    py::object id_p = py::cast(id);
    if (objects.contains(id_p)) {
      objects.attr("__delitem__")(id_p);
    }
  }

  IValue toIValue(Obj obj) const override {
    return torch::jit::toTypeInferredIValue(unwrap(obj));
  }

  Obj call(Obj obj, multipy::ArrayRef<Obj> args) override {
    py::tuple m_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      m_args[i] = unwrap(args[i]);
    }
    return wrap(call(unwrap(obj), m_args));
  }

  Obj call(Obj obj, multipy::ArrayRef<IValue> args) override {
    py::tuple m_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      m_args[i] = torch::jit::toPyObject(args[i]);
    }
    return wrap(call(unwrap(obj), m_args));
  }

  Obj callKwargs(
      Obj obj,
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) override {
    py::tuple py_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      py_args[i] = torch::jit::toPyObject(args[i]);
    }

    py::dict py_kwargs;
    for (auto kv : kwargs) {
      py_kwargs[py::cast(std::get<0>(kv))] =
          torch::jit::toPyObject(std::get<1>(kv));
    }
    return wrap(call(unwrap(obj), py_args, py_kwargs));
  }

  Obj callKwargs(Obj obj, std::unordered_map<std::string, c10::IValue> kwargs)
      override {
    std::vector<at::IValue> args;
    return callKwargs(obj, args, kwargs);
  }

  bool hasattr(Obj obj, const char* attr) override {
    return py::hasattr(unwrap(obj), attr);
  }

  Obj attr(Obj obj, const char* attr) override {
    return wrap(unwrap(obj).attr(attr));
  }

  static py::object call(
      py::handle object,
      py::handle args,
      py::handle kwargs = nullptr) {
    PyObject* result = PyObject_Call(object.ptr(), args.ptr(), kwargs.ptr());
    if (!result) {
      throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
  }

  py::handle unwrap(Obj obj) const {
    return objects_.at(ID(obj));
  }

  Obj wrap(py::object obj) {
    objects_.emplace_back(std::move(obj));
    return Obj(this, objects_.size() - 1);
  }

  ~ConcreteInterpreterSessionImpl() override {
    objects_.clear();
  }
  ConcreteInterpreterImpl* interp_;
  ScopedAcquire acquire_;
  std::vector<py::object> objects_;
};

torch::deploy::InterpreterSessionImpl* ConcreteInterpreterImpl::
    acquireSession() {
  return new ConcreteInterpreterSessionImpl(this);
}

extern "C" __attribute__((visibility("default")))
torch::deploy::InterpreterImpl*
newInterpreterImpl(const std::vector<std::string>& extra_python_paths) {
  return new ConcreteInterpreterImpl(extra_python_paths);
}
