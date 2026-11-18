#include <torch/csrc/utils/python_symnode.h>

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace torch {

namespace {
std::atomic<bool> symnode_cache_enabled{[] {
  const char* value = std::getenv("TORCH_SYMNODE_CACHE");
  return value && std::string_view(value) == "1";
}()};
std::atomic<uint64_t> cache_hits{0}, cache_misses{0}, wrap_hits{0},
    materializations{0}, int_hits{0}, metadata_snapshots{0};

std::shared_ptr<c10::SafePyObject> safe_object(py::object value) {
  return std::make_shared<c10::SafePyObject>(
      value.release().ptr(), getPyInterpreter());
}

py::handle symnode_class() {
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module_::import("torch.fx.experimental.sym_node")
            .attr("SymNode");
      })
      .get_stored();
#else
  static py::handle cls =
      py::object(
          py::module_::import("torch.fx.experimental.sym_node").attr("SymNode"))
          .release();
  return cls;
#endif
}

bool can_cache_now() {
  return symnode_cache_enabled.load(std::memory_order_relaxed) &&
      !c10::impl::TorchDispatchModeTLS::get_mode(
             c10::impl::TorchDispatchModeKey::PROXY) &&
      !c10::impl::tls_local_dispatch_key_set().included_.has(
          c10::DispatchKey::PreDispatch);
}
} // namespace

namespace impl {

struct SymNodeCacheData {
  // The snapshot deliberately excludes ShapeEnv, avoiding a native/Python
  // cycle.
  std::shared_ptr<c10::SafePyObject> fields;
  PyObject* expr = nullptr;
  PyObject* pytype = nullptr;
  int64_t hint = 0;
  bool optimized = false;
  bool constant = false;
  uint64_t int_epoch = UINT64_MAX;
  std::optional<int64_t> int_value;
};

struct SymNodeCacheKey {
  std::string_view op;
  PyObject* lhs;
  PyObject* rhs;
  PyObject* lhs_type;
  PyObject* rhs_type;
  int64_t lhs_hint;
  int64_t rhs_hint;
  bool lhs_optimized;
  bool rhs_optimized;
  bool lhs_constant;
  bool rhs_constant;

  bool operator==(const SymNodeCacheKey& other) const = default;
};

struct SymNodeCacheHash {
  size_t operator()(const SymNodeCacheKey& key) const {
    size_t h = std::hash<std::string_view>{}(key.op);
    auto mix = [&h](size_t x) { h ^= x + 0x9e3779b9 + (h << 6) + (h >> 2); };
    mix(reinterpret_cast<size_t>(key.lhs));
    mix(reinterpret_cast<size_t>(key.rhs));
    mix(reinterpret_cast<size_t>(key.lhs_type));
    mix(reinterpret_cast<size_t>(key.rhs_type));
    mix(std::hash<int64_t>{}(key.lhs_hint));
    mix(std::hash<int64_t>{}(key.rhs_hint));
    mix(key.lhs_optimized | (key.rhs_optimized << 1) | (key.lhs_constant << 2) |
        (key.rhs_constant << 3));
    return h;
  }
};

struct SymNodeCacheEntry {
  // Retaining the operand expressions prevents address reuse in keys.
  std::shared_ptr<SymNodeCacheData> lhs, rhs, result;
};

struct SymNodeCache {
  std::mutex mutex;
  std::atomic<uint64_t> epoch{0};
  std::unordered_map<SymNodeCacheKey, SymNodeCacheEntry, SymNodeCacheHash>
      entries;
  std::unordered_map<int64_t, std::shared_ptr<SymNodeCacheData>> integers;

  void clear() {
    decltype(entries) old_entries;
    decltype(integers) old_integers;
    {
      std::lock_guard<std::mutex> lock(mutex);
      old_entries.swap(entries);
      old_integers.swap(integers);
      epoch.fetch_add(1, std::memory_order_relaxed);
    }
    // SafePyObject destruction can acquire the GIL; never do it under mutex.
  }
};

PythonSymNodeImpl::PythonSymNodeImpl(py::object pyobj)
    : pyobj_(safe_object(std::move(pyobj))), pyobj_ready_(true) {}

bool PythonSymNodeImpl::init_cache_() {
  if (cache_ready_.load(std::memory_order_acquire)) {
    return true;
  }
  py::gil_scoped_acquire acquire;
  std::unique_lock<std::mutex> lock(init_mutex_, std::defer_lock);
  if (!lock.try_lock()) {
    py::gil_scoped_release release;
    lock.lock();
  }
  if (cache_ready_.load(std::memory_order_relaxed)) {
    return true;
  }
  auto obj = getPyObj();
  if (Py_TYPE(obj.ptr()) !=
      reinterpret_cast<PyTypeObject*>(symnode_class().ptr())) {
    return false;
  }
  py::object env = obj.attr("shape_env");
  py::object type = obj.attr("pytype");
  py::object hint = obj.attr("_hint");
  if (env.is_none() ||
      !(type.ptr() == reinterpret_cast<PyObject*>(&PyLong_Type) ||
        type.ptr() == reinterpret_cast<PyObject*>(&PyBool_Type)) ||
      !(PyLong_CheckExact(hint.ptr()) || PyBool_Check(hint.ptr()))) {
    return false;
  }
  int overflow = 0;
  auto hint_value = PyLong_AsLongLongAndOverflow(hint.ptr(), &overflow);
  if (overflow) {
    return false;
  }
  auto native_cache = env.attr("_native_symnode_cache");
  if (native_cache.is_none() ||
      env.attr("_translation_validation_enabled").cast<bool>() ||
      env.attr("should_record_events").cast<bool>() ||
      py::module_::import("torch._logging._internal")
          .attr("GET_DTRACE_STRUCTURED")
          .cast<bool>()) {
    return false;
  }
  cache_ = native_cache.cast<std::shared_ptr<SymNodeCache>>();
  shape_env_ = safe_object(env);
  auto fields =
      py::reinterpret_steal<py::dict>(PyDict_Copy(obj.attr("__dict__").ptr()));
  fields.attr("pop")("shape_env");
  data_ = std::make_shared<SymNodeCacheData>();
  data_->fields = safe_object(fields);
  data_->expr = obj.attr("_expr").ptr();
  data_->pytype = type.ptr();
  data_->hint = hint_value;
  data_->optimized = obj.attr("_optimized_summation").cast<bool>();
  data_->constant = !obj.attr("constant").is_none();
  metadata_snapshots.fetch_add(1, std::memory_order_relaxed);
  cache_ready_.store(true, std::memory_order_release);
  return true;
}

PythonSymNodeImpl::PythonSymNodeImpl(
    std::shared_ptr<SymNodeCache> cache,
    std::shared_ptr<SymNodeCacheData> data,
    std::shared_ptr<c10::SafePyObject> shape_env)
    : cache_(std::move(cache)),
      data_(std::move(data)),
      shape_env_(std::move(shape_env)),
      cache_ready_(true) {}

py::handle PythonSymNodeImpl::getPyObj() const {
  if (!pyobj_ready_.load(std::memory_order_acquire)) {
    // Materialize a distinct Python identity without rerunning symbolic
    // algebra.
    py::gil_scoped_acquire acquire;
    std::unique_lock<std::mutex> lock(init_mutex_, std::defer_lock);
    if (!lock.try_lock()) {
      py::gil_scoped_release release;
      lock.lock();
    }
    if (pyobj_ready_.load(std::memory_order_relaxed)) {
      return py::handle(pyobj_->ptr(getPyInterpreter()));
    }
    auto* type = reinterpret_cast<PyTypeObject*>(symnode_class().ptr());
    auto obj = py::reinterpret_steal<py::object>(type->tp_alloc(type, 0));
    if (!obj) {
      throw py::error_already_set();
    }
    auto fields = py::reinterpret_steal<py::dict>(
        PyDict_Copy(data_->fields->ptr(getPyInterpreter())));
    fields["shape_env"] = py::handle(shape_env_->ptr(getPyInterpreter()));
    // The environment may have advanced since this result was cached.
    fields["_expr_ver"] = -1;
    fields["_expr_range_ver"] = -1;
    fields["_expr_cache"] = py::none();
    if (PyObject_GenericSetDict(obj.ptr(), fields.ptr(), nullptr) < 0) {
      throw py::error_already_set();
    }
    pyobj_ = safe_object(std::move(obj));
    pyobj_ready_.store(true, std::memory_order_release);
    materializations.fetch_add(1, std::memory_order_relaxed);
  }
  return py::handle(pyobj_->ptr(getPyInterpreter()));
}

bool PythonSymNodeImpl::is_int() {
  if (cache_ready_.load(std::memory_order_acquire)) {
    return data_->pytype == reinterpret_cast<PyObject*>(&PyLong_Type);
  }
  py::gil_scoped_acquire acquire;
  return getPyObj().attr("is_int")().is(py::handle(Py_True));
}

bool PythonSymNodeImpl::is_float() {
  if (cache_ready_.load(std::memory_order_acquire)) {
    return false;
  }
  py::gil_scoped_acquire acquire;
  return getPyObj().attr("is_float")().is(py::handle(Py_True));
}

bool PythonSymNodeImpl::is_bool() {
  if (cache_ready_.load(std::memory_order_acquire)) {
    return data_->pytype == reinterpret_cast<PyObject*>(&PyBool_Type);
  }
  py::gil_scoped_acquire acquire;
  return getPyObj().attr("is_bool")().is(py::handle(Py_True));
}

std::optional<int64_t> PythonSymNodeImpl::maybe_as_int() {
  uint64_t epoch = 0;
  // Metadata-only views must not initialize the arithmetic cache.
  bool eligible =
      cache_ready_.load(std::memory_order_acquire) && can_cache_now();
  if (eligible) {
    std::lock_guard<std::mutex> lock(cache_->mutex);
    epoch = cache_->epoch.load(std::memory_order_relaxed);
    if (data_->int_epoch == epoch) {
      int_hits.fetch_add(1, std::memory_order_relaxed);
      return data_->int_value;
    }
  }
  py::gil_scoped_acquire acquire;
  auto result =
      getPyObj().attr("maybe_as_int")().cast<std::optional<int64_t>>();
  if (eligible) {
    std::lock_guard<std::mutex> lock(cache_->mutex);
    if (cache_->epoch.load(std::memory_order_relaxed) == epoch) {
      data_->int_value = result;
      data_->int_epoch = epoch;
    }
  }
  return result;
}

c10::SymNode PythonSymNodeImpl::wrap_int(int64_t num) {
  bool eligible = can_cache_now() && init_cache_();
  if (eligible) {
    std::shared_ptr<SymNodeCacheData> found;
    {
      std::lock_guard<std::mutex> lock(cache_->mutex);
      auto it = cache_->integers.find(num);
      if (it != cache_->integers.end()) {
        found = it->second;
      }
    }
    if (found) {
      wrap_hits.fetch_add(1, std::memory_order_relaxed);
      return c10::make_intrusive<PythonSymNodeImpl>(
          cache_, std::move(found), shape_env_);
    }
  }
  py::gil_scoped_acquire acquire;
  auto result =
      c10::make_intrusive<PythonSymNodeImpl>(getPyObj().attr("wrap_int")(num));
  if (eligible && result->init_cache_()) {
    std::lock_guard<std::mutex> lock(cache_->mutex);
    if (cache_->integers.size() < 1024) {
      cache_->integers.emplace(num, result->data_);
    }
  }
  return result;
}

c10::SymNode PythonSymNodeImpl::dispatch_common_(
    const char* fname,
    const c10::SymNode& other) {
  auto* rhs = dynamic_cast<PythonSymNodeImpl*>(other.get());
  TORCH_CHECK(rhs);
  const std::string_view op(fname);
  bool eligible = can_cache_now() &&
      (op == "add" || op == "sub" || op == "mul" || op == "int_floordiv" ||
       op == "eq" || op == "ne" || op == "lt" || op == "le" || op == "gt" ||
       op == "ge" || op == "sym_min" || op == "sym_max" || op == "sym_and" ||
       op == "sym_or");
  if (eligible) {
    eligible = init_cache_() && rhs->init_cache_() && cache_ == rhs->cache_;
  }
  SymNodeCacheKey key{};
  uint64_t epoch = 0;
  if (eligible) {
    key = {
        op,
        data_->expr,
        rhs->data_->expr,
        data_->pytype,
        rhs->data_->pytype,
        data_->hint,
        rhs->data_->hint,
        data_->optimized,
        rhs->data_->optimized,
        data_->constant,
        rhs->data_->constant};
    std::shared_ptr<SymNodeCacheData> found;
    {
      std::lock_guard<std::mutex> lock(cache_->mutex);
      epoch = cache_->epoch.load(std::memory_order_relaxed);
      auto it = cache_->entries.find(key);
      if (it != cache_->entries.end()) {
        found = it->second.result;
      }
    }
    if (found) {
      cache_hits.fetch_add(1, std::memory_order_relaxed);
      return c10::make_intrusive<PythonSymNodeImpl>(
          cache_, std::move(found), shape_env_);
    }
    cache_misses.fetch_add(1, std::memory_order_relaxed);
  }
  py::gil_scoped_acquire acquire;
  auto result = c10::make_intrusive<PythonSymNodeImpl>(
      getPyObj().attr(fname)(rhs->getPyObj()));
  if (eligible && result->init_cache_()) {
    std::lock_guard<std::mutex> lock(cache_->mutex);
    if (cache_->epoch.load(std::memory_order_relaxed) == epoch &&
        cache_->entries.size() < 4096) {
      cache_->entries.emplace(
          key, SymNodeCacheEntry{data_, rhs->data_, result->data_});
    }
  }
  return result;
}
} // namespace impl

void initSymNodeCache(py::module_& module) {
  py::class_<impl::SymNodeCache, std::shared_ptr<impl::SymNodeCache>>(
      module, "_SymNodeCache")
      .def(py::init<>())
      .def("clear", &impl::SymNodeCache::clear)
      .def(
          "__deepcopy__",
          [](const impl::SymNodeCache&, const py::dict&) {
            return std::make_shared<impl::SymNodeCache>();
          })
      .def(py::pickle(
          [](const impl::SymNodeCache&) { return py::tuple(); },
          [](const py::tuple&) {
            return std::make_shared<impl::SymNodeCache>();
          }));
  module.def("_symnode_cache_enabled", [] {
    return symnode_cache_enabled.load(std::memory_order_relaxed);
  });
  module.def("_set_symnode_cache_enabled", [](bool enabled) {
    symnode_cache_enabled.store(enabled, std::memory_order_relaxed);
  });
  module.def("_symnode_cache_stats", [] {
    py::dict result;
    result["hits"] = cache_hits.load();
    result["misses"] = cache_misses.load();
    result["wrap_int_hits"] = wrap_hits.load();
    result["maybe_as_int_hits"] = int_hits.load();
    result["materializations"] = materializations.load();
    result["metadata_snapshots"] = metadata_snapshots.load();
    return result;
  });
}

py::handle get_symint_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch").attr("SymInt");
      })
      .get_stored();
#else
  static py::handle symint_class =
      py::object(py::module::import("torch").attr("SymInt")).release();
  return symint_class;
#endif
}

py::handle get_symfloat_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch").attr("SymFloat");
      })
      .get_stored();
#else
  static py::handle symfloat_class =
      py::object(py::module::import("torch").attr("SymFloat")).release();
  return symfloat_class;
#endif
}

py::handle get_symbool_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch").attr("SymBool");
      })
      .get_stored();
#else
  static py::handle symbool_class =
      py::object(py::module::import("torch").attr("SymBool")).release();
  return symbool_class;
#endif
}

py::handle get_dynint_class() {
  // NB: leak
#if IS_PYBIND_2_13_PLUS
  PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
      storage;
  return storage
      .call_once_and_store_result([]() -> py::object {
        return py::module::import("torch.fx.experimental.sym_node")
            .attr("DynamicInt");
      })
      .get_stored();
#else
  static py::handle symbool_class =
      py::object(py::module::import("torch.fx.experimental.sym_node")
                     .attr("DynamicInt"))
          .release();
  return symbool_class;
#endif
}

} // namespace torch
