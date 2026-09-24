#include <torch/csrc/symbolic/python_symbolic.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/symbolic/NativeSymNodeImpl.h>
#include <torch/csrc/utils/python_symnode.h>

#include <optional>
#include <string_view>
#include <utility>

// The torch.SymInt / torch.SymBool magic methods of
// torch.fx.experimental.sym_node._make_user_magic, run natively when every
// symbolic operand has a native node. Anything else calls the Python function
// they replace.

namespace torch::symbolic {

namespace {

using BinaryFn = c10::SymNode (c10::SymNodeImpl::*)(const c10::SymNode&);
using UnaryFn = c10::SymNode (c10::SymNodeImpl::*)();

enum class Entry : uint8_t { Unary, Binary, Reflected };

// Set by _seal_glue; process lifetime.
struct Glue {
  PyTypeObject* symint = nullptr;
  PyTypeObject* symbool = nullptr;
  unsigned int symint_tag = 0;
  unsigned int symbool_tag = 0;
  PyTypeObject* native_node = nullptr;
  PyObject* logger = nullptr;
  // logger._cache, which logging clears in place.
  PyObject* log_cache = nullptr;
  PyObject* sym_node_dict = nullptr;
};
Glue glue;

struct Names {
  PyObject* node;
  PyObject* is_constant;
  PyObject* guard_int;
  PyObject* guard_bool;
  PyObject* wrap_node;
  PyObject* isEnabledFor;
  PyObject* debug;
  PyObject* empty_tuple;
};
Names names;

thread_local const char* fallback_reason = nullptr;

struct FallbackScope {
  explicit FallbackScope(const char* reason) : prev(fallback_reason) {
    fallback_reason = reason;
  }
  ~FallbackScope() {
    fallback_reason = prev;
  }
  const char* prev;
};

struct GlueMethod {
  PyObject_HEAD
  vectorcallfunc vectorcall;
  PyObject* original;
  PyObject* method_attr;
  Entry entry;
  // method in bool_becomes_int_magic_methods.
  bool promote;
  // The SymNodeImpl virtual that the node method `method_attr` binds, if any.
  BinaryFn binary_fn;
  UnaryFn unary_fn;
};

BinaryFn binary_virtual(std::string_view name) {
  static const std::pair<std::string_view, BinaryFn> table[] = {
      {"add", &c10::SymNodeImpl::add},
      {"sub", &c10::SymNodeImpl::sub},
      {"mul", &c10::SymNodeImpl::mul},
      {"float_truediv", &c10::SymNodeImpl::float_truediv},
      {"int_truediv", &c10::SymNodeImpl::int_truediv},
      {"float_pow", &c10::SymNodeImpl::float_pow},
      {"pow_by_natural", &c10::SymNodeImpl::pow_by_natural},
      {"int_floordiv", &c10::SymNodeImpl::int_floordiv},
      {"mod", &c10::SymNodeImpl::mod},
      {"eq", &c10::SymNodeImpl::eq},
      {"ne", &c10::SymNodeImpl::ne},
      {"gt", &c10::SymNodeImpl::gt},
      {"lt", &c10::SymNodeImpl::lt},
      {"le", &c10::SymNodeImpl::le},
      {"ge", &c10::SymNodeImpl::ge},
      {"sym_min", &c10::SymNodeImpl::sym_min},
      {"sym_max", &c10::SymNodeImpl::sym_max},
      {"sym_and", &c10::SymNodeImpl::sym_and},
      {"sym_or", &c10::SymNodeImpl::sym_or}};
  for (const auto& [n, fn] : table) {
    if (n == name) {
      return fn;
    }
  }
  return nullptr;
}

UnaryFn unary_virtual(std::string_view name) {
  static const std::pair<std::string_view, UnaryFn> table[] = {
      {"neg", &c10::SymNodeImpl::neg},
      {"sym_not", &c10::SymNodeImpl::sym_not},
      {"ceil", &c10::SymNodeImpl::ceil},
      {"floor", &c10::SymNodeImpl::floor},
      {"sym_float", &c10::SymNodeImpl::sym_float}};
  for (const auto& [n, fn] : table) {
    if (n == name) {
      return fn;
    }
  }
  return nullptr;
}

bool intact() {
  return glue.symint_tag != 0 && glue.symbool_tag != 0 &&
      glue.symint->tp_version_tag == glue.symint_tag &&
      glue.symbool->tp_version_tag == glue.symbool_tag;
}

unsigned int version_tag(PyTypeObject* t) {
#if PY_VERSION_HEX >= 0x030C0000
  PyUnstable_Type_AssignVersionTag(t);
#else
  _PyType_Lookup(t, names.node);
#endif
  return t->tp_version_tag;
}

py::object steal_or_throw(PyObject* obj) {
  if (obj == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(obj);
}

// An operand of a magic method: a SymInt/SymBool with a native node, or an
// exact int (fitting int64) or bool.
struct Operand {
  c10::SymNode node;
  // The Python node while it is `node`.
  py::object node_obj;
  bool symbool = false;
  int64_t value = 0;
  bool is_bool = false;

  py::object py_node() {
    if (!node_obj) {
      node_obj = py::cast(node);
    }
    return node_obj;
  }
};

// Why obj is outside the native domain, or nullptr after filling `op`.
const char* parse(PyObject* obj, Operand& op) {
  PyTypeObject* t = Py_TYPE(obj);
  if (t == &PyBool_Type) {
    op.is_bool = true;
    op.value = obj == Py_True;
    return nullptr;
  }
  if (t == &PyLong_Type) {
    int overflow = 0;
    op.value = PyLong_AsLongLongAndOverflow(obj, &overflow);
    return overflow != 0 ? "int beyond int64" : nullptr;
  }
  if (t != glue.symint && t != glue.symbool) {
    return "operand type";
  }
  PyObject* node = PyObject_GetAttr(obj, names.node);
  if (node == nullptr) {
    PyErr_Clear();
    return "no node";
  }
  op.node_obj = py::reinterpret_steal<py::object>(node);
  if (Py_TYPE(node) != glue.native_node) {
    return "python node";
  }
  op.node = c10::intrusive_ptr<c10::SymNodeImpl>::reclaim_copy(
      py::cast<NativeSymNodeImpl*>(op.node_obj));
  op.symbool = t == glue.symbool;
  return nullptr;
}

// Whether sym_node_log.debug("MAGIC ...") would log.
bool magic_log_on() {
  PyObject* cached = PyDict_GetItemWithError(glue.log_cache, names.debug);
  if (cached == Py_False) {
    return false;
  }
  if (cached == nullptr && PyErr_Occurred()) {
    throw py::error_already_set();
  }
  py::object on = steal_or_throw(
      PyObject_CallMethodOneArg(glue.logger, names.isEnabledFor, names.debug));
  int r = PyObject_IsTrue(on.ptr());
  if (r < 0) {
    throw py::error_already_set();
  }
  return r != 0;
}

// promote of bool_becomes_int_magic_methods.
void promote(Operand& x) {
  if (x.symbool) {
    x.node = x.node->wrap_int(x.node->bool_() ? 1 : 0);
    x.node_obj = py::object();
    x.symbool = false;
  }
  x.is_bool = false;
}

// wrap_node for a native node.
py::object wrap_native(NativeSymNodeImpl& n, py::object node_obj) {
  if (!std::holds_alternative<std::monostate>(n.constant())) {
    return hint_to_py(n.constant());
  }
  if (!node_obj) {
    node_obj = py::cast(c10::SymNode(
        c10::intrusive_ptr<c10::SymNodeImpl>::reclaim_copy(&n)));
  }
  return make_sym_object(
      reinterpret_cast<PyObject*>(n.is_int() ? glue.symint : glue.symbool),
      node_obj);
}

// get_constant(ret) if is_constant(ret) else ret.
py::object constant_result(py::object ret) {
  bool symint = is_symint(ret);
  bool symbool = !symint && is_symbool(ret);
  if (!symint && !symbool && !is_symfloat(ret)) {
    return ret;
  }
  py::object node = ret.attr(names.node);
  py::object c = steal_or_throw(
      PyObject_CallMethodNoArgs(node.ptr(), names.is_constant));
  int r = PyObject_IsTrue(c.ptr());
  if (r < 0) {
    throw py::error_already_set();
  }
  if (r == 0) {
    return ret;
  }
  if (!symint && !symbool) {
    PyErr_SetString(
        PyExc_AssertionError, "expect to be called with constant SymBools");
    throw py::error_already_set();
  }
  return node.attr(symint ? names.guard_int : names.guard_bool)("", 0);
}

// wrap_node(ret), for a binary method followed by the constant check.
py::object wrap_result(py::object ret, bool binary) {
  if (Py_TYPE(ret.ptr()) == glue.native_node) {
    return wrap_native(*py::cast<NativeSymNodeImpl*>(ret), ret);
  }
  PyObject* wrap_node =
      PyDict_GetItemWithError(glue.sym_node_dict, names.wrap_node);
  if (wrap_node == nullptr) {
    if (!PyErr_Occurred()) {
      PyErr_SetObject(PyExc_NameError, names.wrap_node);
    }
    throw py::error_already_set();
  }
  py::object r = steal_or_throw(PyObject_CallOneArg(wrap_node, ret.ptr()));
  return binary ? constant_result(std::move(r)) : r;
}

py::object wrap_result(const c10::SymNode& ret, bool binary) {
  if (auto* n = dynamic_cast<NativeSymNodeImpl*>(ret.get())) {
    return wrap_native(*n, py::object());
  }
  return wrap_result(node_to_py(ret), binary);
}

// binary_magic_impl / rbinary_magic_impl.
std::optional<py::object> binary_fast(
    const GlueMethod& m,
    PyObject* self_obj,
    PyObject* other_obj,
    const char*& reason) {
  Operand self, other;
  if ((reason = parse(self_obj, self)) || (reason = parse(other_obj, other))) {
    return std::nullopt;
  }
  if (!self.node) {
    reason = "operand type";
    return std::nullopt;
  }
  if (m.entry == Entry::Binary && magic_log_on()) {
    reason = "logging";
    return std::nullopt;
  }
  if (m.promote) {
    promote(self);
    promote(other);
  }
  if (!other.node) {
    other.node = other.is_bool ? self.node->wrap_bool(other.value != 0)
                               : self.node->wrap_int(other.value);
  }
  Operand& lhs = m.entry == Entry::Binary ? self : other;
  Operand& rhs = m.entry == Entry::Binary ? other : self;
  if (m.binary_fn != nullptr) {
    return wrap_result(((*lhs.node).*m.binary_fn)(rhs.node), true);
  }
  return wrap_result(
      steal_or_throw(PyObject_CallMethodOneArg(
          lhs.py_node().ptr(), m.method_attr, rhs.py_node().ptr())),
      true);
}

// unary_magic_impl.
std::optional<py::object> unary_fast(
    const GlueMethod& m,
    PyObject* self_obj,
    const char*& reason) {
  Operand self;
  if ((reason = parse(self_obj, self))) {
    return std::nullopt;
  }
  if (!self.node) {
    reason = "operand type";
    return std::nullopt;
  }
  if (m.unary_fn != nullptr) {
    return wrap_result(((*self.node).*m.unary_fn)(), false);
  }
  return wrap_result(
      steal_or_throw(
          PyObject_CallMethodNoArgs(self.py_node().ptr(), m.method_attr)),
      false);
}

PyObject* glue_vectorcall(
    PyObject* callable,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
  const auto& m = *reinterpret_cast<GlueMethod*>(callable);
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  const char* reason = nullptr;
  try {
    std::optional<py::object> r;
    if (kwnames != nullptr) {
      reason = "keywords";
    } else if (nargs != (m.entry == Entry::Unary ? 1 : 2)) {
      reason = "arity";
    } else if (!intact()) {
      reason = "glue modified";
    } else if (m.entry == Entry::Unary) {
      r = unary_fast(m, args[0], reason);
    } else {
      r = binary_fast(m, args[0], args[1], reason);
    }
    if (r) {
      return r->release().ptr();
    }
  } catch (py::builtin_exception& e) {
    e.set_error();
    return nullptr;
  } catch (...) {
    torch::translate_exception_to_python(std::current_exception());
    return nullptr;
  }
  if (m.original == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "native glue method was cleared");
    return nullptr;
  }
  FallbackScope scope(reason);
  return PyObject_Vectorcall(m.original, args, nargsf, kwnames);
}

PyObject* glue_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  PyObject* original = nullptr;
  const char* kind = nullptr;
  PyObject* method_attr = nullptr;
  int promote = 0;
  static const char* kwlist[] = {
      "original", "kind", "method_attr", "promote", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwds,
          "OsUp",
          const_cast<char**>(kwlist),
          &original,
          &kind,
          &method_attr,
          &promote)) {
    return nullptr;
  }
  std::string_view k(kind);
  if (k != "unary" && k != "binary" && k != "rbinary") {
    PyErr_Format(PyExc_ValueError, "unknown glue kind '%s'", kind);
    return nullptr;
  }
  const char* attr = PyUnicode_AsUTF8(method_attr);
  if (attr == nullptr) {
    return nullptr;
  }
  auto* m = PyObject_GC_New(GlueMethod, type);
  if (m == nullptr) {
    return nullptr;
  }
  m->vectorcall = glue_vectorcall;
  m->original = Py_NewRef(original);
  m->method_attr = Py_NewRef(method_attr);
  m->entry = k == "unary" ? Entry::Unary
      : k == "binary"     ? Entry::Binary
                          : Entry::Reflected;
  m->promote = promote != 0;
  m->binary_fn = m->entry == Entry::Unary ? nullptr : binary_virtual(attr);
  m->unary_fn = m->entry == Entry::Unary ? unary_virtual(attr) : nullptr;
  PyObject_GC_Track(m);
  return reinterpret_cast<PyObject*>(m);
}

int glue_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(reinterpret_cast<GlueMethod*>(self)->original);
  return 0;
}

int glue_clear(PyObject* self) {
  Py_CLEAR(reinterpret_cast<GlueMethod*>(self)->original);
  return 0;
}

void glue_dealloc(PyObject* self) {
  PyObject_GC_UnTrack(self);
  glue_clear(self);
  Py_CLEAR(reinterpret_cast<GlueMethod*>(self)->method_attr);
  PyObject_GC_Del(self);
}

PyObject* glue_descr_get(PyObject* self, PyObject* obj, PyObject* /*type*/) {
  if (obj == nullptr || obj == Py_None) {
    return Py_NewRef(self);
  }
  return PyMethod_New(self, obj);
}

PyObject* glue_repr(PyObject* self) {
  return PyUnicode_FromFormat(
      "<native glue %R>", reinterpret_cast<GlueMethod*>(self)->original);
}

PyObject* forward_attr(PyObject* self, void* name) {
  PyObject* original = reinterpret_cast<GlueMethod*>(self)->original;
  if (original == nullptr) {
    PyErr_SetString(PyExc_AttributeError, static_cast<const char*>(name));
    return nullptr;
  }
  return PyObject_GetAttrString(original, static_cast<const char*>(name));
}

PyObject* get_wrapped(PyObject* self, void* /*closure*/) {
  PyObject* original = reinterpret_cast<GlueMethod*>(self)->original;
  return Py_NewRef(original == nullptr ? Py_None : original);
}

PyGetSetDef glue_getset[] = {
    {"__name__", forward_attr, nullptr, nullptr, (void*)"__name__"},
    {"__qualname__", forward_attr, nullptr, nullptr, (void*)"__qualname__"},
    {"__module__", forward_attr, nullptr, nullptr, (void*)"__module__"},
    {"__doc__", forward_attr, nullptr, nullptr, (void*)"__doc__"},
    {"__wrapped__", get_wrapped, nullptr, nullptr, nullptr},
    {nullptr}};

PyTypeObject GlueMethodType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyObject* intern(const char* s) {
  PyObject* r = PyUnicode_InternFromString(s);
  if (r == nullptr) {
    throw py::error_already_set();
  }
  return r;
}

} // namespace

py::object hint_to_py(const Hint& h) {
  return std::visit(
      [](auto v) -> py::object {
        if constexpr (std::is_same_v<decltype(v), std::monostate>) {
          return py::none();
        } else {
          return py::cast(v);
        }
      },
      h);
}

py::object make_sym_object(py::handle cls, py::handle node) {
  py::object r = steal_or_throw(PyBaseObject_Type.tp_new(
      reinterpret_cast<PyTypeObject*>(cls.ptr()), names.empty_tuple, nullptr));
  if (PyObject_SetAttr(r.ptr(), names.node, node.ptr()) < 0) {
    throw py::error_already_set();
  }
  return r;
}

void initGlueBindings(py::module_& sm) {
  names = Names{
      intern("node"),
      intern("is_constant"),
      intern("guard_int"),
      intern("guard_bool"),
      intern("wrap_node"),
      intern("isEnabledFor"),
      PyLong_FromLong(10),
      PyTuple_New(0)};
  if (names.debug == nullptr || names.empty_tuple == nullptr) {
    throw py::error_already_set();
  }

  GlueMethodType.tp_name = "torch._C._symbolic._SymGlueMethod";
  GlueMethodType.tp_basicsize = sizeof(GlueMethod);
  GlueMethodType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
      Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_METHOD_DESCRIPTOR;
  GlueMethodType.tp_doc =
      "A SymInt/SymBool magic method that runs natively for native nodes.";
  GlueMethodType.tp_new = glue_new;
  GlueMethodType.tp_dealloc = glue_dealloc;
  GlueMethodType.tp_traverse = glue_traverse;
  GlueMethodType.tp_clear = glue_clear;
  GlueMethodType.tp_call = PyVectorcall_Call;
  GlueMethodType.tp_vectorcall_offset = offsetof(GlueMethod, vectorcall);
  GlueMethodType.tp_descr_get = glue_descr_get;
  GlueMethodType.tp_repr = glue_repr;
  GlueMethodType.tp_getset = glue_getset;
  if (PyType_Ready(&GlueMethodType) < 0) {
    throw py::error_already_set();
  }
  sm.add_object(
      "_SymGlueMethod",
      py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(&GlueMethodType)));

  // Called after the descriptors are installed: the fast path runs only while
  // SymInt and SymBool are unchanged since.
  sm.def("_seal_glue", [](py::handle logger) {
    glue.symint = reinterpret_cast<PyTypeObject*>(get_symint_class().ptr());
    glue.symbool = reinterpret_cast<PyTypeObject*>(get_symbool_class().ptr());
    glue.native_node = reinterpret_cast<PyTypeObject*>(
        py::type::of<NativeSymNodeImpl>().ptr());
    if (glue.logger == nullptr) {
      glue.logger = Py_NewRef(logger.ptr());
      glue.log_cache = py::object(logger.attr("_cache")).release().ptr();
      glue.sym_node_dict =
          py::object(py::module_::import("torch.fx.experimental.sym_node")
                         .attr("__dict__"))
              .release()
              .ptr();
    }
    glue.symint_tag = version_tag(glue.symint);
    glue.symbool_tag = version_tag(glue.symbool);
  });
  sm.def("_glue_fallback_reason", []() -> py::object {
    if (fallback_reason == nullptr) {
      return py::none();
    }
    return py::str(fallback_reason);
  });
}

} // namespace torch::symbolic
