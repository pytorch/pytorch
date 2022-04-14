#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <memory>

#include "executorch/pytree/pytree.h"

namespace py = pybind11;

namespace torch {
namespace executor {
namespace pytree {

namespace {
class PyTypeRegistry {
 public:
  struct PyTypeReg {
    Kind kind;
    explicit PyTypeReg(Kind k) : kind(k) {}
  };
  static const PyTypeReg* get(py::handle pytype) {
    auto* registry = instance();
    auto it = registry->regs_.find(py::cast<py::object>(pytype));
    return it == registry->regs_.end() ? nullptr : it->second.get();
  }

 private:
  static PyTypeRegistry* instance() {
    static auto* registry_instance = []() -> PyTypeRegistry* {
      auto* registry = new PyTypeRegistry;

      auto add_pytype_reg = [&](PyTypeObject* pytype_obj, Kind kind) {
        py::object pytype = py::reinterpret_borrow<py::object>(
            reinterpret_cast<PyObject*>(pytype_obj));
        registry->regs_.emplace(pytype, std::make_unique<PyTypeReg>(kind));
      };

      add_pytype_reg(&PyTuple_Type, Kind::Tuple);
      add_pytype_reg(&PyList_Type, Kind::List);
      add_pytype_reg(&PyDict_Type, Kind::Dict);

      return registry;
    }();

    return registry_instance;
  }

  struct PyTypeHash {
    using is_transparent = void;
    size_t operator()(const py::object& t) const {
      return PyObject_Hash(t.ptr());
    }
  };

  struct PyTypeEq {
    using is_transparent = void;
    bool operator()(const py::object& a, const py::object& b) const {
      return a.ptr() == b.ptr();
    }
  };

  std::unordered_map<
      py::object,
      std::unique_ptr<PyTypeReg>,
      PyTypeHash,
      PyTypeEq>
      regs_;
};

class PyTree {
  TreeSpec spec_;

  static Kind get_kind(const py::handle& x) {
    const auto* reg = PyTypeRegistry::get(x.get_type());
    if (reg) {
      return reg->kind;
    }
    return Kind::Leaf;
  }

  static void
  flatten_internal(py::handle x, std::vector<py::object>& leaves, TreeSpec& s) {
    const auto kind = get_kind(x);
    switch (kind) {
      case Kind::List: {
        const auto n = PyList_GET_SIZE(x.ptr());
        s = TreeSpec(Kind::List, n);
        size_t leaves_num = 0;
        for (size_t i = 0; i < n; ++i) {
          TreeSpec& child = s.handle->items[i];
          flatten_internal(PyList_GET_ITEM(x.ptr(), i), leaves, child);
          leaves_num += child.leaves_num();
        }
        s.set_leaves_num(leaves_num);
        break;
      }
      case Kind::Tuple: {
        const auto n = PyTuple_GET_SIZE(x.ptr());
        s = TreeSpec(Kind::Tuple, n);
        size_t leaves_num = 0;
        for (size_t i = 0; i < n; ++i) {
          TreeSpec& child = s.handle->items[i];
          flatten_internal(PyTuple_GET_ITEM(x.ptr(), i), leaves, child);
          leaves_num += child.leaves_num();
        }
        s.set_leaves_num(leaves_num);
        break;
      }
      case Kind::Dict: {
        py::dict dict = py::reinterpret_borrow<py::dict>(x);
        py::list keys =
            py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
        const auto n = PyList_GET_SIZE(keys.ptr());
        s = TreeSpec(Kind::Dict, n);
        size_t leaves_num = 0;
        size_t i = 0;
        for (py::handle key : keys) {
          if (py::isinstance<py::str>(key)) {
            s.handle->dict.keys[i] = py::cast<std::string>(key);
          } else if (py::isinstance<py::int_>(key)) {
            s.handle->dict.keys[i] = py::cast<int32_t>(key);
          } else {
            assert(false);
          }

          TreeSpec& child = s.handle->items[i];
          flatten_internal(dict[key], leaves, child);
          leaves_num += child.leaves_num();
          i++;
        }
        s.set_leaves_num(leaves_num);
        break;
      }
      case Kind::Leaf: {
        s = TreeSpec(Kind::Leaf);
        leaves.push_back(py::reinterpret_borrow<py::object>(x));
        s.set_leaves_num(1u);
        break;
      }
      case Kind::None:
        assert(false);
    }
  }

  template <typename T>
  py::object unflatten_internal(const TreeSpec& spec, T&& leaves_it) const {
    switch (spec.kind()) {
      case Kind::Tuple: {
        const auto size = spec.size();
        py::tuple tuple(size);
        for (int i = 0; i < size; ++i) {
          tuple[i] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(tuple);
      }
      case Kind::List: {
        const auto size = spec.size();
        py::list list(size);
        for (int i = 0; i < size; ++i) {
          list[i] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(list);
      }
      case Kind::Dict: {
        const auto size = spec.size();
        py::dict dict;
        for (int i = 0; i < size; ++i) {
          auto& key = spec.key(i);
          auto py_key = [&key]() -> py::handle {
            switch (key.kind()) {
              case Key::Kind::Int:
                return py::cast(key.as_int()).release();
              case Key::Kind::Str:
                return py::cast(key.as_str()).release();
              case Key::Kind::None:
                assert(false);
            }
            assert(false);
            return py::none();
          }();
          dict[py_key] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(dict);
      }
      case Kind::Leaf: {
        py::object o =
            py::reinterpret_borrow<py::object>(*std::forward<T>(leaves_it));
        leaves_it++;
        return o;
      }
      case Kind::None: {
        return py::none();
      }
    }
    assert(false);
  }

 public:
  explicit PyTree(TreeSpec spec) : spec_(std::move(spec)) {}
  static PyTree py_from_str(std::string spec) {
    return PyTree(from_str(spec));
  }

  StrTreeSpec py_to_str() {
    return to_str(spec_);
  }

  static std::pair<std::vector<py::object>, std::unique_ptr<PyTree>> flatten(
      py::handle x) {
    std::vector<py::object> leaves{};
    TreeSpec spec{};
    flatten_internal(x, leaves, spec);
    return {std::move(leaves), std::make_unique<PyTree>(std::move(spec))};
  }

  py::object unflatten(py::iterable leaves) const {
    return unflatten_internal(spec_, leaves.begin());
  }
};
} // namespace

PYBIND11_MODULE(pybindings, m) {
  m.def("flatten", &PyTree::flatten, py::arg("tree"));
  py::class_<PyTree>(m, "PyTree")
      .def("from_str", &PyTree::py_from_str)
      .def(
          "unflatten",
          static_cast<py::object (PyTree::*)(py::iterable leaves) const>(
              &PyTree::unflatten))
      .def("to_str", &PyTree::py_to_str);
}

} // namespace pytree
} // namespace executor
} // namespace torch
