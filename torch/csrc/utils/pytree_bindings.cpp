#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <memory>
#include <stack>

#include "torch/csrc/utils/pytree.h"

namespace py = pybind11;

namespace torch {
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

  static bool is_leaf(const py::handle& x) {
    return get_kind(x) == Kind::Leaf;
  }

  static void flatten_internal(
      py::handle x,
      std::vector<py::object>& leaves,
      TreeSpec& s) {
    const auto kind = get_kind(x);
    switch (kind) {
      case Kind::List: {
        const size_t n = PyList_GET_SIZE(x.ptr());
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
        const size_t n = PyTuple_GET_SIZE(x.ptr());
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
        const size_t size = spec.size();
        py::tuple tuple(size);
        for (size_t i = 0; i < size; ++i) {
          tuple[i] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(tuple);
      }
      case Kind::List: {
        const size_t size = spec.size();
        py::list list(size);
        for (size_t i = 0; i < size; ++i) {
          list[i] = unflatten_internal(spec[i], leaves_it);
        }
        return std::move(list);
      }
      case Kind::Dict: {
        const size_t size = spec.size();
        py::dict dict;
        for (size_t i = 0; i < size; ++i) {
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

  const TreeSpec& spec() const {
    return spec_;
  }

  static PyTree py_from_str(std::string spec) {
    return PyTree(from_str(spec));
  }

  StrTreeSpec py_to_str() const {
    return to_str(spec_);
  }

  // TODO: should it be py::object x argument?
  static std::pair<std::vector<py::object>, std::unique_ptr<PyTree>>
  tree_flatten(py::handle x) {
    std::vector<py::object> leaves{};
    TreeSpec spec{};
    flatten_internal(x, leaves, spec);
    return {std::move(leaves), std::make_unique<PyTree>(std::move(spec))};
  }

  static py::object tree_unflatten(py::iterable leaves, py::object o) {
    return o.cast<PyTree*>()->tree_unflatten(leaves);
  }

  template <typename T>
  py::object tree_unflatten(T leaves) const {
    return unflatten_internal(spec_, leaves.begin());
  }

  bool operator==(const PyTree& rhs) {
    return spec_ == rhs.spec_;
  }
};

inline std::pair<std::vector<py::object>, std::unique_ptr<PyTree>> tree_flatten(
    py::handle x) {
  return PyTree::tree_flatten(x);
}

inline py::object tree_unflatten(py::iterable leaves, py::object o) {
  return PyTree::tree_unflatten(leaves, o);
}

static py::object tree_map(py::function& fn, py::handle x) {
  auto p = tree_flatten(x);
  const auto& leaves = p.first;
  const auto& pytree = p.second;
  std::vector<py::handle> vec;
  for (const py::handle& h : leaves) {
    vec.push_back(fn(h));
  }
  return pytree->tree_unflatten(vec);
}

static std::unique_ptr<PyTree> py_from_str(std::string spec) {
  return std::make_unique<PyTree>(from_str(spec));
}

static py::object broadcast_to_and_flatten(
    py::object x,
    py::object py_tree_spec) {
  auto p = tree_flatten(x);
  const auto& x_leaves = p.first;
  const auto& x_spec = p.second->spec();

  PyTree* tree_spec = py_tree_spec.cast<PyTree*>();

  py::list ret;
  struct StackItem {
    const TreeSpec* tree_spec_node;
    const TreeSpec* x_spec_node;
    const size_t x_leaves_offset;
  };
  std::stack<StackItem> stack;
  stack.push({&tree_spec->spec(), &x_spec, 0u});
  while (!stack.empty()) {
    const auto top = stack.top();
    stack.pop();
    if (top.x_spec_node->isLeaf()) {
      for (size_t i = 0; i < top.tree_spec_node->leaves_num(); ++i) {
        ret.append(x_leaves[top.x_leaves_offset]);
      }
    } else {
      const auto kind = top.tree_spec_node->kind();
      if (kind != top.x_spec_node->kind()) {
        return py::none();
      }
      TORCH_INTERNAL_ASSERT(
          top.tree_spec_node->kind() == top.x_spec_node->kind());
      const size_t child_num = top.tree_spec_node->size();
      if (child_num != top.x_spec_node->size()) {
        return py::none();
      }
      TORCH_INTERNAL_ASSERT(child_num == top.x_spec_node->size());

      size_t x_leaves_offset =
          top.x_leaves_offset + top.x_spec_node->leaves_num();
      auto fn_i = [&](size_t i) {
        x_leaves_offset -= (*top.x_spec_node)[i].leaves_num();
        stack.push(
            {&(*top.tree_spec_node)[i],
             &(*top.x_spec_node)[i],
             x_leaves_offset});
      };
      if (Kind::Dict == kind) {
        for (size_t i = child_num - 1; i < child_num; --i) {
          if (top.tree_spec_node->key(i) != top.x_spec_node->key(i)) {
            return py::none();
          }
          fn_i(i);
        }
      } else {
        for (size_t i = child_num - 1; i < child_num; --i) {
          fn_i(i);
        }
      }
    }
  }
  return std::move(ret);
}

} // namespace

void init_bindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto pytree = m.def_submodule("_pytree");

  pytree.def("tree_flatten", &tree_flatten, py::arg("tree"));
  pytree.def(
      "tree_unflatten", &tree_unflatten, py::arg("leaves"), py::arg("tree"));
  pytree.def("tree_map", &tree_map);
  pytree.def("from_str", &py_from_str);
  pytree.def("broadcast_to_and_flatten", &broadcast_to_and_flatten);

  py::class_<PyTree>(pytree, "TreeSpec")
      .def("from_str", &PyTree::py_from_str)
      .def(
          "tree_unflatten",
          static_cast<py::object (PyTree::*)(py::iterable leaves) const>(
              &PyTree::tree_unflatten))
      .def("__repr__", &PyTree::py_to_str)
      .def("__eq__", &PyTree::operator==)
      .def("to_str", &PyTree::py_to_str);
}

} // namespace pytree
} // namespace torch
