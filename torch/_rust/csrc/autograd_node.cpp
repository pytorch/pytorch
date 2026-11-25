#include <torch/_rust/csrc/autograd_node.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_variable.h>
#include "torch_rust/src/bindings/ffi.rs.h"

#include <optional>

namespace torch::rust {

namespace {

using torch::autograd::Node;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// The backward of a block the CFG executor ran: Rust's node_apply computes
// the gradients of the block's tensor inputs from those of its outputs.
struct CfgBackward : public Node {
  explicit CfgBackward(::rust::Box<NodeState> state)
      : state_(std::move(state)) {}

  ~CfgBackward() override {
    if (!Py_IsInitialized()) {
      // Too late to release Python objects.
      (void)state_->into_raw();
      return;
    }
    pybind11::gil_scoped_acquire gil;
    state_.reset();
  }

  std::string name() const override {
    return "CfgBackward";
  }

  void release_variables() override {
    pybind11::gil_scoped_acquire gil;
    node_release(**state_);
  }

  variable_list apply(variable_list&& grads) override {
    pybind11::gil_scoped_acquire gil;
    std::vector<std::uintptr_t> objs;
    objs.reserve(grads.size());
    for (const auto i : c10::irange(grads.size())) {
      at::Tensor g = grads[i];
      // An output that got no gradient: zeros, as custom Functions
      // materialize them by default.
      if (!g.defined() && !input_metadata(i).was_default_constructed()) {
        g = input_metadata(i).zeros_like();
      }
      PyObject* o = g.defined() ? THPVariable_Wrap(g) : Py_NewRef(Py_None);
      if (!o) {
        for (auto p : objs) {
          Py_DECREF(reinterpret_cast<PyObject*>(p));
        }
        throw_python_error();
      }
      objs.push_back(reinterpret_cast<std::uintptr_t>(o));
    }
    ::rust::Vec<std::uintptr_t> out;
    bool ok = node_apply(
        **state_,
        ::rust::Slice<const std::uintptr_t>(objs.data(), objs.size()),
        out);
    for (auto p : objs) {
      Py_DECREF(reinterpret_cast<PyObject*>(p));
    }
    variable_list result(num_outputs());
    for (const auto i : c10::irange(out.size())) {
      auto* o = reinterpret_cast<PyObject*>(out[i]);
      if (ok && i < result.size() && o != Py_None) {
        result[i] = THPVariable_Unpack(o);
      }
      Py_DECREF(o);
    }
    if (!ok) {
      throw_python_error();
    }
    return result;
  }

  std::optional<::rust::Box<NodeState>> state_;
};

} // namespace

bool make_cfg_node(
    ::rust::Box<NodeState> state,
    ::rust::Slice<const std::uintptr_t> inputs,
    ::rust::Slice<const std::uintptr_t> outputs,
    ::rust::Vec<std::uintptr_t>& wrapped) noexcept {
  try {
    variable_list ins;
    ins.reserve(inputs.size());
    for (auto p : inputs) {
      ins.push_back(THPVariable_Unpack(reinterpret_cast<PyObject*>(p)));
    }
    auto node = c10::make_intrusive<CfgBackward>(std::move(state));
    node->set_next_edges(torch::autograd::collect_next_edges(ins));
    std::vector<std::optional<Variable>> raw;
    raw.reserve(outputs.size());
    for (auto p : outputs) {
      raw.emplace_back(THPVariable_Unpack(reinterpret_cast<PyObject*>(p)));
    }
    auto jvp = [](const variable_list&, const variable_list&) -> variable_list {
      TORCH_CHECK(false, "the CFG executor doesn't support forward-mode AD");
    };
    auto view_as_self = [](const at::Tensor& x) { return x.view_as(x); };
    c10::intrusive_ptr<Node> attached;
    auto outs = torch::autograd::_wrap_outputs(
        ins, {}, {}, raw, node, jvp, {}, view_as_self, false, attached);
    for (auto& o : outs) {
      PyObject* w = THPVariable_Wrap(*o);
      if (!w) {
        return false;
      }
      wrapped.push_back(reinterpret_cast<std::uintptr_t>(w));
    }
    return true;
  } catch (python_error& e) {
    e.restore();
  } catch (const c10::Error& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what_without_backtrace());
  } catch (const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  return false;
}

} // namespace torch::rust
