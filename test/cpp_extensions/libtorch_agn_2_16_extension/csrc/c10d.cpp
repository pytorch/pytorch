#include <Python.h>

#include <torch/csrc/stable/c10d.h>
#include <torch/csrc/stable/pyobject.h>
#include <torch/headeronly/util/Exception.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using torch::stable::Tensor;
using torch::stable::c10d::ProcessGroup;
using torch::stable::c10d::ReduceOp;
using torch::stable::c10d::Work;

namespace {

class ReleaseGIL {
 public:
  explicit ReleaseGIL(bool release = true)
      : state_(release ? PyEval_SaveThread() : nullptr) {}
  ~ReleaseGIL() {
    if (state_ != nullptr) {
      PyEval_RestoreThread(state_);
    }
  }

 private:
  PyThreadState* state_;
};

template <typename F>
PyObject* checked(F&& fn) {
  try {
    return fn();
  } catch (const std::exception& e) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
    return nullptr;
  }
}

template <typename T>
struct Capsule {
  static const char* name;

  static std::optional<T>& get(PyObject* obj) {
    auto* value = static_cast<std::optional<T>*>(PyCapsule_GetPointer(obj, name));
    STD_TORCH_CHECK(value != nullptr, "invalid stable c10d capsule");
    return *value;
  }

  static PyObject* wrap(T value) {
    auto owner = std::make_unique<std::optional<T>>(std::move(value));
    auto* result = PyCapsule_New(owner.get(), name, [](PyObject* obj) {
      delete static_cast<std::optional<T>*>(PyCapsule_GetPointer(obj, name));
    });
    if (result != nullptr) {
      owner.release();
    }
    return result;
  }

  static PyObject* copy(PyObject*, PyObject* obj) {
    return checked([&] { return wrap(get(obj).value()); });
  }

  static PyObject* close(PyObject*, PyObject* args) {
    PyObject* obj = nullptr;
    int gil_held = 0;
    if (!PyArg_ParseTuple(args, "O|p", &obj, &gil_held)) {
      return nullptr;
    }
    return checked([&] {
      auto& value = get(obj);
      {
        ReleaseGIL release(!gil_held);
        value.reset();
      }
      Py_RETURN_NONE;
    });
  }
};

template <>
const char* Capsule<ProcessGroup>::name = "stable_c10d.ProcessGroup";
template <>
const char* Capsule<Work>::name = "stable_c10d.Work";

std::vector<Tensor> tensors_from_pylist(PyObject* obj) {
  const auto size = PyList_Size(obj);
  STD_TORCH_CHECK(size >= 0, "expected a list of tensors");
  std::vector<Tensor> tensors;
  tensors.reserve(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    tensors.push_back(torch::stable::tensor_from_pyobject(PyList_GetItem(obj, i)));
  }
  return tensors;
}

PyObject* process_group(PyObject*, PyObject* obj) {
  return checked([&] {
    return Capsule<ProcessGroup>::wrap(ProcessGroup::from_pyobject(obj));
  });
}

PyObject* group_info(PyObject*, PyObject* obj) {
  return checked([&] {
    auto& group = Capsule<ProcessGroup>::get(obj).value();
    long long rank = 0;
    long long size = 0;
    std::string backend;
    {
      ReleaseGIL release;
      rank = group.rank();
      size = group.size();
      backend = group.backend();
    }
    return Py_BuildValue("LLs", rank, size, backend.c_str());
  });
}

template <bool Coalesced>
PyObject* allreduce(PyObject*, PyObject* args) {
  PyObject* obj = nullptr;
  PyObject* inputs = nullptr;
  int reduction = 0;
  if (!PyArg_ParseTuple(args, "OO|i", &obj, &inputs, &reduction)) {
    return nullptr;
  }
  return checked([&] {
    auto& group = Capsule<ProcessGroup>::get(obj).value();
    auto tensors = tensors_from_pylist(inputs);
    auto work = [&] {
      ReleaseGIL release;
      if constexpr (Coalesced) {
        return group.allreduce_coalesced(tensors, static_cast<ReduceOp>(reduction));
      } else {
        return group.allreduce(tensors, static_cast<ReduceOp>(reduction));
      }
    }();
    return Capsule<Work>::wrap(std::move(work));
  });
}

PyObject* broadcast(PyObject*, PyObject* args) {
  PyObject* obj = nullptr;
  PyObject* inputs = nullptr;
  long long root_rank = 0;
  long long root_tensor = 0;
  if (!PyArg_ParseTuple(args, "OOL|L", &obj, &inputs, &root_rank, &root_tensor)) {
    return nullptr;
  }
  return checked([&] {
    auto& group = Capsule<ProcessGroup>::get(obj).value();
    auto tensors = tensors_from_pylist(inputs);
    auto work = [&] {
      ReleaseGIL release;
      return group.broadcast(tensors, root_rank, root_tensor);
    }();
    return Capsule<Work>::wrap(std::move(work));
  });
}

PyObject* allgather(PyObject*, PyObject* args) {
  PyObject* obj = nullptr;
  PyObject* input = nullptr;
  PyObject* outputs = nullptr;
  if (!PyArg_ParseTuple(args, "OOO", &obj, &input, &outputs)) {
    return nullptr;
  }
  return checked([&] {
    auto& group = Capsule<ProcessGroup>::get(obj).value();
    auto tensor = torch::stable::tensor_from_pyobject(input);
    auto tensors = tensors_from_pylist(outputs);
    auto work = [&] {
      ReleaseGIL release;
      return group.allgather(tensor, tensors);
    }();
    return Capsule<Work>::wrap(std::move(work));
  });
}

PyObject* barrier(PyObject*, PyObject* obj) {
  return checked([&] {
    auto& group = Capsule<ProcessGroup>::get(obj).value();
    auto work = [&] {
      ReleaseGIL release;
      return group.barrier();
    }();
    return Capsule<Work>::wrap(std::move(work));
  });
}

PyObject* wait(PyObject*, PyObject* args) {
  PyObject* obj = nullptr;
  long long timeout_ms = 0;
  int gil_held = 0;
  if (!PyArg_ParseTuple(args, "O|Lp", &obj, &timeout_ms, &gil_held)) {
    return nullptr;
  }
  return checked([&] {
    auto& work = Capsule<Work>::get(obj).value();
    bool result = false;
    {
      ReleaseGIL release(!gil_held);
      result = work.wait(timeout_ms);
    }
    return PyBool_FromLong(result);
  });
}

PyObject* is_completed(PyObject*, PyObject* obj) {
  return checked([&] {
    auto& work = Capsule<Work>::get(obj).value();
    bool result = false;
    {
      ReleaseGIL release;
      result = work.is_completed();
    }
    return PyBool_FromLong(result);
  });
}

PyMethodDef methods[] = {
    {"process_group", process_group, METH_O, nullptr},
    {"group_info", group_info, METH_O, nullptr},
    {"copy_group", Capsule<ProcessGroup>::copy, METH_O, nullptr},
    {"close_group", Capsule<ProcessGroup>::close, METH_VARARGS, nullptr},
    {"allreduce", allreduce<false>, METH_VARARGS, nullptr},
    {"allreduce_coalesced", allreduce<true>, METH_VARARGS, nullptr},
    {"broadcast", broadcast, METH_VARARGS, nullptr},
    {"allgather", allgather, METH_VARARGS, nullptr},
    {"barrier", barrier, METH_O, nullptr},
    {"wait", wait, METH_VARARGS, nullptr},
    {"is_completed", is_completed, METH_O, nullptr},
    {"copy_work", Capsule<Work>::copy, METH_O, nullptr},
    {"close_work", Capsule<Work>::close, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_c10d",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

} // namespace

PyMODINIT_FUNC PyInit__c10d() {
  return PyModule_Create(&module);
}
