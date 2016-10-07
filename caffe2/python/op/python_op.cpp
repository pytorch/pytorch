#include <unordered_map>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/arrayobject.h>

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace caffe2 {

namespace py = pybind11;

namespace {

using FuncRegistery = std::unordered_map<std::string, py::object>;
static FuncRegistery& gRegistery() {
  // Always leak the objects registered here.
  static FuncRegistery* r = new FuncRegistery();
  return *r;
}

py::object& getOpFunc(const std::string& token) {
  return gRegistery()[token];
}

py::object& getGradientFunc(const std::string& token) {
  return gRegistery()[token + "_gradient"];
}

}

class PythonOpBase : public Operator<CPUContext> {
 public:
  using Operator::Operator;

  bool RunOnDevice() final {
    std::vector<TensorCPU*> inputs;
    inputs.reserve(InputSize());
    for (auto i = 0; i < InputSize(); ++i) {
      inputs.push_back(const_cast<TensorCPU*>(&Input(i)));
    }
    std::vector<TensorCPU*> outputs;
    outputs.reserve(OutputSize());
    for (auto i = 0; i < OutputSize(); ++i) {
      outputs.push_back(Output(i));
    }
    auto& pyFunc = getFunc();
    {
      // Acquire GIL for call to Python runtime.
      py::gil_scoped_acquire g;
      try {
        pyFunc(inputs, outputs);
      } catch (const py::error_already_set& e) {
        LOG(ERROR) << "Exception encountered running PythonOp function: "
                   << e.what() << "\nTraceback: ";
        PyObject *type = nullptr, *value = nullptr, *trace = nullptr;
        PyErr_Fetch(&type, &value, &trace);
        PyTracebackObject* traceback =
            reinterpret_cast<PyTracebackObject*>(trace);
        vector<PyTracebackObject*> trace_vec;
        while (traceback) {
          trace_vec.push_back(traceback);
          traceback = traceback->tb_next;
        }
        for (int i = trace_vec.size() - 1; i >= 0; --i) {
          int line = trace_vec[i]->tb_lineno;
          const char* filename =
              PyString_AsString(trace_vec[i]->tb_frame->f_code->co_filename);
          const char* funcname =
              PyString_AsString(trace_vec[i]->tb_frame->f_code->co_name);
          LOG(ERROR) << "    # " << trace_vec.size() - i - 1 << "  " << filename
                     << " (" << line << "): " << funcname;
        }
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(trace);
        return false;
      }
    }
    return true;
  }

 private:
  virtual py::object& getFunc() = 0;
};

class PythonOp final : public PythonOpBase {
 public:
  using PythonOpBase::PythonOpBase;

 private:
  py::object& getFunc() override {
    const std::string& token =
        OperatorBase::GetSingleArgument<std::string>("token", "");
    return getOpFunc(token);
  }
};

class PythonGradientOp final : public PythonOpBase {
 public:
  using PythonOpBase::PythonOpBase;

 private:
  py::object& getFunc() override {
    const std::string& token =
        OperatorBase::GetSingleArgument<std::string>("token", "");
    return getGradientFunc(token);
  }
};

PYBIND11_PLUGIN(python_ops_python) {
  py::module m("python_ops_python", "pybind11 interface to operators");

  py::class_<TensorCPU>(m, "TensorCPU")
      .def_property_readonly(
          "data",
          [](TensorCPU* t) -> py::object {
            CAFFE_ENFORCE(t->size() > 0);
            std::vector<npy_intp> npy_dims;
            for (const auto dim : t->dims()) {
              npy_dims.push_back(dim);
            }
            PyObject* array = PyArray_SimpleNewFromData(
                t->ndim(),
                npy_dims.data(),
                NPY_FLOAT32,
                t->mutable_data<float>());
            return py::object(array, /* borrowed= */ false);
          })
      .def_property_readonly(
          "_shape", [](const TensorCPU& t) { return t.dims(); })
      .def("_reshape", [](TensorCPU* t, std::vector<TIndex> dims) {
        t->Resize(dims);
      });

  m.def("register", [](py::object func) {
    CAFFE_ENFORCE(func != py::none());
    const std::string name = func.attr("__name__").cast<std::string>();
    // Unique name since registry is never cleared.
    const std::string token = name + std::to_string(gRegistery().size());
    CAFFE_ENFORCE(gRegistery().find(name) == gRegistery().end());
    gRegistery()[token] = func;
    return token;
  });

  m.def("register_gradient", [](const std::string& token, py::object func) {
    CAFFE_ENFORCE(func != py::none());
    CAFFE_ENFORCE(gRegistery().find(token) != gRegistery().end());
    gRegistery()[token + "_gradient"] = func;
  });
  ([]() {
    // This is a workaround so we can deal with numpy's import_array behavior.
    // Despite the fact that you may think import_array() is a function call,
    // it is defined as a macro (as of 1.10).
    import_array();
  })();
  return m.ptr();
}

namespace {

struct GetPythonGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    std::vector<std::string> gradientInputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientInputs.push_back(I(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(O(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(GO(i));
    }
    std::vector<std::string> gradientOutputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientOutputs.push_back(GI(i));
    }

    return SingleGradientDef(
        "PythonGradient", "", gradientInputs, gradientOutputs);
  }
};

REGISTER_CPU_OPERATOR(Python, PythonOp);
REGISTER_CPU_OPERATOR(PythonGradient, PythonGradientOp);
// Always allow running in-place
OPERATOR_SCHEMA(Python).AllowInplace([](int, int) { return true; });
OPERATOR_SCHEMA(PythonGradient).AllowInplace([](int, int) { return true; });

REGISTER_GRADIENT(Python, GetPythonGradient);
}
}
