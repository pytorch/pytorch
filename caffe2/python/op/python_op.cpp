#include <unordered_map>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

// Produce deprecation warnings (needs to come before arrayobject.h inclusion).
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace caffe2 {

namespace bp = boost::python;

namespace detail {

class PythonGuard : private boost::noncopyable {
 public:
  PythonGuard() : gstate_(PyGILState_Ensure()) {}
  ~PythonGuard() {
    PyGILState_Release(gstate_);
  }

 private:
  PyGILState_STATE gstate_;
};

using FuncRegistery = std::unordered_map<std::string, bp::object>;
static FuncRegistery& gRegistery() {
  // Always leak the objects registered here.
  static FuncRegistery* r = new FuncRegistery();
  return *r;
}

bp::object& getFunc(const std::string& token) {
  return gRegistery()[token];
}

bp::object& getGradientFunc(const std::string& token) {
  return gRegistery()[token + "_gradient"];
}

std::string registerFunc(const bp::object& func) {
  CHECK(!func.is_none());
  const std::string name = bp::extract<std::string>(func.attr("__name__"));
  // Unique name since registry is never cleared.
  const std::string token = name + std::to_string(gRegistery().size());
  CHECK(gRegistery().find(name) == gRegistery().end());
  gRegistery()[token] = func;
  return token;
}

bp::object registerGradientFunc(
    const bp::object& token_,
    const bp::object& func) {
  CHECK(!token_.is_none());
  CHECK(!func.is_none());
  const std::string token = bp::extract<std::string>(token_);
  CHECK(gRegistery().find(token) != gRegistery().end());
  gRegistery()[token + "_gradient"] = func;
  return bp::object();
}

struct NdarrayConverterGenerator {
  template <typename T>
  struct apply;
};

template <>
struct NdarrayConverterGenerator::apply<float*> {
  struct type {
    PyObject* operator()(float* data) const {
      // Just store the data pointer, and add the shape information in postcall.
      return PyArray_SimpleNewFromData(0, nullptr, NPY_FLOAT32, data);
    }
    const PyTypeObject* get_pytype() {
      return &PyArray_Type;
    }
  };
};

struct NdarrayCallPolicies : public bp::default_call_policies {
  typedef NdarrayConverterGenerator result_converter;
  PyObject* postcall(PyObject* pyargs, PyObject* result) {
    bp::object pyblob = bp::extract<bp::tuple>(pyargs)()[0];
    boost::shared_ptr<TensorCPU> blob =
        bp::extract<boost::shared_ptr<TensorCPU>>(pyblob);
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int num_axes = blob->ndim();
    std::vector<npy_intp> dims(blob->dims().begin(), blob->dims().end());
    PyObject* arr_obj =
        PyArray_SimpleNewFromData(num_axes, dims.data(), NPY_FLOAT32, data);
    // SetBaseObject steals a ref, so we need to INCREF.
    Py_INCREF(pyblob.ptr());
    PyArray_SetBaseObject(
        reinterpret_cast<PyArrayObject*>(arr_obj), pyblob.ptr());
    return arr_obj;
  }
};

bp::tuple TensorCPU_shape(const TensorCPU& t) {
  return bp::tuple(t.dims());
}

bp::object TensorCPU_reshape(TensorCPU* t, const bp::tuple& dims_) {
  std::vector<TIndex> dims;
  dims.reserve(bp::len(dims_));
  for (auto i = 0; i < bp::len(dims_); ++i) {
    dims.push_back(bp::extract<int64_t>(dims_[i]));
  }
  t->Resize(dims);
  return bp::object();
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
    CHECK(!pyFunc.is_none());
    {
      detail::PythonGuard g;
      try {
        pyFunc(inputs, outputs);
      } catch (bp::error_already_set&) {
        PyErr_Print();
        LOG(FATAL) << "Exception in Python operator for token: "
                   << OperatorBase::GetSingleArgument<std::string>("token", "");
      }
    }
    return true;
  }

 private:
  virtual bp::object& getFunc() = 0;
};

class PythonOp final : public PythonOpBase {
 public:
  using PythonOpBase::PythonOpBase;

 private:
  bp::object& getFunc() override {
    const std::string& token =
        OperatorBase::GetSingleArgument<std::string>("token", "");
    return detail::getFunc(token);
  }
};

class PythonGradientOp final : public PythonOpBase {
 public:
  using PythonOpBase::PythonOpBase;

 private:
  bp::object& getFunc() override {
    const std::string& token =
        OperatorBase::GetSingleArgument<std::string>("token", "");
    return detail::getGradientFunc(token);
  }
};


BOOST_PYTHON_MODULE(python_ops_python) {
  bp::class_<TensorCPU, boost::shared_ptr<TensorCPU>, boost::noncopyable>(
      "TensorCPU")
      .add_property(
          "data",
          bp::make_function(
              &TensorCPU::template mutable_data<float>,
              detail::NdarrayCallPolicies()))
      .add_property("shape", detail::TensorCPU_shape)
      .def("reshape", detail::TensorCPU_reshape);

  bp::class_<std::vector<TensorCPU*>>("RawTensorVec")
      .def(bp::vector_indexing_suite<std::vector<TensorCPU*>, true>());

  bp::class_<vector<TIndex>>("IntVec").def(
      bp::vector_indexing_suite<std::vector<TIndex>>());

  bp::def(
      "register",
      detail::registerFunc,
      bp::args("func"),
      "Register a function, returning a token");
  bp::def(
      "register_gradient",
      detail::registerGradientFunc,
      bp::args("token", "func"),
      "Register a gradient function for a token");
  // boost python expects a void (missing) return value, while import_array
  // returns NULL for python3. import_array1() forces a void return value.
  import_array1();
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
