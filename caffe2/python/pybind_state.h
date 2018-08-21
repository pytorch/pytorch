#pragma once

#include <unordered_map>

#include "caffe2/core/context.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/memonger.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/python/pybind_state_detail.h"
#include "caffe2/python/pybind_state_dlpack.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL caffe2_python_ARRAY_API
#include <numpy/arrayobject.h>

// Temporary solution for numpy < 1.7 versions: old macro, no promises.
// You're strongly advised to upgrade to >= 1.7.
#ifndef NPY_ARRAY_C_CONTIGUOUS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define PyArray_SetBaseObject(arr, x) (PyArray_BASE(arr) = (x))
#endif

namespace caffe2 {
namespace python {

namespace py = pybind11;

template <class Context, bool use_dlpack>
class PythonOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  PythonOpBase(
      const OperatorDef& operator_def,
      Workspace* ws,
      const std::string& pickled_builder_arg_name)
      : Operator<Context>(operator_def, ws),
        ws_(ws),
        token_(OperatorBase::template GetSingleArgument<std::string>(
            "token",
            "")) {
    using namespace python_detail;
    auto pickled = OperatorBase::template GetSingleArgument<std::string>(
        pickled_builder_arg_name, "");
    auto forced_cpu_outputs_arg =
        OperatorBase::template GetRepeatedArgument<int>("forced_cpu_outputs");
    forced_cpu_outputs_.insert(
        forced_cpu_outputs_arg.begin(), forced_cpu_outputs_arg.end());
    CAFFE_ENFORCE(
        !pickled.empty() || !token_.empty(),
        "PythonOp requires either pickled_builder or token arg.");
    if (!pickled.empty()) {
      py::gil_scoped_acquire g;
      try {
        auto pickle =
            py::reinterpret_steal<py::object>(PyImport_ImportModule("pickle"));
        CAFFE_ENFORCE(pickle);
        auto loads = pickle.attr("loads").cast<py::object>();
        CAFFE_ENFORCE(loads);
        auto builder_call = loads(py::bytes(pickled)).cast<py::tuple>();
        CAFFE_ENFORCE(builder_call);
        CAFFE_ENFORCE_EQ(py::len(builder_call), 3);
        auto func = builder_call[0].cast<py::object>();
        auto args = builder_call[1].cast<py::tuple>();
        auto kwargs = builder_call[2].cast<py::dict>();
        auto built_func = func(*args, **kwargs);
        CAFFE_ENFORCE(built_func);
        built_func_.reset(
            new Func{built_func,
                     OperatorBase::template GetSingleArgument<bool>(
                         "pass_workspace", false)});
      } catch (const py::error_already_set& e) {
        std::stringstream error;
        error << "Python exception encountered while creating PythonOp: "
              << e.what();
        LOG(ERROR) << error.str();
        CAFFE_THROW(error.str());
      }
    }
  }

  bool RunOnDevice() override final {
    auto* pyFunc = built_func_ ? built_func_.get() : &getFunc(token_);
    CAFFE_ENFORCE(pyFunc);
    {
      // Acquire GIL for call to Python runtime.
      py::gil_scoped_acquire g;

      DeviceOption cpu_option;
      cpu_option.set_device_type(CPU);

      std::vector<py::object> inputs;
      inputs.reserve(InputSize());
      for (auto i = 0; i < InputSize(); ++i) {
        const auto* blob = &InputBlob(i);
        // Allow CPU tensors in addition to operator context's tensors
        py::object py_obj;
        if (blob->template IsType<Tensor>()) {
          if (use_dlpack) {
            DLPackWrapper<CPUContext> wrapper(
                const_cast<Tensor*>(&blob->template Get<Tensor>()), cpu_option);
            // copy wrapper
            py_obj = py::cast(wrapper, py::return_value_policy::copy);
          } else {
            py_obj = py::cast(
                &blob->template Get<Tensor>(),
                py::return_value_policy::reference);
          }
        } else {
          if (use_dlpack) {
            DLPackWrapper<Context> wrapper(
                const_cast<Tensor*>(&blob->template Get<Tensor>()),
                this->device_option());
            py_obj = py::cast(wrapper, py::return_value_policy::copy);
          } else {
            py_obj = py::cast(
                &blob->template Get<Tensor>(),
                py::return_value_policy::reference);
          }
        }
        inputs.push_back(py_obj);
      }
      std::vector<py::object> outputs;
      outputs.reserve(OutputSize());
      for (auto i = 0; i < OutputSize(); ++i) {
        auto* blob = OutputBlob(i);

        // Python op is always used with CPUContext only and treats inputs and
        // outputs as CPU tensors, CUDA version of PythonOp is implemented
        // through GPUFallbackOp that copies input CUDA blobs to CPU and copies
        // outputs from CUDA to CPU.
        // GPUFallbackOp also allows keeping some of the output blobs on CPU
        // by specifying their indices explicitly in template parameters.

        // PythonDLPack op allows working with CUDA and CPU blobs directly
        // through DLPack tensors. In order to properly setup mapping we need
        // to know in advance a type (CUDA or CPU) of an output blob.
        // Output blob might not be initialized yet, so by default we treat
        // output blobs as having the same type as operator's context.
        // This can be overwritten though forced_cpu_outputs argument

        // make sure output blob is initialized before creating the binding
        if (forced_cpu_outputs_.count(i)) {
          blob->GetMutableTensor(Context::GetDeviceType());
        } else {
          blob->GetMutableTensor(Context::GetDeviceType());
        }

        py::object py_obj;
        if (blob->template IsType<Tensor>()) {
          if (use_dlpack) {
            DLPackWrapper<CPUContext> wrapper(
                blob->GetMutableTensor(Context::GetDeviceType()), cpu_option);
            py_obj = py::cast(wrapper, py::return_value_policy::copy);
          } else {
            py_obj = py::cast(
                blob->GetMutableTensor(Context::GetDeviceType()),
                py::return_value_policy::reference);
          }
        } else {
          if (use_dlpack) {
            DLPackWrapper<Context> wrapper(
                blob->GetMutableTensor(Context::GetDeviceType()),
                this->device_option());
            py_obj = py::cast(wrapper, py::return_value_policy::copy);
          } else {
            py_obj = py::cast(
                blob->GetMutableTensor(Context::GetDeviceType()),
                py::return_value_policy::reference);
          }
        }
        outputs.push_back(py_obj);
      }

      try {
        if (pyFunc->needs_workspace) {
          pyFunc->py_func(inputs, outputs, ws_);
        } else {
          pyFunc->py_func(inputs, outputs);
        }
      } catch (const py::error_already_set& e) {
        std::stringstream error;
        error << "Exception encountered running PythonOp function: "
              << e.what();
        LOG(ERROR) << error.str();
        CAFFE_THROW(error.str());
      }
    }
    return true;
  }

  virtual ~PythonOpBase() {
    if (built_func_) {
      // since it may trigger python interpreter when refcount reaches zero
      py::gil_scoped_acquire g;
      built_func_.reset();
    }
  }

 protected:
  virtual const python_detail::Func& getFunc(const std::string& token) = 0;
  Workspace* ws_;
  // output indices forced to be on CPU
  std::unordered_set<int> forced_cpu_outputs_;

 private:
  const std::string token_;
  std::unique_ptr<python_detail::Func> built_func_;
};

template <class Context, bool use_dlpack>
class PythonOp : public PythonOpBase<Context, use_dlpack> {
 public:
  PythonOp(const OperatorDef& operator_def, Workspace* ws)
      : PythonOpBase<Context, use_dlpack>(operator_def, ws, "pickled_builder") {
  }

 protected:
  const python_detail::Func& getFunc(const std::string& token) override {
    return python_detail::getOpFunc(token);
  }
};

template <class Context, bool use_dlpack>
class PythonGradientOp : public PythonOpBase<Context, use_dlpack> {
 public:
  PythonGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : PythonOpBase<Context, use_dlpack>(
            operator_def,
            ws,
            "pickled_grad_builder") {}

 protected:
  const python_detail::Func& getFunc(const std::string& token) override {
    return python_detail::getGradientFunc(token);
  }
};

} // namespace python
} // namespace caffe2
