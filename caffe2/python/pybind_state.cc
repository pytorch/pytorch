#include "pybind_state.h"

#include <chrono>
#include <future>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/contrib/script/compiler.h"
#include "caffe2/core/asan.h"
#include "caffe2/core/blob_stats.h"
#include "caffe2/core/db.h"
#include "caffe2/core/numa.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/stats.h"
#include "caffe2/core/transform.h"
#include "caffe2/mkl/mkl_utils.h"
#include "caffe2/observers/runcnt_observer.h"
#include "caffe2/observers/time_observer.h"
#include "caffe2/onnx/backend.h"
#include "caffe2/onnx/helper.h"
#include "caffe2/onnx/onnx_exporter.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"
#include "caffe2/opt/mobile.h"
#include "caffe2/opt/onnxifi_transformer.h"
#include "caffe2/opt/optimize_ideep.h"
#include "caffe2/opt/sink.h"
#include "caffe2/predictor/predictor.h"
#include "caffe2/python/pybind_state_registry.h"
#include "caffe2/utils/cpuid.h"
#include "caffe2/utils/string_utils.h"

namespace caffe2 {
namespace python {

namespace py = pybind11;

class GetPythonGradient : public GradientMakerBase {
 public:
  using GradientMakerBase::GradientMakerBase;
  std::vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE(Def().type() == "Python" || Def().type() == "PythonDLPack");
    ArgumentHelper helper(Def());
    auto gradOutputIndices =
        helper.GetRepeatedArgument<int>("grad_output_indices");
    auto gradInputIndices =
        helper.GetRepeatedArgument<int>("grad_input_indices");
    std::vector<std::string> gradientInputs;
    for (int i = 0; i < def_.input_size(); ++i) {
      gradientInputs.push_back(I(i));
    }
    for (int i = 0; i < def_.output_size(); ++i) {
      gradientInputs.push_back(O(i));
    }
    if (gradOutputIndices.size() > 0) {
      for (int i = 0; i < gradOutputIndices.size(); ++i) {
        int GO_i = gradOutputIndices[i];
        gradientInputs.push_back(GO(GO_i));
      }
    } else {
      for (int i = 0; i < def_.output_size(); ++i) {
        gradientInputs.push_back(GO(i));
      }
    }
    std::vector<std::string> gradientOutputs;
    if (gradInputIndices.size() > 0) {
      for (int i = 0; i < gradInputIndices.size(); ++i) {
        int GI_i = gradInputIndices[i];
        gradientOutputs.push_back(GI(GI_i));
      }
    } else {
      for (int i = 0; i < def_.input_size(); ++i) {
        gradientOutputs.push_back(GI(i));
      }
    }

    std::string grad_op_name = "PythonGradient";
    if (Def().type() == "PythonDLPack") {
      grad_op_name = "PythonDLPackGradient";
    }
    return SingleGradientDef(grad_op_name, "", gradientInputs, gradientOutputs);
  }
};

REGISTER_CPU_OPERATOR(Python, PythonOp<CPUContext, false>);
REGISTER_CPU_OPERATOR(PythonGradient, PythonGradientOp<CPUContext, false>);
// Always allow running in-place
OPERATOR_SCHEMA(Python).AllowInplace([](int, int) { return true; });
OPERATOR_SCHEMA(PythonGradient).AllowInplace([](int, int) { return true; });
REGISTER_GRADIENT(Python, GetPythonGradient);

REGISTER_CPU_OPERATOR(PythonDLPack, PythonOp<CPUContext, true>);
REGISTER_CPU_OPERATOR(PythonDLPackGradient, PythonGradientOp<CPUContext, true>);
OPERATOR_SCHEMA(PythonDLPack).AllowInplace([](int, int) { return true; });
OPERATOR_SCHEMA(PythonDLPackGradient).AllowInplace([](int, int) {
  return true;
});
REGISTER_GRADIENT(PythonDLPack, GetPythonGradient);

PYBIND11_MODULE(caffe2_pybind11_state, m) {
  m.doc() = "pybind11 stateful interface to Caffe2 workspaces";

  for (const auto& addition : PybindAdditionRegistry()->Keys()) {
    PybindAdditionRegistry()->Create(addition, m);
  }
}

} // namespace python
} // namespace caffe2
