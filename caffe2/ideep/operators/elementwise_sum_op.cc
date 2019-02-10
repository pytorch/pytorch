#include <caffe2/ideep/ideep_utils.h>
#include <caffe2/ideep/operators/operator_fallback_ideep.h>
#include "caffe2/operators/utility_ops.h"
#include "caffe2/operators/elementwise_add_op.h"

namespace caffe2 {

class IDEEPSumOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();
  using FALLBACK_SUM = IDEEPFallbackOp<SumOp<CPUContext>, SkipIndices<0>>;
  using FALLBACK_ADD = IDEEPFallbackOp<BinaryElementwiseOp<
    NumericTypes, CPUContext, AddFunctor<CPUContext>>, SkipIndices<0>>;

  IDEEPSumOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        fallback_sum_(operator_def, ws),
        fallback_add_(operator_def, ws) {}
  virtual ~IDEEPSumOp() {}

  bool RunOnDevice() override {
    itensor::dims input_dims;
    bool fallback_to_cpu = false;
    vector<itensor> inputs_itensor;

    // We only support element-wise sum for ideep tensors here.
    // If a CPU tensor is detected in input list, we have to fallback
    // to corresponding CPU operator.
    for (int i = 0; i < InputSize(); ++i) {
      if (OperatorBase::InputBlob(i).template IsType<itensor>()) {
        auto& tensor_ideep = Input(i);
        if (input_dims.empty()) {
          input_dims = tensor_ideep.get_dims();
        } else if (input_dims != tensor_ideep.get_dims()) {
          fallback_to_cpu = true;
          break;
        }
        inputs_itensor.emplace_back(tensor_ideep);
      } else {
        CAFFE_ENFORCE(
            BlobIsTensorType(OperatorBase::InputBlob(i), CPU),
            "Expect cpu tensor if not itensor");
        fallback_to_cpu = true;
        break;
      }
    }

    if (!fallback_to_cpu) {
      auto* Y = Output(OUTPUT);
      if (InputSize() == 1) {
        const auto& X = Input(INPUT0);
        ideep::direct_copy::compute(X, *Y);
      } else {
        const vector<float> scales(InputSize(), 1.0);
        ideep::sum::compute(scales, inputs_itensor, *Y);
      }
      return true;
    }

    if (InputSize() == 2) {
      return fallback_add_.Run(0);
    }

    return fallback_sum_.Run(0);
  }

 private:
  FALLBACK_SUM fallback_sum_;
  FALLBACK_ADD fallback_add_;

  INPUT_TAGS(INPUT0);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(Sum, IDEEPSumOp);
REGISTER_IDEEP_OPERATOR(Add, IDEEPSumOp);

} // namespace caffe2
