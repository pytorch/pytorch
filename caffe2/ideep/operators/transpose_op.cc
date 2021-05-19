#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

class IDEEPTransposeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        axes_(this->template GetRepeatedArgument<int>("axes")){ }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPTransposeOp() override {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    Y->transpose_from(X.to_public(nullptr, X.get_data_type()), axes_);

    return true;
  }

 private:
  std::vector<int> axes_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_IDEEP_OPERATOR(Transpose, IDEEPTransposeOp);

} // namespace
