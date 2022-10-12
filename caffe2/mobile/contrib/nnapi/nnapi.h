#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/proto_utils.h"

#include "NeuralNetworks.h"
#include "dlnnapi.h"

namespace caffe2 {

class NNApi {
 public:
  using TensorVector = std::vector<TensorCPU*>;

  // Three different modes:
  // ANEURALNETWORKS_PREFER_LOW_POWER
  // ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER
  // ANEURALNETWORKS_PREFER_SUSTAINED_SPEED
  NNApi(
      const NetDef& init_net,
      const NetDef& run_net,
      Workspace* ws = nullptr,
      const PreferenceCode pref = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED)
      : preference_(pref),
        run_net_(run_net),
        ws_(ws) {
    if (!loadNNApiLibrary()) {
      CAFFE_THROW("NNApi is not supported");
    }
    CAFFE_ENFORCE(ws_.RunNetOnce(init_net));
  }

  ~NNApi();

  bool loadNNApiLibrary();

  bool run(const TensorVector& inputs, TensorVector* outputs);

 private:
  dlnnapi libnnapi_;
  ANeuralNetworksModel* model_{nullptr};
  ANeuralNetworksCompilation* compilation_{nullptr};
  ANeuralNetworksExecution* run_{nullptr};
  ANeuralNetworksEvent* run_end_{nullptr};
  PreferenceCode preference_;
  NetDef run_net_;
  Workspace ws_;
  OperandCode tensor_type_;
  uint32_t operand_idx{0};
  std::unordered_map<std::string, uint32_t> operand_map_;
  // dimensions for the tensors
  std::unordered_map<std::string, std::vector<uint32_t>> tensor_dims_;

  // mapping of the operator name "Conv" to OperatorType CONV
  enum OperatorType {
    AVERAGEPOOL,
    CONV,
    MAXPOOL,
    RELU,
    SOFTMAX,
  };

  std::unordered_map<std::string, OperatorType> operator_map_{
      {"AveragePool", AVERAGEPOOL},
      {"Conv", CONV},
      {"MaxPool", MAXPOOL},
      {"Relu", RELU},
      {"Softmax", SOFTMAX}};

  struct ConvPoolArgs {
    int kernel_h{0};
    int kernel_w{0};
    int stride_x{0};
    int stride_y{0};
    int pad_t{0};
    int pad_l{0};
    int pad_b{0};
    int pad_r{0};
  };

  void getConvPoolArgs(const ArgumentHelper& helper, ConvPoolArgs& args);

  uint32_t addScalarOperand(int32_t val);

  uint32_t addFloatOperand(float val);

  uint32_t addTensorOperand(
      const std::string& blob,
      OperandCode type,
      std::vector<uint32_t>& dims,
      float scale = 1.0,
      int32_t zero_point = 0);

  // lazily initialize model_ in run()
  void init(const TensorVector& inputs, TensorVector* outputs);

  void addConv(const OperatorDef& op, bool fuse_relu = false);

  void addPooling(
      const OperatorDef& op,
      OperationCode op_code,
      bool fuse_relu = false);

  void addRelu(const OperatorDef& op);

  void addSoftmax(const OperatorDef& op);
};
} // namespace caffe2
