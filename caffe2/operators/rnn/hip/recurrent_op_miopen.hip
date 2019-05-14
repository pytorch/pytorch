#include "caffe2/operators/rnn/hip/recurrent_op_miopen.h"
#include "caffe2/utils/math.h"

#include <map>

namespace caffe2 {

namespace detail {

template <typename T>
TensorDescriptors<T>::TensorDescriptors(
    size_t n,
    std::vector<int>& dim,
    std::vector<int>& stride) {
  descs_.resize(n);
  CAFFE_ENFORCE_EQ(dim.size(), stride.size());
  for (auto i = 0; i < n; ++i) {
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&descs_[i]));
    MIOPEN_ENFORCE(miopenSetTensorDescriptor(
        descs_[i],
        miopenTypeWrapper<T>::type,
        dim.size(),
        dim.data(),
        stride.data()));
  }
}

template <typename T>
TensorDescriptors<T>::~TensorDescriptors() {
  for (auto desc : descs_) {
    miopenDestroyTensorDescriptor(desc);
  }
}
}

template <typename T>
RecurrentBaseOp<T>::RecurrentBaseOp(
    const OperatorDef& operator_def,
    Workspace* ws)
    : Operator<HIPContext>(operator_def, ws), miopen_wrapper_(&context_) {
  MIOPEN_ENFORCE(miopenCreateRNNDescriptor(&rnnDesc_));
  MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&wDesc_));
  MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&hxDesc_));
  MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&cxDesc_));
  MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&hyDesc_));
  MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&cyDesc_));
}

template <typename T>
RecurrentBaseOp<T>::~RecurrentBaseOp() {
  MIOPEN_ENFORCE(miopenDestroyRNNDescriptor(rnnDesc_));
  MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(wDesc_));
  MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(hxDesc_));
  MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(cxDesc_));
  MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(hyDesc_));
  MIOPEN_ENFORCE(miopenDestroyTensorDescriptor(cyDesc_));
}

template <typename T>
void RecurrentBaseOp<T>::initialize(
    const Tensor& input,
    Tensor* output,
    Tensor* hiddenOutput,
    Tensor* cellOutput) {
  static_assert(sizeof(T) == 4, ""); // workaround clang bug
  CAFFE_ENFORCE_GE(input.ndim(), 3);
  const int seqLength = input.dim(0);
  const int batchSize = input.dim(1);
  const int inputDim = input.dim(2);
  const int hiddenSize = OperatorBase::GetSingleArgument<int>("hidden_size", 0);
  CAFFE_ENFORCE_GT(hiddenSize, 0);
  const auto bidirectional =
      OperatorBase::GetSingleArgument<int>("bidirectional", 0);
  CAFFE_ENFORCE(bidirectional == 0 || bidirectional == 1);
  const auto numDirections = bidirectional == 1 ? 2 : 1;
  const auto outputDim = hiddenSize * numDirections;
  const auto rnnDirection =
      bidirectional == 1 ? miopenRNNbidirection : miopenRNNunidirection;
  const auto numLayers = OperatorBase::GetSingleArgument<int>("num_layers", 0);
  CAFFE_ENFORCE_GT(numLayers, 0);
  const auto& rnnModeStr =
      OperatorBase::GetSingleArgument<string>("rnn_mode", "");
  CAFFE_ENFORCE(rnnModeStr == "lstm" || rnnModeStr == "gru");
  const auto rnnMode = rnnModeStr == "lstm" ? miopenLSTM : miopenGRU;
  const auto& rnnInputStr =
      OperatorBase::GetSingleArgument<string>("input_mode", "");
  CAFFE_ENFORCE(rnnInputStr == "linear" || rnnInputStr == "skip");
  const auto rnnInput =
      rnnInputStr == "linear" ? miopenRNNlinear : miopenRNNskip;

  // RNN setup
  {
    MIOPEN_ENFORCE(miopenSetRNNDescriptor(
        rnnDesc_,
        hiddenSize,
        numLayers,
        rnnInput,
        rnnDirection,
        rnnMode,
        miopenRNNwithBias,
        miopenRNNdefault,
        miopenTypeWrapper<T>::type));
  }
  // X setup
  {
    std::vector<int> xDesc_dim = {batchSize, inputDim, 1};
    std::vector<int> xDesc_stride = {inputDim, 1, 1};
    xDesc_.reset(new detail::TensorDescriptors<T>(
        seqLength,
        xDesc_dim,
        xDesc_stride));
  }
  // Y setup
  { std::vector<int> yDesc_dim = {batchSize, hiddenSize * numDirections, 1};
    std::vector<int> yDesc_stride = {numDirections * hiddenSize, 1, 1};
    yDesc_.reset(new detail::TensorDescriptors<T>(
        seqLength,
        yDesc_dim,
        yDesc_stride));

    if (output) {
      output->Resize(std::vector<int>{seqLength, batchSize, outputDim});
    }
  }

  // Hidden/Cell setup
  {
    std::array<int, 3> dim{
        numLayers * numDirections, batchSize, hiddenSize};
    std::array<int, 3> stride{batchSize * hiddenSize, hiddenSize, 1};
    MIOPEN_ENFORCE(miopenSetTensorDescriptor(
        hxDesc_, miopenTypeWrapper<T>::type, 3, dim.data(), stride.data()));
    MIOPEN_ENFORCE(miopenSetTensorDescriptor(
        cxDesc_, miopenTypeWrapper<T>::type, 3, dim.data(), stride.data()));
    MIOPEN_ENFORCE(miopenSetTensorDescriptor(
        hyDesc_, miopenTypeWrapper<T>::type, 3, dim.data(), stride.data()));
    MIOPEN_ENFORCE(miopenSetTensorDescriptor(
        cyDesc_, miopenTypeWrapper<T>::type, 3, dim.data(), stride.data()));

    if (hiddenOutput) {
      hiddenOutput->Resize(
          std::vector<int>{numLayers * numDirections, batchSize, hiddenSize});
    }

    if (cellOutput) {
      cellOutput->Resize(
          std::vector<int>{numLayers * numDirections, batchSize, hiddenSize});
    }
  }

  // Weights setup
  {
    MIOPEN_ENFORCE(miopenGetRNNParamsDescriptor(
        miopen_wrapper_.inline_miopen_handle(),
        rnnDesc_,
        xDesc_->descs()[0],
        wDesc_,
        miopenTypeWrapper<T>::type));
  }

  // RNN workspace size
  {
    MIOPEN_ENFORCE(miopenGetRNNWorkspaceSize(
        miopen_wrapper_.inline_miopen_handle(),
        rnnDesc_,
        seqLength,
        xDesc_->descs(),
        &miopenWsNbytes_));
  }
}

template <typename T>
bool RecurrentOp<T>::RunOnDevice() {
  const int seqLength = Input(INPUT).dim32(0);
  if (Input(INPUT).sizes() != cachedInputDims_) {
    initialize(
        Input(INPUT),
        Output(OUTPUT),
        Output(HIDDEN_OUTPUT),
        Output(CELL_OUTPUT));
    cachedInputDims_ = Input(INPUT).sizes().vec();
  }

  // Validation checks
  size_t weightsSize;
  MIOPEN_ENFORCE(miopenGetRNNParamsSize(
      miopen_wrapper_.inline_miopen_handle(),
      rnnDesc_,
      xDesc_->descs()[0],
      &weightsSize,
      miopenTypeWrapper<T>::type));

  CAFFE_ENFORCE_EQ(Input(WEIGHT).nbytes(), weightsSize);

  // Training reserve size
  MIOPEN_ENFORCE(miopenGetRNNTrainingReserveSize(
      miopen_wrapper_.inline_miopen_handle(),
      rnnDesc_,
      seqLength,
      xDesc_->descs(),
      &reserveNbytes_));
  Output(RNN_SCRATCH)
      ->Resize(std::vector<int>{static_cast<int>(
          reserveNbytes_ / 4)}); // sizeof(T) - workaround clang bug
  Output(RNN_SCRATCH)->template mutable_data<T>();

  auto InputData = [this](int i) { return this->Input(i).template data<T>(); };
  auto OutputData = [this](int i) {
    return this->Output(i)->template mutable_data<T>();
  };

  if (OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) {
    miopen_wrapper_.with_miopen_state(0, [&](MIOPENState* state) {
      MIOPEN_ENFORCE(miopenRNNForwardInference(
          state->miopen_handle(),
          rnnDesc_,
          seqLength,
          xDesc_->descs(),
          InputData(INPUT), //.template data<T>(),
          hxDesc_,
          InputData(HIDDEN_INPUT), //.template data<T>(),
          cxDesc_,
          InputData(CELL_INPUT), //.template data<T>(),
          wDesc_,
          InputData(WEIGHT), //.template data<T>(),
          yDesc_->descs(),
          OutputData(OUTPUT), //->template mutable_data<T>(),
          hyDesc_,
          OutputData(HIDDEN_OUTPUT), //->template mutable_data<T>(),
          cyDesc_,
          OutputData(CELL_OUTPUT), //->template mutable_data<T>(),
          state->workspace().get(miopenWsNbytes_),
          miopenWsNbytes_));
    });
  } else {
    miopen_wrapper_.with_miopen_state(0, [&](MIOPENState* state) {
      MIOPEN_ENFORCE(miopenRNNForwardTraining(
          state->miopen_handle(),
          rnnDesc_,
          seqLength,
          xDesc_->descs(),
          InputData(INPUT), //.template data<T>(),
          hxDesc_,
          InputData(HIDDEN_INPUT), //.template data<T>(),
          cxDesc_,
          InputData(CELL_INPUT), //.template data<T>(),
          wDesc_,
          InputData(WEIGHT), //.template data<T>(),
          yDesc_->descs(),
          OutputData(OUTPUT), //->template mutable_data<T>(),
          hyDesc_,
          OutputData(HIDDEN_OUTPUT), //->template mutable_data<T>(),
          cyDesc_,
          OutputData(CELL_OUTPUT), //->template mutable_data<T>(),
          state->workspace().get(miopenWsNbytes_),
          miopenWsNbytes_,
          OutputData(RNN_SCRATCH), //->template mutable_data<T>(),
          reserveNbytes_));
    });
  }
  return true;
}

template <typename T>
bool RecurrentGradientOp<T>::RunOnDevice() {
  const int seqLength = Input(INPUT).dim32(0);
  if (Input(INPUT).sizes() != cachedInputDims_) {
    initialize(Input(INPUT));
    cachedInputDims_ = Input(INPUT).sizes().vec();
  }
  MIOPEN_ENFORCE(miopenGetRNNTrainingReserveSize(
      miopen_wrapper_.inline_miopen_handle(),
      rnnDesc_,
      seqLength,
      xDesc_->descs(),
      &reserveNbytes_));
  CAFFE_ENFORCE_EQ(reserveNbytes_, Input(RNN_SCRATCH).nbytes());
  Output(GRAD_INPUT)->ResizeLike(Input(INPUT));
  Output(GRAD_HIDDEN_INPUT)->ResizeLike(Input(HIDDEN_INPUT));
  Output(GRAD_CELL_INPUT)->ResizeLike(Input(CELL_INPUT));

  Output(GRAD_WEIGHT)->ResizeLike(Input(WEIGHT));
  math::Set<T, HIPContext>(
      Output(GRAD_WEIGHT)->size(),
      0.0,
      Output(GRAD_WEIGHT)->template mutable_data<T>(),
      &context_);

  auto* reserve = Output(RNN_SCRATCH_OUT)->template mutable_data<T>();
  auto InputData = [this](int i) { return this->Input(i).template data<T>(); };
  auto OutputData = [this](int i) {
    return this->Output(i)->template mutable_data<T>();
  };

  miopen_wrapper_.with_miopen_state(0, [&](MIOPENState* state) {
    MIOPEN_ENFORCE(miopenRNNBackwardData(
        state->miopen_handle(),
        rnnDesc_,
        seqLength,
        yDesc_->descs(),
        InputData(OUTPUT), // Input(OUTPUT).template data<T>(),
        yDesc_->descs(),
        InputData(GRAD_OUTPUT), // Input(GRAD_OUTPUT).template data<T>(),
        hyDesc_,
        nullptr,
        cyDesc_,
        nullptr,
        wDesc_,
        InputData(WEIGHT), // Input(WEIGHT).template data<T>(),
        hxDesc_,
        InputData(HIDDEN_INPUT), // Input(HIDDEN_INPUT).template data<T>(),
        cxDesc_,
        InputData(CELL_INPUT),
        xDesc_->descs(),
        OutputData(GRAD_INPUT),
        hxDesc_,
        OutputData(GRAD_HIDDEN_INPUT),
        cxDesc_,
        OutputData(GRAD_CELL_INPUT),
        state->workspace().get(miopenWsNbytes_),
        miopenWsNbytes_,
        reserve,
        reserveNbytes_));
    MIOPEN_ENFORCE(miopenRNNBackwardWeights(
        state->miopen_handle(),
        rnnDesc_,
        seqLength,
        xDesc_->descs(),
        InputData(INPUT), // Input(INPUT).template data<T>(),
        hxDesc_,
        InputData(HIDDEN_INPUT), // Input(HIDDEN_INPUT).template data<T>(),
        yDesc_->descs(),
        InputData(OUTPUT), // Input(OUTPUT).template data<T>(),
        wDesc_,
        OutputData(
            GRAD_WEIGHT), // Output(GRAD_WEIGHT)->template mutable_data<T>(),
        state->workspace().get(miopenWsNbytes_),
        miopenWsNbytes_,
        reserve,
        reserveNbytes_));
  });
  return true;
}

template <typename T, RecurrentParamOpMode mode>
bool RecurrentParamAccessOp<T, mode>::RunOnDevice() {
  initialize(Input(0));

  if (mode == SET_PARAM) {
    size_t paramsSize;
    MIOPEN_ENFORCE(miopenGetRNNParamsSize(
        miopen_wrapper_.inline_miopen_handle(),
        rnnDesc_,
        xDesc_->descs()[0],
        &paramsSize,
        miopenTypeWrapper<T>::type));

    CAFFE_ENFORCE_EQ(
        paramsSize / 4, Input(1).size(), "Incorrect weight initialization");
  }

  int layer = OperatorBase::GetSingleArgument<int>("layer", 0);
  std::string param_type =
      OperatorBase::GetSingleArgument<string>("param_type", "");
  std::string input_type =
      OperatorBase::GetSingleArgument<string>("input_type", "");

  // Mapping to MIOPEN constants
  std::map<string, int> weight_constants = {{"input_gate_w", 0},
                                            {"forget_gate_w", 1},
                                            {"cell_w", 3},
                                            {"output_gate_w", 2}};
  std::map<string, int> bias_constants = {{"input_gate_b", 0},
                                          {"forget_gate_b", 1},
                                          {"cell_b", 3},
                                          {"output_gate_b", 2}};
  if (bias_constants.find(param_type) != bias_constants.end()) {
    int param_id = bias_constants[param_type] + 4 * (input_type == "recurrent");

    miopenTensorDescriptor_t biasDesc;
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&biasDesc));

    size_t bias_size = 0;
    MIOPEN_ENFORCE(miopenGetRNNLayerBiasSize(
        miopen_wrapper_.inline_miopen_handle(),
        rnnDesc_,
        layer,
        param_id,
        &bias_size));

    void* bias;
    miopen_wrapper_.with_miopen_state(0, [&](MIOPENState* state) {
        bias = state->workspace().get(bias_size);
    });
    MIOPEN_ENFORCE(miopenGetRNNLayerBias(
        miopen_wrapper_.inline_miopen_handle(),
        rnnDesc_,
        layer,
        xDesc_->descs()[0],
        wDesc_,
        Input(1).template data<T>(),
        param_id,
        biasDesc,
        bias));
    std::array<int,3> biasDims {1,1,1};
    std::array<int,3> strideDims {1,1,1};
    miopenDataType_t dt;

    MIOPEN_ENFORCE(miopenGetTensorDescriptor(
      biasDesc, &dt, biasDims.data(), strideDims.data()));
    CAFFE_ENFORCE_EQ(biasDims.size(), 3);

    if (mode == SET_PARAM) {
      CAFFE_ENFORCE_EQ(
          biasDims[0] * biasDims[1] * biasDims[2], Input(2).size());
      context_.template CopySameDevice<T>(
          biasDims[0] * biasDims[1] * biasDims[2],
          Input(2).template data<T>(),
          static_cast<T*>(bias));
    } else {
      Output(0)->Resize(biasDims);
      context_.template CopySameDevice<T>(
          biasDims[0] * biasDims[1] * biasDims[2],
          static_cast<T*>(bias),
          Output(0)->template mutable_data<T>());
    }
  } else if (weight_constants.find(param_type) != weight_constants.end()) {
    int param_id =
        weight_constants[param_type] + 4 * (input_type == "recurrent");
    miopenTensorDescriptor_t matrixParamDesc;
    MIOPEN_ENFORCE(miopenCreateTensorDescriptor(&matrixParamDesc));

    size_t param_size = 0;
    MIOPEN_ENFORCE(miopenGetRNNLayerParamSize(
        miopen_wrapper_.inline_miopen_handle(),
        rnnDesc_,
        layer,
        xDesc_->descs()[0],
        param_id,
        &param_size));

    void* pmatrix;
    miopen_wrapper_.with_miopen_state(0, [&](MIOPENState* state) {
        pmatrix = state->workspace().get(param_size);
    });
    MIOPEN_ENFORCE(miopenGetRNNLayerParam(
        miopen_wrapper_.inline_miopen_handle(),
        rnnDesc_,
        layer,
        xDesc_->descs()[0],
        wDesc_,
        Input(1).template data<T>(),
        param_id,
        matrixParamDesc,
        pmatrix));
    std::array<int,3> matDims {1,1,1};
    std::array<int,3> strideDims {1,1,1};
    miopenDataType_t dt;

    MIOPEN_ENFORCE(miopenGetTensorDescriptor(
        matrixParamDesc, &dt, matDims.data(), strideDims.data()));
    CAFFE_ENFORCE_EQ(matDims.size(), 3);
    if (mode == SET_PARAM) {
      CAFFE_ENFORCE_EQ(matDims[0] * matDims[1] * matDims[2], Input(2).size());
      context_.template CopySameDevice<T>(
          matDims[0] * matDims[1] * matDims[2],
          Input(2).template data<T>(),
          static_cast<T*>(pmatrix));
    } else {
      Output(0)->Resize(matDims);
      context_.template CopySameDevice<T>(
          matDims[0] * matDims[1] * matDims[2],
          static_cast<T*>(pmatrix),
          Output(0)->template mutable_data<T>());
    }
  } else {
    CAFFE_ENFORCE(false, "Unknown param type:", param_type);
  }

  return true;
}

REGISTER_MIOPEN_OPERATOR(Recurrent, RecurrentOp<float>);
OPERATOR_SCHEMA(Recurrent).NumInputs(4).NumOutputs(5);

REGISTER_MIOPEN_OPERATOR(RecurrentGradient, RecurrentGradientOp<float>);
OPERATOR_SCHEMA(RecurrentGradient)
    .NumInputs(7)
    .NumOutputs(6)
    .AllowInplace({{4, 5}});

REGISTER_MIOPEN_OPERATOR(
    RecurrentParamSet,
    RecurrentParamAccessOp<float, SET_PARAM>);
OPERATOR_SCHEMA(RecurrentParamSet)
    .NumInputs(3)
    .NumOutputs(1)
    .EnforceInplace({{1, 0}});

REGISTER_MIOPEN_OPERATOR(
    RecurrentParamGet,
    RecurrentParamAccessOp<float, GET_PARAM>);
OPERATOR_SCHEMA(RecurrentParamGet)
    .NumInputs(2)
    .NumOutputs(1);

struct GetRecurrentGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "RecurrentGradient",
        "",
        vector<string>{I(0), // INPUT
                       I(1), // HIDDEN_INPUT
                       I(2), // CELL_INPUT
                       I(3), // WEIGHT
                       O(3), // RNN_SCRATCH
                       O(0), // OUTPUT
                       GO(0)}, // GRAD_OUTPUT
        // TODO: not currently using these gradients, investigate t16675365
        //     GO(1), // GRAD_HIDDEN_OUTPUT
        //     GO(2)}, // GRAD_CELL_OUTPUT
        vector<string>{
            GI(0), // GRAD_INPUT
            GI(1), // GRAD_HIDDEN_INPUT
            GI(2), // GRAD_CELL_INPUT
            GI(3), // GRAD_WEIGHT
            O(4), // DROPOUT_STATES
            O(3) // RNN_SCRATCH
        });
  }
};
REGISTER_GRADIENT(Recurrent, GetRecurrentGradient);
}
