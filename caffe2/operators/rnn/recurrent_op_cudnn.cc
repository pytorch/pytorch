#include "caffe2/operators/rnn/recurrent_op_cudnn.h"
#include "caffe2/utils/math.h"

#include <map>

namespace caffe2 {

namespace detail {

template <typename T>
TensorDescriptors<T>::TensorDescriptors(
    size_t n,
    const std::vector<int>& dim,
    const std::vector<int>& stride) {
  descs_.resize(n);
  CAFFE_ENFORCE_EQ(dim.size(), stride.size());
  for (auto i = 0; i < n; ++i) {
    CUDNN_ENFORCE(cudnnCreateTensorDescriptor(&descs_[i]));
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        descs_[i],
        cudnnTypeWrapper<T>::type,
        dim.size(),
        dim.data(),
        stride.data()));
  }
}

template <typename T>
TensorDescriptors<T>::~TensorDescriptors() {
  for (auto desc : descs_) {
    cudnnDestroyTensorDescriptor(desc);
  }
}
}

template <typename T>
RecurrentBaseOp<T>::~RecurrentBaseOp() {
  CUDNN_ENFORCE(cudnnDestroyDropoutDescriptor(dropoutDesc_));
  CUDNN_ENFORCE(cudnnDestroyRNNDescriptor(rnnDesc_));
  CUDNN_ENFORCE(cudnnDestroyFilterDescriptor(wDesc_));
  CUDNN_ENFORCE(cudnnDestroyTensorDescriptor(hxDesc_));
}

template <typename T>
void RecurrentBaseOp<T>::initialize(
    const Tensor& input,
    Tensor* dropoutStates,
    Tensor* output,
    Tensor* hiddenOutput,
    Tensor* cellOutput) {
  static_assert(sizeof(T) == 4, ""); // workaround clang bug
  CAFFE_ENFORCE_GE(input.dim(), 3);
  const int seqLength = input.size(0);
  const int batchSize = input.size(1);
  const int inputDim = input.size(2);
  const int hiddenSize = OperatorBase::GetSingleArgument<int>("hidden_size", 0);
  CAFFE_ENFORCE_GT(hiddenSize, 0);
  const auto bidirectional =
      OperatorBase::GetSingleArgument<int>("bidirectional", 0);
  CAFFE_ENFORCE(bidirectional == 0 || bidirectional == 1);
  const auto numDirections = bidirectional == 1 ? 2 : 1;
  const auto outputDim = hiddenSize * numDirections;
  const auto rnnDirection =
      bidirectional == 1 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;
  const auto numLayers = OperatorBase::GetSingleArgument<int>("num_layers", 0);
  CAFFE_ENFORCE_GT(numLayers, 0);
  const auto& rnnModeStr =
      OperatorBase::GetSingleArgument<string>("rnn_mode", "");
  CAFFE_ENFORCE(rnnModeStr == "lstm" || rnnModeStr == "gru");
  const auto rnnMode = rnnModeStr == "lstm" ? CUDNN_LSTM : CUDNN_GRU;
  const auto& rnnInputStr =
      OperatorBase::GetSingleArgument<string>("input_mode", "");
  CAFFE_ENFORCE(rnnInputStr == "linear" || rnnInputStr == "skip");
  const auto rnnInput =
      rnnInputStr == "linear" ? CUDNN_LINEAR_INPUT : CUDNN_SKIP_INPUT;

  // Dropout setup
  {
    if (dropoutStates) {
      size_t stateSize;
      float dropout_param =
          OperatorBase::GetSingleArgument<float>("dropout", 1.0);
      if (dropout_param < 1.0) {
        CUDNN_ENFORCE(cudnnDropoutGetStatesSize(
            cudnn_wrapper_.inline_cudnn_handle(), &stateSize));
        dropoutStates->Resize(std::vector<int>{static_cast<int>(
            stateSize / 4 /* sizeof(T) - workaround clang bug */)});
        CUDNN_ENFORCE(cudnnSetDropoutDescriptor(
            dropoutDesc_,
            cudnn_wrapper_.inline_cudnn_handle(),
            dropout_param,
            dropoutStates->template mutable_data<T>(),
            stateSize,
            OperatorBase::GetSingleArgument<int>("seed", 0)));
      }
    }
  }

  // RNN setup
  {
    CUDNN_ENFORCE(cudnnSetRNNDescriptor_v6(
        cudnn_wrapper_.inline_cudnn_handle(),
        rnnDesc_,
        hiddenSize,
        numLayers,
        dropoutDesc_,
        rnnInput,
        rnnDirection,
        rnnMode,
        CUDNN_RNN_ALGO_STANDARD, // TODO: verify correctness / efficiency.
        cudnnTypeWrapper<T>::type));
  }
  // X setup
  {
    xDesc_.reset(new detail::TensorDescriptors<T>(
        seqLength,
        // Third dimension is unused
        {batchSize, inputDim, 1},
        // Fully-packed
        {inputDim, 1, 1}));
  }
  // Y setup
  {
    yDesc_.reset(new detail::TensorDescriptors<T>(
        seqLength,
        // Third dimension is unused
        {batchSize, hiddenSize * numDirections, 1},
        // Fully-packed
        {numDirections * hiddenSize, 1, 1}));

    if (output) {
      output->Resize(std::vector<int>{seqLength, batchSize, outputDim});
    }
  }

  // Hidden/Cell setup
  {
    const std::array<int, 3> dim{
        numLayers * numDirections, batchSize, hiddenSize};
    const std::array<int, 3> stride{batchSize * hiddenSize, hiddenSize, 1};
    CUDNN_ENFORCE(cudnnSetTensorNdDescriptor(
        hxDesc_, cudnnTypeWrapper<T>::type, 3, dim.data(), stride.data()));
    cxDesc_ = hxDesc_;
    hyDesc_ = hxDesc_;
    cyDesc_ = hxDesc_;

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
    size_t weightsSize;
    CUDNN_ENFORCE(cudnnGetRNNParamsSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        rnnDesc_,
        xDesc_->descs()[0],
        &weightsSize,
        cudnnTypeWrapper<T>::type));
    const std::array<int, 3> dims{
        static_cast<int>(
            weightsSize / 4 /* sizeof(T) - workaround clang bug */),
        1,
        1};
    CUDNN_ENFORCE(cudnnSetFilterNdDescriptor(
        wDesc_, cudnnTypeWrapper<T>::type, CUDNN_TENSOR_NCHW, 3, dims.data()));
  }

  // RNN workspace size
  {
    CUDNN_ENFORCE(cudnnGetRNNWorkspaceSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        rnnDesc_,
        seqLength,
        xDesc_->descs(),
        &cudnnWsNbytes_));
  }
}

template <typename T>
bool RecurrentOp<T>::RunOnDevice() {
  const int seqLength = Input(INPUT).dim32(0);
  if (Input(INPUT).sizes() != cachedInputDims_) {
    initialize(
        Input(INPUT),
        Output(DROPOUT_STATES),
        Output(OUTPUT),
        Output(HIDDEN_OUTPUT),
        Output(CELL_OUTPUT));
    cachedInputDims_ = Input(INPUT).sizes().vec();
  }

  // Validation checks
  size_t weightsSize;
  CUDNN_ENFORCE(cudnnGetRNNParamsSize(
      cudnn_wrapper_.inline_cudnn_handle(),
      rnnDesc_,
      xDesc_->descs()[0],
      &weightsSize,
      cudnnTypeWrapper<T>::type));
  CAFFE_ENFORCE_EQ(Input(WEIGHT).nbytes(), weightsSize);

  // Training reserve size
  CUDNN_ENFORCE(cudnnGetRNNTrainingReserveSize(
      cudnn_wrapper_.inline_cudnn_handle(),
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
    cudnn_wrapper_.with_cudnn_state(0, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnRNNForwardInference(
          state->cudnn_handle(),
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
          state->workspace().get(cudnnWsNbytes_),
          cudnnWsNbytes_));
    });
  } else {
    cudnn_wrapper_.with_cudnn_state(0, [&](CuDNNState* state) {
      CUDNN_ENFORCE(cudnnRNNForwardTraining(
          state->cudnn_handle(),
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
          state->workspace().get(cudnnWsNbytes_),
          cudnnWsNbytes_,
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
    initialize(Input(INPUT), Output(DROPOUT_STATES));
    cachedInputDims_ = Input(INPUT).sizes().vec();
  }
  CUDNN_ENFORCE(cudnnGetRNNTrainingReserveSize(
      cudnn_wrapper_.inline_cudnn_handle(),
      rnnDesc_,
      seqLength,
      xDesc_->descs(),
      &reserveNbytes_));
  CAFFE_ENFORCE_EQ(reserveNbytes_, Input(RNN_SCRATCH).nbytes());
  Output(GRAD_INPUT)->ResizeLike(Input(INPUT));
  Output(GRAD_HIDDEN_INPUT)->ResizeLike(Input(HIDDEN_INPUT));
  Output(GRAD_CELL_INPUT)->ResizeLike(Input(CELL_INPUT));

  Output(GRAD_WEIGHT)->ResizeLike(Input(WEIGHT));
  math::Set<T, CUDAContext>(
      Output(GRAD_WEIGHT)->numel(),
      0.0,
      Output(GRAD_WEIGHT)->template mutable_data<T>(),
      &context_);

#if CUDNN_VERSION_MIN(6,0,0)
  auto * reserve = Output(RNN_SCRATCH_OUT)->template mutable_data<T>();
#else
  const auto * reserve = Output(RNN_SCRATCH_OUT)->template data<T>();
#endif
  auto InputData = [this](int i) { return this->Input(i).template data<T>(); };
  auto OutputData = [this](int i) {
    return this->Output(i)->template mutable_data<T>();
  };

  cudnn_wrapper_.with_cudnn_state(0, [&](CuDNNState* state) {
    CUDNN_ENFORCE(cudnnRNNBackwardData(
        state->cudnn_handle(),
        rnnDesc_,
        seqLength,
        yDesc_->descs(),
        InputData(OUTPUT), // Input(OUTPUT).template data<T>(),
        yDesc_->descs(),
        InputData(GRAD_OUTPUT), // Input(GRAD_OUTPUT).template data<T>(),
        hyDesc_,
        // Note: like CNTK, ignore these gradient inputs. t16675365 to
        // reconsider.
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
        state->workspace().get(cudnnWsNbytes_),
        cudnnWsNbytes_,
        reserve,
        reserveNbytes_));
    CUDNN_ENFORCE(cudnnRNNBackwardWeights(
        state->cudnn_handle(),
        rnnDesc_,
        seqLength,
        xDesc_->descs(),
        InputData(INPUT), // Input(INPUT).template data<T>(),
        hxDesc_,
        InputData(HIDDEN_INPUT), // Input(HIDDEN_INPUT).template data<T>(),
        yDesc_->descs(),
        InputData(OUTPUT), // Input(OUTPUT).template data<T>(),
        state->workspace().get(cudnnWsNbytes_),
        cudnnWsNbytes_,
        wDesc_,
        OutputData(
            GRAD_WEIGHT), // Output(GRAD_WEIGHT)->template mutable_data<T>(),
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
    CUDNN_ENFORCE(cudnnGetRNNParamsSize(
        cudnn_wrapper_.inline_cudnn_handle(),
        rnnDesc_,
        xDesc_->descs()[0],
        &paramsSize,
        cudnnTypeWrapper<T>::type));

    CAFFE_ENFORCE_EQ(
        paramsSize / 4, Input(1).numel(), "Incorrect weight initialization");
  }

  int layer = OperatorBase::GetSingleArgument<int>("layer", 0);
  std::string param_type =
      OperatorBase::GetSingleArgument<string>("param_type", "");
  std::string input_type =
      OperatorBase::GetSingleArgument<string>("input_type", "");

  // Mapping to CUDNN constants
  std::map<string, int> weight_constants = {{"input_gate_w", 0},
                                            {"forget_gate_w", 1},
                                            {"cell_w", 2},
                                            {"output_gate_w", 3}};
  std::map<string, int> bias_constants = {{"input_gate_b", 0},
                                          {"forget_gate_b", 1},
                                          {"cell_b", 2},
                                          {"output_gate_b", 3}};
  if (bias_constants.find(param_type) != bias_constants.end()) {
    int param_id = bias_constants[param_type] + 4 * (input_type == "recurrent");

    cudnnFilterDescriptor_t biasDesc;
    CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&biasDesc));
    void* bias;

    CUDNN_ENFORCE(cudnnGetRNNLinLayerBiasParams(
        cudnn_wrapper_.inline_cudnn_handle(),
        rnnDesc_,
        layer,
        xDesc_->descs()[0],
        wDesc_,
        Input(1).template data<T>(),
        param_id, // Forget gate bias for recurrent input
        biasDesc,
        &bias));
    int numBiasDims;
    std::vector<int> biasDims(3);
    cudnnDataType_t dt;
    cudnnTensorFormat_t tf;
    // For some reason, the CuDNN Bias tensor is 3 dimensional
    CUDNN_ENFORCE(cudnnGetFilterNdDescriptor(
        biasDesc, 3, &dt, &tf, &numBiasDims, biasDims.data()));
    CAFFE_ENFORCE_EQ(numBiasDims, 3);

    if (mode == SET_PARAM) {
      CAFFE_ENFORCE_EQ(
          biasDims[0] * biasDims[1] * biasDims[2], Input(2).numel());
      this->context_.template CopySameDevice<T>(
          biasDims[0] * biasDims[1] * biasDims[2],
          Input(2).template data<T>(),
          static_cast<T*>(bias));
    } else {
      Output(0)->Resize(biasDims);
      this->context_.template CopySameDevice<T>(
          biasDims[0] * biasDims[1] * biasDims[2],
          static_cast<T*>(bias),
          Output(0)->template mutable_data<T>());
    }
  } else if (weight_constants.find(param_type) != weight_constants.end()) {
    int param_id =
        weight_constants[param_type] + 4 * (input_type == "recurrent");
    cudnnFilterDescriptor_t matrixParamDesc;
    CUDNN_ENFORCE(cudnnCreateFilterDescriptor(&matrixParamDesc));
    void* pmatrix;
    CUDNN_ENFORCE(cudnnGetRNNLinLayerMatrixParams(
        cudnn_wrapper_.inline_cudnn_handle(),
        rnnDesc_,
        layer,
        xDesc_->descs()[0],
        wDesc_,
        Input(1).template data<T>(),
        param_id, // Forget gate bias for recurrent input
        matrixParamDesc,
        &pmatrix));
    int numDims;
    std::vector<int> matDims(3);
    cudnnDataType_t dt;
    cudnnTensorFormat_t tf;

    CUDNN_ENFORCE(cudnnGetFilterNdDescriptor(
        matrixParamDesc, 3, &dt, &tf, &numDims, matDims.data()));
    CAFFE_ENFORCE_EQ(numDims, 3);
    if (mode == SET_PARAM) {
      CAFFE_ENFORCE_EQ(matDims[0] * matDims[1] * matDims[2], Input(2).numel());
      this->context_.template CopySameDevice<T>(
          matDims[0] * matDims[1] * matDims[2],
          Input(2).template data<T>(),
          static_cast<T*>(pmatrix));
    } else {
      Output(0)->Resize(matDims);
      this->context_.template CopySameDevice<T>(
          matDims[0] * matDims[1] * matDims[2],
          static_cast<T*>(pmatrix),
          Output(0)->template mutable_data<T>());
    }
  } else {
    CAFFE_ENFORCE(false, "Unknown param type:", param_type);
  }

  return true;
}

REGISTER_CUDNN_OPERATOR(Recurrent, RecurrentOp<float>);
OPERATOR_SCHEMA(Recurrent).NumInputs(4).NumOutputs(5).SetDoc(R"DOC(

Recurrent wraps the CuDNN R5 RNN implementation. See the CuDNN R5
documentation for more information.

In general, the implementation takes an input (TxNxD) tensor, the
hidden state input (NxD), the cell input (NxD), and a weight tensor
(effectively an opaque blob, where the size and layout is dictated by
CuDNN).

The outputs are the output (again, TxNxD), the final hidden/cell
states (NxD). These can be reset (at sequence boundaries across
minibatches) by multiplying by zero.

The CuDNN arguments (hidden_size, bidirectional, num_layers, rnn_mode,
input_mode) are passed directly through to CuDNN.

)DOC");
REGISTER_CUDNN_OPERATOR(RecurrentGradient, RecurrentGradientOp<float>);
OPERATOR_SCHEMA(RecurrentGradient)
    .NumInputs(7)
    .NumOutputs(6)
    .AllowInplace({{4, 5}});

REGISTER_CUDNN_OPERATOR(
    RecurrentParamSet,
    RecurrentParamAccessOp<float, SET_PARAM>);
OPERATOR_SCHEMA(RecurrentParamSet)
    .NumInputs(3)
    .NumOutputs(1)
    .EnforceInplace({{1, 0}})
    .SetDoc("Set individual parameters of a recurrent net.")
    .Arg("param_type", R"DOC(Type of param to be set:
                  "input_gate_w", "forget_gate_w", "cell_w", "output_gate_w"
                  "input_gate_b", "forget_gate_b", "cell_b", "output_gate_b"
                  )DOC")
    .Arg("input_type", "'recurrent' or 'input'")
    .Arg("layer", "layer index (starting from 0)")
    .Input(0, "input", R"DOC(Input blob. Needed for inferring the shapes.
                        A dummy tensor matching the input shape is ok.)DOC")
    .Input(1, "all_params", "Blob holding all the parameters")
    .Input(2, "param", "Values for the specified parameter")
    .Output(
        0,
        "all_params",
        "Blob holding all the parameters (same as input(1))");

REGISTER_CUDNN_OPERATOR(
    RecurrentParamGet,
    RecurrentParamAccessOp<float, GET_PARAM>);
OPERATOR_SCHEMA(RecurrentParamGet)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc("Retrieve individual parameters of a recurrent net op.")
    .Arg("param_type", R"DOC(Type of param to be set:
                  "input_gate_w", "forget_gate_w", "cell_w", "output_gate_w"
                  "input_gate_b", "forget_gate_b", "cell_b", "output_gate_b"
                  )DOC")
    .Arg("input_type", "'recurrent' or 'input'")
    .Arg("layer", "layer index (starting from 0)")
    .Input(0, "input", R"DOC(Input blob. Needed for inferring the shapes.
                        A dummy tensor matching the input shape is ok.)DOC")
    .Input(1, "all_params", "Blob holding all the parameters")
    .Output(0, "param", "Blob holding the requested values");

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
