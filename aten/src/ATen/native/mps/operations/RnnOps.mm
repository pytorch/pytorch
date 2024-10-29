//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/MemoryOverlap.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/RNN.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/_lstm_mps_native.h>
#include <ATen/ops/lstm_mps_backward_native.h>
#import <MetalPerformanceShadersGraph/MPSGraphRNNOps.h>

namespace at::native {

static std::vector<long long> getTensorShape(MPSGraphTensor* mpsTensor) {
  std::vector<long long> output_dimensions = {};
  auto dims = mpsTensor.shape;
  for (NSUInteger i = 0; i < [dims count]; i++) {
    output_dimensions.push_back([dims[i] intValue]);
  }
  return output_dimensions;
}

/**
 * Accepts tensors in Pytorch API format and returns tensors in MPS API format
 * @return tuple of tensors to use with MPS API in order:
 * stateTensor, cellStateTensor, recurrentWeight, inputWeight, biasTensor
 */
static std::tuple<MPSGraphTensor*, MPSGraphTensor*, MPSGraphTensor*, MPSGraphTensor*, MPSGraphTensor*>
getMPSTensorsFromPytorchTensors(MPSGraph* mpsGraph,
                                MPSGraphTensor* stateTensor,
                                MPSGraphTensor* cellStateTensor,
                                NSMutableArray<MPSGraphTensor*>* recurrentKernelWeightsList,
                                NSMutableArray<MPSGraphTensor*>* kernelWeightsList,
                                NSMutableArray<MPSGraphTensor*>* kernelBiasList,
                                NSMutableArray<MPSGraphTensor*>* recurrentBiasList,
                                bool has_biases,
                                bool bidirectional,
                                size_t layer_no) {
  MPSGraphTensor* biasTensor_ = nil;
  MPSGraphTensor *stateTensor_ = nil, *cellStateTensor_ = nil;
  MPSGraphTensor *recurrentWeight_ = nil, *inputWeight_ = nil;

  if (bidirectional) {
    stateTensor_ = [mpsGraph sliceTensor:stateTensor dimension:0 start:layer_no * 2 length:2 name:nil];
    // [2, N, H] -> [N, 2, H]
    stateTensor_ = [mpsGraph transposeTensor:stateTensor_ dimension:0 withDimension:1 name:nil];
    // [N, 2, H] -> [N, 2 * H]
    stateTensor_ = [mpsGraph flatten2DTensor:stateTensor_ axis:1 name:nil];
    cellStateTensor_ = [mpsGraph sliceTensor:cellStateTensor dimension:0 start:layer_no * 2 length:2 name:nil];
    cellStateTensor_ = [mpsGraph transposeTensor:cellStateTensor_ dimension:0 withDimension:1 name:nil];
    cellStateTensor_ = [mpsGraph flatten2DTensor:cellStateTensor_ axis:1 name:nil];

    recurrentWeight_ = [mpsGraph
        concatTensor:[mpsGraph expandDimsOfTensor:recurrentKernelWeightsList[layer_no * 2] axis:0 name:nil]
          withTensor:[mpsGraph expandDimsOfTensor:recurrentKernelWeightsList[layer_no * 2 + 1] axis:0 name:nil]
           dimension:0
                name:nil];
    inputWeight_ = [mpsGraph concatTensor:kernelWeightsList[layer_no * 2]
                               withTensor:kernelWeightsList[layer_no * 2 + 1]
                                dimension:0
                                     name:nil];
    if (has_biases) {
      auto biasTensorFwd_ = [mpsGraph additionWithPrimaryTensor:kernelBiasList[layer_no * 2]
                                                secondaryTensor:recurrentBiasList[layer_no * 2]
                                                           name:nil];
      auto biasTensorBack_ = [mpsGraph additionWithPrimaryTensor:kernelBiasList[layer_no * 2 + 1]
                                                 secondaryTensor:recurrentBiasList[layer_no * 2 + 1]
                                                            name:nil];

      biasTensor_ = [mpsGraph concatTensor:biasTensorFwd_ withTensor:biasTensorBack_ dimension:0 name:nil];
    }
  } else {
    stateTensor_ = [mpsGraph sliceTensor:stateTensor dimension:0 start:layer_no length:1 name:nil];
    cellStateTensor_ = [mpsGraph sliceTensor:cellStateTensor dimension:0 start:layer_no length:1 name:nil];
    recurrentWeight_ = recurrentKernelWeightsList[layer_no];
    inputWeight_ = kernelWeightsList[layer_no];
    if (has_biases) {
      biasTensor_ = [mpsGraph additionWithPrimaryTensor:kernelBiasList[layer_no]
                                        secondaryTensor:recurrentBiasList[layer_no]
                                                   name:nil];
    }
  }
  return std::make_tuple(stateTensor_, cellStateTensor_, recurrentWeight_, inputWeight_, biasTensor_);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> _lstm_mps(const Tensor& input,
                                                                     TensorList hx,
                                                                     TensorList params,
                                                                     bool has_biases,
                                                                     int64_t num_layers,
                                                                     double dropout_p,
                                                                     bool train,
                                                                     bool bidirectional,
                                                                     bool batch_first) {
  using namespace mps;

  // Projections are not currently supported, raise an error if needed
  bool has_projections = (hx[0].size(2) != hx[1].size(2));
  if (has_projections) {
    TORCH_CHECK(false, "LSTM with projections is not currently supported with MPS.");
  }

  std::vector<Tensor> kernel_weights;
  std::vector<Tensor> recurrent_kernel_weights;
  std::vector<Tensor> biases;
  std::vector<Tensor> recurrent_biases;

  const int64_t total_layers = num_layers * (bidirectional ? 2 : 1);

  for (const auto i : c10::irange(total_layers)) {
    const int stride = (has_biases ? 4 : 2);
    kernel_weights.push_back(params[i * stride]);
    recurrent_kernel_weights.push_back(params[i * stride + 1]);

    if (has_biases) {
      biases.push_back(params[i * stride + 2]);
      recurrent_biases.push_back(params[i * stride + 3]);
    }
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    std::vector<MPSGraphTensor*> inputTensors_;
    std::vector<MPSGraphTensor*> outputTensors_;
    NSMutableArray<MPSGraphTensor*>* kernelWeightsList_ = nil;
    NSMutableArray<MPSGraphTensor*>* recurrentKernelWeightsList_ = nil;
    NSMutableArray<MPSGraphTensor*>* biasList_ = nil;
    NSMutableArray<MPSGraphTensor*>* recurrentBiasList_ = nil;
  };

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    string key = "lstm_" + getTensorsStringKey({input, hx[0], hx[1]}) + getMPSTypeString(input) + "_num_layers_" +
        std::to_string(num_layers) + "_bidirectional_" + std::to_string(bidirectional) + "_has_biases_" +
        std::to_string(has_biases) + "_dropout_" + std::to_string(dropout_p) + "_batch_first_" +
        std::to_string(batch_first);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      NSMutableArray<MPSGraphTensor*>* kernelWeightsList = [[NSMutableArray alloc] initWithCapacity:params.size()];
      NSMutableArray<MPSGraphTensor*>* recurrentKernelWeightsList =
          [[NSMutableArray alloc] initWithCapacity:params.size()];
      NSMutableArray<MPSGraphTensor*>* kernelBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];
      NSMutableArray<MPSGraphTensor*>* recurrentBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];
      NSMutableArray<MPSGraphTensor*>* layersOutputsList = [[NSMutableArray alloc] initWithCapacity:num_layers];

      for (const auto i : c10::irange(total_layers)) {
        [kernelWeightsList
            addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(kernel_weights[i]))];
        [recurrentKernelWeightsList
            addObject:mpsGraphRankedPlaceHolder(
                          mpsGraph, getMPSDataType(input), getMPSShape(recurrent_kernel_weights[i]))];
        if (has_biases) {
          [kernelBiasList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(biases[i]))];
          [recurrentBiasList
              addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(recurrent_biases[i]))];
        }
      }

      MPSGraphLSTMDescriptor* opDesc = [MPSGraphLSTMDescriptor descriptor];
      opDesc.training = true;
      opDesc.bidirectional = bidirectional;
      opDesc.produceCell = true;

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(input));
      MPSGraphTensor* stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(hx[0]));
      MPSGraphTensor* cellStateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(hx[1]));
      std::vector<MPSGraphTensor*> inputTensors = {
          inputTensor,
          stateTensor,
          cellStateTensor,
      };

      if (batch_first) {
        inputTensor = [mpsGraph transposeTensor:inputTensor dimension:0 withDimension:1 name:nil];
      }

      MPSGraphTensor* inputTensor_ = inputTensor;
      NSArray<MPSGraphTensor*>* outputs = nil;
      NSMutableArray<MPSGraphTensor*>* outputStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      NSMutableArray<MPSGraphTensor*>* outputCellStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      NSMutableArray<MPSGraphTensor*>* outputZStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      NSMutableArray<MPSGraphTensor*>* outputCellStateFwdArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      for (int i = 0; i < num_layers; i++) {
        auto tensorsData = getMPSTensorsFromPytorchTensors(mpsGraph,
                                                           stateTensor,
                                                           cellStateTensor,
                                                           recurrentKernelWeightsList,
                                                           kernelWeightsList,
                                                           kernelBiasList,
                                                           recurrentBiasList,
                                                           has_biases,
                                                           bidirectional,
                                                           i);
        MPSGraphTensor *stateTensor_ = std::get<0>(tensorsData), *cellStateTensor_ = std::get<1>(tensorsData);
        MPSGraphTensor *recurrentWeight_ = std::get<2>(tensorsData), *inputWeight_ = std::get<3>(tensorsData);
        MPSGraphTensor* biasTensor_ = std::get<4>(tensorsData);

        outputs = [mpsGraph LSTMWithSourceTensor:inputTensor_
                                 recurrentWeight:recurrentWeight_
                                     inputWeight:inputWeight_
                                            bias:biasTensor_
                                       initState:stateTensor_
                                        initCell:cellStateTensor_
                                      descriptor:opDesc
                                            name:nil];

        inputTensor_ = [outputs objectAtIndex:0];
        // no need to keep the final layer output copy as it is
        // returned anyway and not used in backprop
        if (i != num_layers - 1) {
          [layersOutputsList addObject:[mpsGraph expandDimsOfTensor:inputTensor_ axis:0 name:nil]];
        }
        if (dropout_p > 0.0 && train && (i != num_layers - 1)) {
          inputTensor_ = [mpsGraph dropoutTensor:inputTensor_ rate:dropout_p name:nil];
        }

        if (bidirectional) {
          // [1, N, 2 * H]
          auto stateLastT = [mpsGraph sliceTensor:[outputs objectAtIndex:0] dimension:0 start:-1 length:1 name:nil];
          auto stateFirstT = [mpsGraph sliceTensor:[outputs objectAtIndex:0] dimension:0 start:0 length:1 name:nil];
          // [1, N, H] ([1, N, 0:H])
          auto stateForward = [mpsGraph sliceTensor:stateLastT dimension:-1 start:0 length:hx[0].sizes()[2] name:nil];
          // [1, N, H] ([1, N, H:2H])
          auto stateBack = [mpsGraph sliceTensor:stateFirstT
                                       dimension:-1
                                           start:hx[0].sizes()[2]
                                          length:hx[0].sizes()[2]
                                            name:nil];
          [outputStateArray addObject:stateForward];
          [outputStateArray addObject:stateBack];

          auto cellStateLastT = [mpsGraph sliceTensor:[outputs objectAtIndex:1] dimension:0 start:-1 length:1 name:nil];
          auto cellStateFirstT = [mpsGraph sliceTensor:[outputs objectAtIndex:1] dimension:0 start:0 length:1 name:nil];
          auto cellStateForward = [mpsGraph sliceTensor:cellStateLastT
                                              dimension:-1
                                                  start:0
                                                 length:hx[1].sizes()[2]
                                                   name:nil];
          auto cellStateBack = [mpsGraph sliceTensor:cellStateFirstT
                                           dimension:-1
                                               start:hx[1].sizes()[2]
                                              length:hx[1].sizes()[2]
                                                name:nil];
          [outputCellStateArray addObject:cellStateForward];
          [outputCellStateArray addObject:cellStateBack];
        } else {
          [outputStateArray addObject:[mpsGraph sliceTensor:[outputs objectAtIndex:0]
                                                  dimension:0
                                                      start:-1
                                                     length:1
                                                       name:nil]];
          [outputCellStateArray addObject:[mpsGraph sliceTensor:[outputs objectAtIndex:1]
                                                      dimension:0
                                                          start:-1
                                                         length:1
                                                           name:nil]];
        }
        [outputCellStateFwdArray addObject:[mpsGraph expandDimsOfTensor:[outputs objectAtIndex:1] axis:0 name:nil]];
        [outputZStateArray addObject:[mpsGraph expandDimsOfTensor:[outputs objectAtIndex:2] axis:0 name:nil]];
      }

      MPSGraphTensor* outputTensor = inputTensor_;
      if (batch_first) {
        outputTensor = [mpsGraph transposeTensor:outputTensor dimension:0 withDimension:1 name:nil];
      }
      MPSGraphTensor* outputStates = [mpsGraph concatTensors:outputStateArray dimension:0 name:nil];
      MPSGraphTensor* outputCellStates = [mpsGraph concatTensors:outputCellStateArray dimension:0 name:nil];
      MPSGraphTensor* outputZStates = [mpsGraph concatTensors:outputZStateArray dimension:0 name:nil];
      MPSGraphTensor* outputCellStatesFwd = [mpsGraph concatTensors:outputCellStateFwdArray dimension:0 name:nil];
      MPSGraphTensor* layersOutputs =
          (num_layers > 1) ? [mpsGraph concatTensors:layersOutputsList dimension:0 name:nil] : nil;

      std::vector<MPSGraphTensor*> outputTensors = {
          outputTensor, outputStates, outputCellStates, outputZStates, outputCellStatesFwd, layersOutputs};
      newCachedGraph->inputTensors_ = inputTensors;
      newCachedGraph->outputTensors_ = outputTensors;
      newCachedGraph->kernelWeightsList_ = kernelWeightsList;
      newCachedGraph->recurrentKernelWeightsList_ = recurrentKernelWeightsList;
      newCachedGraph->biasList_ = kernelBiasList;
      newCachedGraph->recurrentBiasList_ = recurrentBiasList;
    });

    NSMutableArray<MPSGraphTensor*>* kernelWeightsList = cachedGraph->kernelWeightsList_;
    NSMutableArray<MPSGraphTensor*>* recurrentKernelWeightsList = cachedGraph->recurrentKernelWeightsList_;
    NSMutableArray<MPSGraphTensor*>* biasList = cachedGraph->biasList_;
    NSMutableArray<MPSGraphTensor*>* recurrentBiasList = cachedGraph->recurrentBiasList_;

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
    for (const auto i : c10::irange(total_layers)) {
      Placeholder kernelWeight = Placeholder([kernelWeightsList objectAtIndex:i], kernel_weights[i]);
      Placeholder recurrentKernelWeight =
          Placeholder([recurrentKernelWeightsList objectAtIndex:i], recurrent_kernel_weights[i]);
      [feeds setObject:kernelWeight.getMPSGraphTensorData() forKey:kernelWeight.getMPSGraphTensor()];
      [feeds setObject:recurrentKernelWeight.getMPSGraphTensorData() forKey:recurrentKernelWeight.getMPSGraphTensor()];
      if (has_biases) {
        Placeholder bias = Placeholder([biasList objectAtIndex:i], biases[i]);
        Placeholder recurrentBias = Placeholder([recurrentBiasList objectAtIndex:i], recurrent_biases[i]);
        [feeds setObject:bias.getMPSGraphTensorData() forKey:bias.getMPSGraphTensor()];
        [feeds setObject:recurrentBias.getMPSGraphTensorData() forKey:recurrentBias.getMPSGraphTensor()];
      }
    }
    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensors_[0], input);
    Placeholder selfState = Placeholder(cachedGraph->inputTensors_[1], hx[0]);
    Placeholder selfCellState = Placeholder(cachedGraph->inputTensors_[2], hx[1]);
    [feeds setObject:selfPlaceholder.getMPSGraphTensorData() forKey:selfPlaceholder.getMPSGraphTensor()];
    [feeds setObject:selfState.getMPSGraphTensorData() forKey:selfState.getMPSGraphTensor()];
    [feeds setObject:selfCellState.getMPSGraphTensorData() forKey:selfCellState.getMPSGraphTensor()];

    auto dims = getTensorShape(cachedGraph->outputTensors_[0]);
    Tensor output = at::empty(IntArrayRef(dims), input.options());
    Tensor hy = at::empty_like(hx[0], input.options());
    Tensor cy = at::empty_like(hx[1], input.options());
    Tensor zState = at::empty(IntArrayRef(getTensorShape(cachedGraph->outputTensors_[3])), input.options());
    Tensor cellStateFwd = at::empty(IntArrayRef(getTensorShape(cachedGraph->outputTensors_[4])), input.options());
    Tensor layerOutputs = (num_layers > 1)
        ? at::empty(IntArrayRef(getTensorShape(cachedGraph->outputTensors_[5])), input.options())
        : at::empty({1}, input.options()); // not used if num_layers == 1

    Placeholder outputPlaceholder0 = Placeholder(cachedGraph->outputTensors_[0], output);
    Placeholder outputPlaceholder1 = Placeholder(cachedGraph->outputTensors_[1], hy);
    Placeholder outputPlaceholder2 = Placeholder(cachedGraph->outputTensors_[2], cy);
    Placeholder outputPlaceholder3 = Placeholder(cachedGraph->outputTensors_[3], zState);
    Placeholder outputPlaceholder4 = Placeholder(cachedGraph->outputTensors_[4], cellStateFwd);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = [@{
      outputPlaceholder0.getMPSGraphTensor() : outputPlaceholder0.getMPSGraphTensorData(),
      outputPlaceholder1.getMPSGraphTensor() : outputPlaceholder1.getMPSGraphTensorData(),
      outputPlaceholder2.getMPSGraphTensor() : outputPlaceholder2.getMPSGraphTensorData(),
      outputPlaceholder3.getMPSGraphTensor() : outputPlaceholder3.getMPSGraphTensorData(),
      outputPlaceholder4.getMPSGraphTensor() : outputPlaceholder4.getMPSGraphTensorData(),
    } mutableCopy];

    if (num_layers > 1) {
      Placeholder outputPlaceholder5 = Placeholder(cachedGraph->outputTensors_[5], layerOutputs);
      [results setObject:outputPlaceholder5.getMPSGraphTensorData() forKey:outputPlaceholder5.getMPSGraphTensor()];
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    return std::make_tuple(output, hy, cy, zState, cellStateFwd, layerOutputs);
  }
}

std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> lstm_mps_backward(const std::optional<Tensor>& grad_y_opt,
                                                                               const std::optional<Tensor>& grad_hy_opt,
                                                                               const std::optional<Tensor>& grad_cy_opt,
                                                                               const Tensor& z_state,
                                                                               const Tensor& cell_state_fwd,
                                                                               const Tensor& input,
                                                                               const Tensor& layersOutputs,
                                                                               TensorList hx,
                                                                               TensorList params,
                                                                               bool has_biases,
                                                                               int64_t num_layers,
                                                                               double dropout_p,
                                                                               bool train,
                                                                               bool bidirectional,
                                                                               bool batch_first) {
  using namespace mps;
  bool is_macos_14_4_or_newer = is_macos_13_or_newer(MacOSVersion::MACOS_VER_14_4_PLUS);

  const Tensor& grad_y_r = grad_y_opt.value_or(Tensor());
  const Tensor& grad_hy_r = grad_hy_opt.value_or(Tensor());
  const Tensor& grad_cy_r = grad_cy_opt.value_or(Tensor());
  const auto grad_hy = grad_hy_r.defined() ? grad_hy_r : at::zeros_like(hx[0], input.options());
  const auto grad_cy = grad_cy_r.defined() ? grad_cy_r : at::zeros_like(hx[1], input.options());

  const auto hidden_size = hx[0].sizes()[2];
  const auto batch_size = hx[0].sizes()[1];
  const auto seq_len = input.sizes()[batch_first ? 1 : 0];
  const auto grad_y = grad_y_r.defined() ? grad_y_r
                                         : at::zeros({batch_first ? batch_size : seq_len,
                                                      batch_first ? seq_len : batch_size,
                                                      hidden_size * (bidirectional ? 2 : 1)},
                                                     input.options());

  std::vector<Tensor> kernel_weights;
  std::vector<Tensor> recurrent_kernel_weights;
  std::vector<Tensor> biases;
  std::vector<Tensor> recurrent_biases;

  const int64_t total_layers = num_layers * (bidirectional ? 2 : 1);

  for (const auto i : c10::irange(total_layers)) {
    const int stride = (has_biases ? 4 : 2);
    kernel_weights.push_back(params[i * stride]);
    recurrent_kernel_weights.push_back(params[i * stride + 1]);
    if (has_biases) {
      biases.push_back(params[i * stride + 2]);
      recurrent_biases.push_back(params[i * stride + 3]);
    }
  }

  struct CachedGraph : public MPSCachedGraph {
    CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    std::vector<MPSGraphTensor*> inputTensors_;
    NSMutableArray<MPSGraphTensor*>* kernelWeightsList_ = nil;
    NSMutableArray<MPSGraphTensor*>* recurrentKernelWeightsList_ = nil;
    NSMutableArray<MPSGraphTensor*>* biasList_ = nil;
    NSMutableArray<MPSGraphTensor*>* recurrentBiasList_ = nil;
    NSMutableArray<MPSGraphTensor*>* gradRecWeights_ = nil;
    NSMutableArray<MPSGraphTensor*>* gradWeights_ = nil;
    NSMutableArray<MPSGraphTensor*>* gradBias_ = nil;
    MPSGraphTensor* gradOutput_ = nil;
    MPSGraphTensor* gradState_ = nil;
    MPSGraphTensor* gradCellState_ = nil;
  };

  // Get stream
  MPSStream* stream = getCurrentMPSStream();
  @autoreleasepool {
    string key = "lstm_backward_" + getTensorsStringKey({input, z_state, cell_state_fwd, grad_y, grad_cy, grad_hy}) +
        getMPSTypeString(input) + "_num_layers_" + std::to_string(num_layers) + "_bidirectional_" +
        std::to_string(bidirectional) + "_has_biases_" + std::to_string(has_biases) + "_batch_first_" +
        std::to_string(batch_first);
    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      NSMutableArray<MPSGraphTensor*>* kernelWeightsList = [[NSMutableArray alloc] initWithCapacity:params.size()];
      NSMutableArray<MPSGraphTensor*>* recurrentKernelWeightsList =
          [[NSMutableArray alloc] initWithCapacity:params.size()];
      NSMutableArray<MPSGraphTensor*>* kernelBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];
      NSMutableArray<MPSGraphTensor*>* recurrentBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];

      for (const auto i : c10::irange(total_layers)) {
        [kernelWeightsList
            addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(kernel_weights[i]))];
        [recurrentKernelWeightsList
            addObject:mpsGraphRankedPlaceHolder(
                          mpsGraph, getMPSDataType(input), getMPSShape(recurrent_kernel_weights[i]))];
        if (has_biases) {
          [kernelBiasList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(biases[i]))];
          [recurrentBiasList
              addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(recurrent_biases[i]))];
        }
      }

      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(input));
      MPSGraphTensor* stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(hx[0]));
      MPSGraphTensor* cellStateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(hx[1]));
      MPSGraphTensor* zStateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input), getMPSShape(z_state));
      MPSGraphTensor* gradientTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_y), getMPSShape(grad_y));
      MPSGraphTensor* gradientCyTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_cy), getMPSShape(grad_cy));
      MPSGraphTensor* gradientHyTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_hy), getMPSShape(grad_hy));
      MPSGraphTensor* cellStateFwdTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(cell_state_fwd), getMPSShape(cell_state_fwd));
      MPSGraphTensor* layersOutputsTensor =
          mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(layersOutputs), getMPSShape(layersOutputs));

      std::vector<MPSGraphTensor*> inputs = {inputTensor,
                                             stateTensor,
                                             cellStateTensor,
                                             gradientTensor,
                                             zStateTensor,
                                             cellStateFwdTensor,
                                             gradientHyTensor,
                                             gradientCyTensor,
                                             layersOutputsTensor};

      if (batch_first) {
        inputTensor = [mpsGraph transposeTensor:inputTensor dimension:0 withDimension:1 name:nil];

        gradientTensor = [mpsGraph transposeTensor:gradientTensor dimension:0 withDimension:1 name:nil];
      }

      newCachedGraph->recurrentKernelWeightsList_ = recurrentKernelWeightsList;
      newCachedGraph->kernelWeightsList_ = kernelWeightsList;
      newCachedGraph->biasList_ = kernelBiasList;
      newCachedGraph->recurrentBiasList_ = recurrentBiasList;
      newCachedGraph->inputTensors_ = inputs;

      MPSGraphLSTMDescriptor* opDesc = [MPSGraphLSTMDescriptor descriptor];
      opDesc.training = true; // train;
      opDesc.bidirectional = bidirectional;
      opDesc.produceCell = true;

      MPSGraphTensor* gradientTensor_ = gradientTensor;

      NSArray<MPSGraphTensor*>* outputs = nil;

      NSMutableArray<MPSGraphTensor*>* gradRecWeightsArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      NSMutableArray<MPSGraphTensor*>* gradWeightsArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      NSMutableArray<MPSGraphTensor*>* gradBiasArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      NSMutableArray<MPSGraphTensor*>* gradStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
      NSMutableArray<MPSGraphTensor*>* gradCellStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];

      for (int i = num_layers - 1; i >= 0; i--) {
        MPSGraphTensor* zState = [mpsGraph sliceTensor:zStateTensor dimension:0 start:i length:1 name:nil];
        zState = [mpsGraph squeezeTensor:zState axis:0 name:nil];
        MPSGraphTensor* cellStateFwd = [mpsGraph sliceTensor:cellStateFwdTensor dimension:0 start:i length:1 name:nil];
        cellStateFwd = [mpsGraph squeezeTensor:cellStateFwd axis:0 name:nil];
        auto tensorsData = getMPSTensorsFromPytorchTensors(mpsGraph,
                                                           stateTensor,
                                                           cellStateTensor,
                                                           recurrentKernelWeightsList,
                                                           kernelWeightsList,
                                                           kernelBiasList,
                                                           recurrentBiasList,
                                                           has_biases,
                                                           bidirectional,
                                                           i);
        MPSGraphTensor *stateTensor_ = std::get<0>(tensorsData), *cellStateTensor_ = std::get<1>(tensorsData);
        MPSGraphTensor *recurrentWeight_ = std::get<2>(tensorsData), *inputWeight_ = std::get<3>(tensorsData);
        MPSGraphTensor* biasTensor_ = std::get<4>(tensorsData);

        MPSGraphTensor *gradientHyTensor_ = nil, *gradientCyTensor_ = nil;
        if (bidirectional) {
          gradientHyTensor_ = [mpsGraph sliceTensor:gradientHyTensor dimension:0 start:i * 2 length:2 name:nil];
          // [2, N, H] -> [N, 2, H]
          gradientHyTensor_ = [mpsGraph transposeTensor:gradientHyTensor_ dimension:0 withDimension:1 name:nil];
          // [N, 2, H] -> [N, 2 * H]
          gradientHyTensor_ = [mpsGraph flatten2DTensor:gradientHyTensor_ axis:1 name:nil];

          gradientCyTensor_ = [mpsGraph sliceTensor:gradientCyTensor dimension:0 start:i * 2 length:2 name:nil];
          gradientCyTensor_ = [mpsGraph transposeTensor:gradientCyTensor_ dimension:0 withDimension:1 name:nil];
          gradientCyTensor_ = [mpsGraph flatten2DTensor:gradientCyTensor_ axis:1 name:nil];
        } else {
          gradientHyTensor_ = [mpsGraph sliceTensor:gradientHyTensor dimension:0 start:i length:1 name:nil];

          gradientCyTensor_ = [mpsGraph sliceTensor:gradientCyTensor dimension:0 start:i length:1 name:nil];
        }

        MPSGraphTensor* iterationInputTensor_ = nil;
        if (i == 0) {
          iterationInputTensor_ = inputTensor;
        } else {
          iterationInputTensor_ = [mpsGraph sliceTensor:layersOutputsTensor
                                              dimension:0
                                                  // the last element in layersOutputsTensor
                                                  // contains **inputs** for the **last** layer
                                                  // and so on
                                                  start:i - num_layers
                                                 length:1
                                                   name:nil];
          if (is_macos_14_4_or_newer) {
            // Prevents shape optimization bug in kernel when num_layers > 2
            iterationInputTensor_ = [mpsGraph identityWithTensor:iterationInputTensor_ name:nil];
          }
          iterationInputTensor_ = [mpsGraph squeezeTensor:iterationInputTensor_ axis:0 name:nil];
        }

        outputs = [mpsGraph LSTMGradientsWithSourceTensor:iterationInputTensor_
                                          recurrentWeight:recurrentWeight_
                                           sourceGradient:gradientTensor_
                                                   zState:zState
                                            cellOutputFwd:cellStateFwd
                                            stateGradient:gradientHyTensor_
                                             cellGradient:gradientCyTensor_
                                              inputWeight:inputWeight_
                                                     bias:biasTensor_
                                                initState:stateTensor_
                                                 initCell:cellStateTensor_
                                                     mask:nil
                                                 peephole:nil
                                               descriptor:opDesc
                                                     name:nil];

        gradientTensor_ = [outputs objectAtIndex:0];
        if (bidirectional) {
          int outputIter = 1;
          auto gradRecWeightsBidirectional = [outputs objectAtIndex:outputIter++];
          auto gradRecWeightFwd = [mpsGraph sliceTensor:gradRecWeightsBidirectional
                                              dimension:0
                                                  start:0
                                                 length:1
                                                   name:nil];
          gradRecWeightFwd = [mpsGraph squeezeTensor:gradRecWeightFwd axis:0 name:nil];
          auto gradRecWeightBack = [mpsGraph sliceTensor:gradRecWeightsBidirectional
                                               dimension:0
                                                   start:1
                                                  length:1
                                                    name:nil];
          gradRecWeightBack = [mpsGraph squeezeTensor:gradRecWeightBack axis:0 name:nil];

          // inverse order
          [gradRecWeightsArray insertObject:gradRecWeightBack atIndex:0];
          [gradRecWeightsArray insertObject:gradRecWeightFwd atIndex:0];

          auto gradWeightsBidirectional = [outputs objectAtIndex:outputIter++];
          auto gradWeightFwd = [mpsGraph sliceTensor:gradWeightsBidirectional
                                           dimension:0
                                               start:0
                                              length:hidden_size * 4
                                                name:nil];
          auto gradWeightBack = [mpsGraph sliceTensor:gradWeightsBidirectional
                                            dimension:0
                                                start:hidden_size * 4
                                               length:hidden_size * 4
                                                 name:nil];

          [gradWeightsArray insertObject:gradWeightBack atIndex:0];
          [gradWeightsArray insertObject:gradWeightFwd atIndex:0];

          if (has_biases) {
            // has shape [1, 1, 8H] vs [8H] as should be
            // so, squeeze these two first dimensions
            auto gradBiasBidirectional = [outputs objectAtIndex:outputIter++];
            gradBiasBidirectional = [mpsGraph squeezeTensor:gradBiasBidirectional axes:@[ @0, @1 ] name:nil];
            auto gradBiasFwd = [mpsGraph sliceTensor:gradBiasBidirectional
                                           dimension:0
                                               start:0
                                              length:hidden_size * 4
                                                name:nil];
            auto gradBiasBack = [mpsGraph sliceTensor:gradBiasBidirectional
                                            dimension:0
                                                start:hidden_size * 4
                                               length:hidden_size * 4
                                                 name:nil];

            [gradBiasArray insertObject:gradBiasBack atIndex:0];
            [gradBiasArray insertObject:gradBiasFwd atIndex:0];
          }

          auto gradStateBidirectional = [outputs objectAtIndex:outputIter++];
          auto gradStateFwd = [mpsGraph sliceTensor:gradStateBidirectional
                                          dimension:1
                                              start:0
                                             length:hidden_size
                                               name:nil];
          auto gradStateBack = [mpsGraph sliceTensor:gradStateBidirectional
                                           dimension:1
                                               start:hidden_size
                                              length:hidden_size
                                                name:nil];

          [gradStateArray insertObject:[mpsGraph expandDimsOfTensor:gradStateBack axis:0 name:nil] atIndex:0];
          [gradStateArray insertObject:[mpsGraph expandDimsOfTensor:gradStateFwd axis:0 name:nil] atIndex:0];

          auto gradCellStateBidirectional = [outputs objectAtIndex:outputIter++];
          auto gradCellStateFwd = [mpsGraph sliceTensor:gradCellStateBidirectional
                                              dimension:1
                                                  start:0
                                                 length:hidden_size
                                                   name:nil];
          auto gradCellStateBack = [mpsGraph sliceTensor:gradCellStateBidirectional
                                               dimension:1
                                                   start:hidden_size
                                                  length:hidden_size
                                                    name:nil];

          [gradCellStateArray insertObject:[mpsGraph expandDimsOfTensor:gradCellStateBack axis:0 name:nil] atIndex:0];
          [gradCellStateArray insertObject:[mpsGraph expandDimsOfTensor:gradCellStateFwd axis:0 name:nil] atIndex:0];
        } else {
          int outputIter = 1;
          [gradRecWeightsArray insertObject:[outputs objectAtIndex:outputIter++] atIndex:0];
          [gradWeightsArray insertObject:[outputs objectAtIndex:outputIter++] atIndex:0];
          if (has_biases) {
            [gradBiasArray insertObject:[outputs objectAtIndex:outputIter++] atIndex:0];
          }
          [gradStateArray insertObject:[mpsGraph expandDimsOfTensor:[outputs objectAtIndex:outputIter++]
                                                               axis:0
                                                               name:nil]
                               atIndex:0];
          [gradCellStateArray insertObject:[mpsGraph expandDimsOfTensor:[outputs objectAtIndex:outputIter++]
                                                                   axis:0
                                                                   name:nil]
                                   atIndex:0];
        }
      }
      if (batch_first) {
        MPSGraphTensor* gradientTensorTransposed = [mpsGraph transposeTensor:gradientTensor_
                                                                   dimension:0
                                                               withDimension:1
                                                                        name:nil];
        newCachedGraph->gradOutput_ = gradientTensorTransposed;
      } else {
        newCachedGraph->gradOutput_ = gradientTensor_;
      }

      newCachedGraph->gradRecWeights_ = gradRecWeightsArray;
      newCachedGraph->gradWeights_ = gradWeightsArray;
      newCachedGraph->gradBias_ = gradBiasArray;
      newCachedGraph->gradState_ = [mpsGraph concatTensors:gradStateArray dimension:0 name:nil];
      newCachedGraph->gradCellState_ = [mpsGraph concatTensors:gradCellStateArray dimension:0 name:nil];
    });

    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensors_[0], input);
    Placeholder statePlaceholder = Placeholder(cachedGraph->inputTensors_[1], hx[0]);
    Placeholder cellStatePlaceholder = Placeholder(cachedGraph->inputTensors_[2], hx[1]);
    Placeholder gradientPlaceholder = Placeholder(cachedGraph->inputTensors_[3], grad_y);
    Placeholder zStatePlaceholder = Placeholder(cachedGraph->inputTensors_[4], z_state);
    Placeholder cellStateFwdPlaceholder = Placeholder(cachedGraph->inputTensors_[5], cell_state_fwd);
    Placeholder gradientHyPlaceholder = Placeholder(cachedGraph->inputTensors_[6], grad_hy);
    Placeholder gradientCyPlaceholder = Placeholder(cachedGraph->inputTensors_[7], grad_cy);
    Placeholder layersOutputsPlaceholder = Placeholder(cachedGraph->inputTensors_[8], layersOutputs);

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = [[[NSMutableDictionary alloc] init] autorelease];
    [feeds setObject:gradientPlaceholder.getMPSGraphTensorData() forKey:gradientPlaceholder.getMPSGraphTensor()];
    [feeds setObject:gradientHyPlaceholder.getMPSGraphTensorData() forKey:gradientHyPlaceholder.getMPSGraphTensor()];
    [feeds setObject:gradientCyPlaceholder.getMPSGraphTensorData() forKey:gradientCyPlaceholder.getMPSGraphTensor()];
    [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
    [feeds setObject:statePlaceholder.getMPSGraphTensorData() forKey:statePlaceholder.getMPSGraphTensor()];
    [feeds setObject:cellStatePlaceholder.getMPSGraphTensorData() forKey:cellStatePlaceholder.getMPSGraphTensor()];
    [feeds setObject:zStatePlaceholder.getMPSGraphTensorData() forKey:zStatePlaceholder.getMPSGraphTensor()];
    [feeds setObject:cellStateFwdPlaceholder.getMPSGraphTensorData()
              forKey:cellStateFwdPlaceholder.getMPSGraphTensor()];
    [feeds setObject:layersOutputsPlaceholder.getMPSGraphTensorData()
              forKey:layersOutputsPlaceholder.getMPSGraphTensor()];

    NSMutableArray<MPSGraphTensor*>* kernelWeightsList = cachedGraph->kernelWeightsList_;
    NSMutableArray<MPSGraphTensor*>* recurrentKernelWeightsList = cachedGraph->recurrentKernelWeightsList_;
    NSMutableArray<MPSGraphTensor*>* biasList = cachedGraph->biasList_;
    NSMutableArray<MPSGraphTensor*>* recurrentBiasList = cachedGraph->recurrentBiasList_;

    for (const auto i : c10::irange(total_layers)) {
      Placeholder kernelWeight = Placeholder([kernelWeightsList objectAtIndex:i], kernel_weights[i]);
      Placeholder recurrentKernelWeight =
          Placeholder([recurrentKernelWeightsList objectAtIndex:i], recurrent_kernel_weights[i]);
      [feeds setObject:kernelWeight.getMPSGraphTensorData() forKey:kernelWeight.getMPSGraphTensor()];
      [feeds setObject:recurrentKernelWeight.getMPSGraphTensorData() forKey:recurrentKernelWeight.getMPSGraphTensor()];
      if (has_biases) {
        Placeholder bias = Placeholder([biasList objectAtIndex:i], biases[i]);
        Placeholder recurrentBias = Placeholder([recurrentBiasList objectAtIndex:i], recurrent_biases[i]);
        [feeds setObject:bias.getMPSGraphTensorData() forKey:bias.getMPSGraphTensor()];
        [feeds setObject:recurrentBias.getMPSGraphTensorData() forKey:recurrentBias.getMPSGraphTensor()];
      }
    }

    Tensor output_out = at::empty_like(input);
    Tensor grad_state_out = at::empty_like(hx[0]);
    Tensor grad_cell_state_out = at::empty_like(hx[1]);

    std::vector<Tensor> grad_hx = {grad_state_out, grad_cell_state_out};

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        [[[NSMutableDictionary alloc] init] autorelease];
    NSMutableArray<MPSGraphTensor*>* gradRecWeightsArray = cachedGraph->gradRecWeights_;
    NSMutableArray<MPSGraphTensor*>* gradWeightsArray = cachedGraph->gradWeights_;
    NSMutableArray<MPSGraphTensor*>* gradBiasArray = cachedGraph->gradBias_;
    MPSGraphTensor* gradOutput = cachedGraph->gradOutput_;
    MPSGraphTensor* gradState = cachedGraph->gradState_;
    MPSGraphTensor* gradCellState = cachedGraph->gradCellState_;

    Placeholder gradStatePlaceholder = Placeholder(gradState, grad_state_out);
    Placeholder gradCellStatePlaceholder = Placeholder(gradCellState, grad_cell_state_out);
    Placeholder outputPlaceholder = Placeholder(gradOutput, output_out);
    [results setObject:gradStatePlaceholder.getMPSGraphTensorData() forKey:gradStatePlaceholder.getMPSGraphTensor()];
    [results setObject:gradCellStatePlaceholder.getMPSGraphTensorData()
                forKey:gradCellStatePlaceholder.getMPSGraphTensor()];
    [results setObject:outputPlaceholder.getMPSGraphTensorData() forKey:outputPlaceholder.getMPSGraphTensor()];

    Placeholder gradRecWeightsPlaceholder, gradWeightsPlaceholder, gradBiasPlaceholder;

    std::vector<Tensor> weights;
    for (const auto i : c10::irange(total_layers)) {
      Tensor grad_rec_weights = at::empty_like(recurrent_kernel_weights[i]);
      Tensor grad_weights = at::empty_like(kernel_weights[i]);

      weights.push_back(grad_weights);
      weights.push_back(grad_rec_weights);

      gradRecWeightsPlaceholder = Placeholder([gradRecWeightsArray objectAtIndex:i], grad_rec_weights);
      gradWeightsPlaceholder = Placeholder([gradWeightsArray objectAtIndex:i], grad_weights);

      [results setObject:gradRecWeightsPlaceholder.getMPSGraphTensorData()
                  forKey:gradRecWeightsPlaceholder.getMPSGraphTensor()];
      [results setObject:gradWeightsPlaceholder.getMPSGraphTensorData()
                  forKey:gradWeightsPlaceholder.getMPSGraphTensor()];

      if (has_biases) {
        Tensor grad_bias = at::empty((kernel_weights[i].size(0)), kernel_weights[i].options());

        // In PyTorch LSTM API there are two biases. The second bias is included for CuDNN compatibility.
        // In this implementation these two biases are added together and used further.
        // Therefore, they have equal gradient, and it is pushed
        // twice for each of two bias vectors.
        weights.push_back(grad_bias);
        weights.push_back(grad_bias);

        gradBiasPlaceholder = Placeholder([gradBiasArray objectAtIndex:i], grad_bias);
        [results setObject:gradBiasPlaceholder.getMPSGraphTensorData() forKey:gradBiasPlaceholder.getMPSGraphTensor()];
      }
    }

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

    return std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>>(output_out, grad_hx, weights);
  }
}

} // namespace at::native
