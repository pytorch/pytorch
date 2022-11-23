//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/RNN.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/mps/OperationUtils.h>
#import <MetalPerformanceShadersGraph/MPSGraphRNNOps.h>
#include <torch/library.h>

namespace at {
namespace native {

std::vector<long long> getTensorShape(MPSGraphTensor* mpsTensor) {
    std::vector<long long> output_dimensions = {};
    auto dims = mpsTensor.shape;
    for (int i = 0; i<[dims count];i++){
        output_dimensions.push_back([dims[i] intValue]);
    }
    return output_dimensions;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _lstm_mps(const Tensor& input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
    using namespace mps;
    std::vector<Tensor> kernel_weights;
    std::vector<Tensor> recurrent_kernel_weights;
    std::vector<Tensor> biases;
    std::vector<Tensor> recurrent_biases;
    for (size_t i = 0; i < num_layers; i+=1) {
        kernel_weights.push_back(params[i*4]);
        recurrent_kernel_weights.push_back(params[i*4+1]);
        biases.push_back(params[i*4+2]);
        recurrent_biases.push_back(params[i*4+3]);
    }

    struct CachedGraph : public MPSCachedGraph {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      std::vector<MPSGraphTensor*> inputTensors_;
      std::vector<MPSGraphTensor*> outputTensors_;
      NSMutableArray<MPSGraphTensor*> *kernelWeightsList_ = nil;
      NSMutableArray<MPSGraphTensor*> *recurrentKernelWeightsList_ = nil;
      NSMutableArray<MPSGraphTensor*> *biasList_ = nil;
      NSMutableArray<MPSGraphTensor*> *recurrentBiasList_ = nil;
      std::vector<MPSGraphTensor*> outputCellStateFwdVector_;
      std::vector<MPSGraphTensor*> outputZStateVector_;
    };

    MPSGraphCache* cache_ = MPSGraphCache::getInstance();

    MPSStream* stream = getCurrentMPSStream();

    @autoreleasepool {
      string key = "lstm_" + getTensorsStringKey({input, hx[0], hx[1]}) + getMPSTypeString(input.scalar_type()) + "_num_layers_" + std::to_string(num_layers);
      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
      if(!cachedGraph) {
        MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
            MPSGraph* mpsGraph = make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);
            NSMutableArray<MPSGraphTensor*> *kernelWeightsList = [[NSMutableArray alloc] initWithCapacity:params.size()];
            NSMutableArray<MPSGraphTensor*> *recurrentKernelWeightsList = [[NSMutableArray alloc] initWithCapacity:params.size()];
            NSMutableArray<MPSGraphTensor*> *kernelBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];
            NSMutableArray<MPSGraphTensor*> *recurrentBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];

            for (size_t i = 0; i < num_layers; i += 1) {
                [kernelWeightsList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(kernel_weights[i]))];
                [recurrentKernelWeightsList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()),getMPSShape(recurrent_kernel_weights[i]))];
                [kernelBiasList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()),getMPSShape(biases[i]))];
                [recurrentBiasList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()),getMPSShape(recurrent_biases[i]))];
            }

            MPSGraphLSTMDescriptor * opDesc = [MPSGraphLSTMDescriptor descriptor];
            opDesc.training = true;
            opDesc.bidirectional = bidirectional;
            opDesc.produceCell = true;

            MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(input));
            MPSGraphTensor* stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(hx[0]));
            MPSGraphTensor* cellStateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(hx[1]));
            std::vector<MPSGraphTensor*> inputTensors = {inputTensor, stateTensor, cellStateTensor,};

            if(batch_first) {
                inputTensor = [mpsGraph transposeTensor:inputTensor
                                                dimension:0
                                                withDimension:1
                                                name:nil];
            }

            MPSGraphTensor* inputTensor_ = inputTensor;
            MPSGraphTensor* stateTensor_ = [mpsGraph sliceTensor:stateTensor
                                                        dimension:0
                                                        start:0
                                                        length:1
                                                        name:nil];
            MPSGraphTensor* cellStateTensor_ = [mpsGraph sliceTensor:cellStateTensor
                                                                dimension:0
                                                                start:0
                                                                length:1
                                                                name:nil];
            NSArray<MPSGraphTensor*>* outputs = nil;
            NSMutableArray<MPSGraphTensor*>* outputStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
            NSMutableArray<MPSGraphTensor*>* outputCellStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
            NSMutableArray<MPSGraphTensor*>* outputZStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
            NSMutableArray<MPSGraphTensor*>* outputCellStateFwdArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
            for(int i = 0; i < num_layers; i++) {
                MPSGraphTensor* biasTensor = [mpsGraph additionWithPrimaryTensor:kernelBiasList[i]
                                                                    secondaryTensor:recurrentBiasList[i]
                                                                            name:nil];
                outputs = [mpsGraph LSTMWithSourceTensor:inputTensor_
                                        recurrentWeight:recurrentKernelWeightsList[i]
                                            inputWeight:kernelWeightsList[i]
                                                   bias:biasTensor
                                              initState:stateTensor_
                                               initCell:cellStateTensor_
                                             descriptor:opDesc
                                                   name:nil];


                stateTensor_ = [mpsGraph sliceTensor:stateTensor
                                                            dimension:0
                                                            start:i
                                                            length:1
                                                            name:nil];
                cellStateTensor_ = [mpsGraph sliceTensor:cellStateTensor
                                                                    dimension:0
                                                                    start:i
                                                                    length:1
                                                                    name:nil];
                inputTensor_ = [outputs objectAtIndex:0];
                if(dropout_p>0.0 && train && (i!=num_layers-1)) {
                    inputTensor_ = [mpsGraph dropoutTensor:inputTensor_
                                                      rate:dropout_p
                                                      name:nil];

                }

                [outputStateArray addObject:[mpsGraph sliceTensor:[outputs objectAtIndex:0] dimension:0 start:-1 length:1 name:nil]];
                [outputCellStateArray addObject:[mpsGraph sliceTensor:[outputs objectAtIndex:1] dimension:0 start:-1 length:1 name:nil]];
                [outputCellStateFwdArray addObject: [mpsGraph expandDimsOfTensor:[outputs objectAtIndex:1]
                                                                            axis:0
                                                                            name:nil]];
                [outputZStateArray addObject: [mpsGraph expandDimsOfTensor:[outputs objectAtIndex:2]
                                                            axis:0
                                                            name:nil]];
            }

            MPSGraphTensor* outputTensor = [outputs objectAtIndex:0];
            if (batch_first) {
                outputTensor = [mpsGraph transposeTensor:outputTensor
                                               dimension:0
                                           withDimension:1
                                                    name:nil];
            }
            MPSGraphTensor* outputStates = [mpsGraph concatTensors:outputStateArray
                                                            dimension:0
                                                            name:nil];
            MPSGraphTensor* outputCellStates = [mpsGraph concatTensors:outputCellStateArray
                                                            dimension:0
                                                            name:nil];
            MPSGraphTensor* outputZStates = [mpsGraph concatTensors:outputZStateArray
                                                            dimension:0
                                                            name:nil];
            MPSGraphTensor* outputCellStatesFwd = [mpsGraph concatTensors:outputCellStateFwdArray
                                                            dimension:0
                                                            name:nil];

            std::vector<MPSGraphTensor*> outputTensors = {outputTensor, outputStates, outputCellStates, outputZStates, outputCellStatesFwd};
            newCachedGraph->inputTensors_ = inputTensors;
            newCachedGraph->outputTensors_ = outputTensors;
            newCachedGraph->kernelWeightsList_ = kernelWeightsList;
            newCachedGraph->recurrentKernelWeightsList_ = recurrentKernelWeightsList;
            newCachedGraph->biasList_ = kernelBiasList;
            newCachedGraph->recurrentBiasList_ = recurrentBiasList;
          }
          return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
      }

      NSMutableArray<MPSGraphTensor*> *kernelWeightsList = cachedGraph->kernelWeightsList_;
      NSMutableArray<MPSGraphTensor*> *recurrentKernelWeightsList = cachedGraph->recurrentKernelWeightsList_;
      NSMutableArray<MPSGraphTensor*> *biasList = cachedGraph->biasList_;
      NSMutableArray<MPSGraphTensor*> *recurrentBiasList = cachedGraph->recurrentBiasList_;

      Placeholder kernelWeight;
      Placeholder recurrentKernelWeight;
      Placeholder bias;
      Placeholder recurrentBias;
      NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = [[[NSMutableDictionary alloc] init] autorelease];
      for (size_t i = 0; i < num_layers; i+=1) {
          kernelWeight = Placeholder([kernelWeightsList objectAtIndex:i], kernel_weights[i]);
          recurrentKernelWeight = Placeholder([recurrentKernelWeightsList objectAtIndex:i], recurrent_kernel_weights[i]);
          bias = Placeholder([biasList objectAtIndex:i], biases[i]);
          recurrentBias = Placeholder([recurrentBiasList objectAtIndex:i], recurrent_biases[i]);
          [feeds setObject:kernelWeight.getMPSGraphTensorData() forKey:kernelWeight.getMPSGraphTensor()];
          [feeds setObject:recurrentKernelWeight.getMPSGraphTensorData() forKey:recurrentKernelWeight.getMPSGraphTensor()];
          [feeds setObject:bias.getMPSGraphTensorData() forKey:bias.getMPSGraphTensor()];
          [feeds setObject:recurrentBias.getMPSGraphTensorData() forKey:recurrentBias.getMPSGraphTensor()];

      }
      Placeholder selfPlaceholder   = Placeholder(cachedGraph->inputTensors_[0], input);
      Placeholder selfState   = Placeholder(cachedGraph->inputTensors_[1], hx[0]);
      Placeholder selfCellState   = Placeholder(cachedGraph->inputTensors_[2], hx[1]);
      [feeds setObject:selfPlaceholder.getMPSGraphTensorData() forKey:selfPlaceholder.getMPSGraphTensor()];
      [feeds setObject:selfState.getMPSGraphTensorData() forKey:selfState.getMPSGraphTensor()];
      [feeds setObject:selfCellState.getMPSGraphTensorData() forKey:selfCellState.getMPSGraphTensor()];


      auto dims = getTensorShape(cachedGraph->outputTensors_[0]);
      Tensor output = at::empty(IntArrayRef(dims), input.options());
      Tensor hy = at::empty_like(hx[0], input.options());
      Tensor cy = at::empty_like(hx[1], input.options());
      Tensor zState = at::empty(IntArrayRef(getTensorShape(cachedGraph->outputTensors_[3])), input.options());
      Tensor cellStateFwd = at::empty(IntArrayRef(getTensorShape(cachedGraph->outputTensors_[4])), input.options());

      Placeholder outputPlaceholder0 = Placeholder(cachedGraph->outputTensors_[0], output);
      Placeholder outputPlaceholder1 = Placeholder(cachedGraph->outputTensors_[1], hy);
      Placeholder outputPlaceholder2 = Placeholder(cachedGraph->outputTensors_[2], cy);
      Placeholder outputPlaceholder3 = Placeholder(cachedGraph->outputTensors_[3], zState);
      Placeholder outputPlaceholder4 = Placeholder(cachedGraph->outputTensors_[4], cellStateFwd);

      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputPlaceholder0.getMPSGraphTensor() : outputPlaceholder0.getMPSGraphTensorData(),
        outputPlaceholder1.getMPSGraphTensor() : outputPlaceholder1.getMPSGraphTensorData(),
        outputPlaceholder2.getMPSGraphTensor() : outputPlaceholder2.getMPSGraphTensorData(),
        outputPlaceholder3.getMPSGraphTensor() : outputPlaceholder3.getMPSGraphTensorData(),
        outputPlaceholder4.getMPSGraphTensor() : outputPlaceholder4.getMPSGraphTensorData()
      };

      runMPSGraph(stream, cachedGraph->graph(), feeds, results);
      return std::make_tuple(output, hy, cy, zState, cellStateFwd);
    }
}

std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> lstm_mps_backward(const Tensor& grad_y, const c10::optional<Tensor>& grad_hy_opt, const c10::optional<Tensor>& grad_cy_opt, const Tensor& z_state, const Tensor& cell_state_fwd, const Tensor& input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
    using namespace mps;
    const Tensor& grad_hy_r = c10::value_or_else(grad_hy_opt, [] {return Tensor();});
    const Tensor& grad_cy_r = c10::value_or_else(grad_cy_opt, [] {return Tensor();});
    auto grad_hy = grad_hy_r.defined() ? grad_hy_r : at::zeros_like(hx[0], input.options());
    auto grad_cy = grad_cy_r.defined() ? grad_cy_r : at::zeros_like(hx[1], input.options());

    std::vector<Tensor> kernel_weights;
    std::vector<Tensor> recurrent_kernel_weights;
    std::vector<Tensor> biases;
    std::vector<Tensor> recurrent_biases;
    for (size_t i = 0; i < num_layers; i+=1) {
        kernel_weights.push_back(params[i*4]);
        recurrent_kernel_weights.push_back(params[i*4+1]);
        biases.push_back(params[i*4+2]);
        recurrent_biases.push_back(params[i*4+3]);
    }

    struct CachedGraph : public MPSCachedGraph {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      std::vector<MPSGraphTensor*> inputTensors_;
      std::vector<MPSGraphTensor*> outputTensors_;
      NSMutableArray<MPSGraphTensor*> *kernelWeightsList_ = nil;
      NSMutableArray<MPSGraphTensor*> *recurrentKernelWeightsList_ = nil;
      NSMutableArray<MPSGraphTensor*> *biasList_ = nil;
      NSMutableArray<MPSGraphTensor*> *recurrentBiasList_ = nil;
      NSMutableArray<MPSGraphTensor*> *gradOutput_ = nil;
      NSMutableArray<MPSGraphTensor*> *gradRecWeights_ = nil;
      NSMutableArray<MPSGraphTensor*> *gradWeights_ = nil;
      NSMutableArray<MPSGraphTensor*> *gradBias_ = nil;
      NSMutableArray<MPSGraphTensor*> *gradState_ = nil;
      NSMutableArray<MPSGraphTensor*> *gradCellState_ = nil;
    };

    MPSGraphCache* cache_ = MPSGraphCache::getInstance();

    // Get stream
    MPSStream* stream = getCurrentMPSStream();
    @autoreleasepool {

        string key = "lstm_backward_" + getTensorsStringKey({input, z_state, cell_state_fwd, grad_y, grad_cy, grad_hy})+ getMPSTypeString(input.scalar_type()) + "_num_layers_" + std::to_string(num_layers);
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
        if(!cachedGraph) {
            MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

                CachedGraph *newCachedGraph = nil;
                @autoreleasepool {
                    MPSGraph* mpsGraph = make_mps_graph();
                    newCachedGraph = new CachedGraph(mpsGraph);

                    NSMutableArray<MPSGraphTensor*> *kernelWeightsList = [[NSMutableArray alloc] initWithCapacity:params.size()];
                    NSMutableArray<MPSGraphTensor*> *recurrentKernelWeightsList = [[NSMutableArray alloc] initWithCapacity:params.size()];
                    NSMutableArray<MPSGraphTensor*> *kernelBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];
                    NSMutableArray<MPSGraphTensor*> *recurrentBiasList = [[NSMutableArray alloc] initWithCapacity:params.size()];

                    for (size_t i = 0; i < num_layers; i += 1) {
                        [kernelWeightsList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(kernel_weights[i]))];
                        [recurrentKernelWeightsList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()),getMPSShape(recurrent_kernel_weights[i]))];
                        [kernelBiasList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()),getMPSShape(biases[i]))];
                        [recurrentBiasList addObject:mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()),getMPSShape(recurrent_biases[i]))];
                    }

                    MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(input));
                    MPSGraphTensor* stateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(hx[0]));
                    MPSGraphTensor* cellStateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(hx[1]));
                    MPSGraphTensor* zStateTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(input.scalar_type()), getMPSShape(z_state));
                    MPSGraphTensor* gradientTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_y.scalar_type()), getMPSShape(grad_y));
                    MPSGraphTensor* gradientCyTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_cy.scalar_type()), getMPSShape(grad_cy));
                    MPSGraphTensor* gradientHyTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(grad_hy.scalar_type()), getMPSShape(grad_hy));
                    MPSGraphTensor* cellStateFwdTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSDataType(cell_state_fwd.scalar_type()), getMPSShape(cell_state_fwd));

                    std::vector<MPSGraphTensor*> inputs = {inputTensor, stateTensor, cellStateTensor, gradientTensor, zStateTensor, cellStateFwdTensor, gradientHyTensor, gradientCyTensor};
                    newCachedGraph->recurrentKernelWeightsList_ = recurrentKernelWeightsList;
                    newCachedGraph->kernelWeightsList_ = kernelWeightsList;
                    newCachedGraph->biasList_ = kernelBiasList;
                    newCachedGraph->recurrentBiasList_ = recurrentBiasList;
                    newCachedGraph->inputTensors_ = inputs;

                    MPSGraphLSTMDescriptor * opDesc = [MPSGraphLSTMDescriptor descriptor];
                    opDesc.training = true; //train;
                    opDesc.bidirectional = bidirectional;
                    opDesc.produceCell = true;

                    MPSGraphTensor* gradientTensor_ = gradientTensor;

                    NSArray<MPSGraphTensor*>* outputs = nil;

                    NSMutableArray<MPSGraphTensor*>* gradOutputArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
                    NSMutableArray<MPSGraphTensor*>* gradRecWeightsArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
                    NSMutableArray<MPSGraphTensor*>* gradWeightsArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
                    NSMutableArray<MPSGraphTensor*>* gradBiasArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
                    NSMutableArray<MPSGraphTensor*>* gradStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];
                    NSMutableArray<MPSGraphTensor*>* gradCellStateArray = [[NSMutableArray alloc] initWithCapacity:num_layers];

                    for (int i = num_layers - 1; i >= 0; i--) {
                        MPSGraphTensor* zState = [mpsGraph sliceTensor:zStateTensor
                                                                dimension:0
                                                                start:i
                                                                length:1
                                                                name:nil];
                        zState = [mpsGraph squeezeTensor:zState
                                                    axis:0
                                                    name:nil];
                        MPSGraphTensor* cellStateFwd = [mpsGraph sliceTensor:cellStateFwdTensor
                                                                dimension:0
                                                                start:i
                                                                length:1
                                                                name:nil];
                        cellStateFwd = [mpsGraph squeezeTensor:cellStateFwd
                                                    axis:0
                                                    name:nil];
                        MPSGraphTensor* biasTensor = [mpsGraph additionWithPrimaryTensor:kernelBiasList[i]
                                                                            secondaryTensor:recurrentBiasList[i]
                                                                            name:nil];

                        MPSGraphTensor* stateTensor_ = [mpsGraph sliceTensor:stateTensor
                                                                    dimension:0
                                                                    start:i
                                                                    length:1
                                                                    name:nil];
                        MPSGraphTensor* cellStateTensor_ = [mpsGraph sliceTensor:cellStateTensor
                                                                            dimension:0
                                                                            start:i
                                                                            length:1
                                                                            name:nil];
                        MPSGraphTensor* gradientHyTensor_ = [mpsGraph sliceTensor:gradientHyTensor
                                                                    dimension:0
                                                                    start:i
                                                                    length:1
                                                                    name:nil];

                        MPSGraphTensor* gradientCyTensor_ = [mpsGraph sliceTensor:gradientCyTensor
                                                                            dimension:0
                                                                            start:i
                                                                            length:1
                                                                            name:nil];

                        outputs = [mpsGraph LSTMGradientsWithSourceTensor: inputTensor
                                             recurrentWeight: recurrentKernelWeightsList[i]
                                              sourceGradient: gradientTensor_
                                                      zState: zState
                                               cellOutputFwd: cellStateFwd
                                               stateGradient: gradientHyTensor_
                                                cellGradient: gradientCyTensor_
                                                 inputWeight: kernelWeightsList[i]
                                                        bias: biasTensor
                                                   initState: stateTensor_
                                                    initCell: cellStateTensor_
                                                        mask: nil
                                                    peephole: nil
                                                  descriptor: opDesc
                                                        name: nil];


                        gradientTensor_ = [outputs objectAtIndex:0];
                        [gradOutputArray addObject:[outputs objectAtIndex:0]];
                        [gradRecWeightsArray addObject:[outputs objectAtIndex:1]];
                        [gradWeightsArray addObject:[outputs objectAtIndex:2]];
                        [gradBiasArray addObject:[outputs objectAtIndex:3]];
                        [gradStateArray addObject:[outputs objectAtIndex:4]];
                        [gradCellStateArray addObject:[outputs objectAtIndex:5]];
                    }
                    std::vector<MPSGraphTensor*> outputTensors = {[outputs objectAtIndex:0],[outputs objectAtIndex:1],[outputs objectAtIndex:2],[outputs objectAtIndex:3], [outputs objectAtIndex:4], [outputs objectAtIndex:5]};
                    newCachedGraph->outputTensors_ = outputTensors;
                    newCachedGraph->gradOutput_ = gradOutputArray;
                    newCachedGraph->gradRecWeights_ = gradRecWeightsArray;
                    newCachedGraph->gradWeights_ = gradWeightsArray;
                    newCachedGraph->gradBias_ = gradBiasArray;
                    newCachedGraph->gradState_ = gradStateArray;
                    newCachedGraph->gradCellState_ = gradCellStateArray;

                }
                return newCachedGraph;
            });
            cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        Placeholder inputPlaceholder   = Placeholder(cachedGraph->inputTensors_[0], input);
        Placeholder statePlaceholder   = Placeholder(cachedGraph->inputTensors_[1], hx[0]);
        Placeholder cellStatePlaceholder   = Placeholder(cachedGraph->inputTensors_[2], hx[1]);
        Placeholder gradientPlaceholder   = Placeholder(cachedGraph->inputTensors_[3], grad_y);
        Placeholder zStatePlaceholder   = Placeholder(cachedGraph->inputTensors_[4], z_state);
        Placeholder cellStateFwdPlaceholder   = Placeholder(cachedGraph->inputTensors_[5], cell_state_fwd);
        Placeholder gradientHyPlaceholder   = Placeholder(cachedGraph->inputTensors_[6], grad_hy);
        Placeholder gradientCyPlaceholder   = Placeholder(cachedGraph->inputTensors_[7], grad_cy);

        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = [[[NSMutableDictionary alloc] init] autorelease];
        [feeds setObject:gradientPlaceholder.getMPSGraphTensorData() forKey:gradientPlaceholder.getMPSGraphTensor()];
        [feeds setObject:gradientHyPlaceholder.getMPSGraphTensorData() forKey:gradientHyPlaceholder.getMPSGraphTensor()];
        [feeds setObject:gradientCyPlaceholder.getMPSGraphTensorData() forKey:gradientCyPlaceholder.getMPSGraphTensor()];
        [feeds setObject:inputPlaceholder.getMPSGraphTensorData() forKey:inputPlaceholder.getMPSGraphTensor()];
        [feeds setObject:statePlaceholder.getMPSGraphTensorData() forKey: statePlaceholder.getMPSGraphTensor()];
        [feeds setObject:cellStatePlaceholder.getMPSGraphTensorData() forKey:cellStatePlaceholder.getMPSGraphTensor()];
        [feeds setObject:zStatePlaceholder.getMPSGraphTensorData() forKey:zStatePlaceholder.getMPSGraphTensor()];
        [feeds setObject:cellStateFwdPlaceholder.getMPSGraphTensorData() forKey:cellStateFwdPlaceholder.getMPSGraphTensor()];

        NSMutableArray<MPSGraphTensor*> *kernelWeightsList = cachedGraph->kernelWeightsList_;
        NSMutableArray<MPSGraphTensor*> *recurrentKernelWeightsList = cachedGraph->recurrentKernelWeightsList_;
        NSMutableArray<MPSGraphTensor*> *biasList = cachedGraph->biasList_;
        NSMutableArray<MPSGraphTensor*> *recurrentBiasList = cachedGraph->recurrentBiasList_;
        Placeholder kernelWeight;
        Placeholder recurrentKernelWeight;
        Placeholder bias;
        Placeholder recurrentBias;
        for (size_t i = 0; i < num_layers; i+=1) {
            kernelWeight = Placeholder([kernelWeightsList objectAtIndex:i], kernel_weights[i]);
            recurrentKernelWeight = Placeholder([recurrentKernelWeightsList objectAtIndex:i], recurrent_kernel_weights[i]);
            bias = Placeholder([biasList objectAtIndex:i], biases[i]);
            recurrentBias = Placeholder([recurrentBiasList objectAtIndex:i], recurrent_biases[i]);
            [feeds setObject:kernelWeight.getMPSGraphTensorData() forKey:kernelWeight.getMPSGraphTensor()];
            [feeds setObject:recurrentKernelWeight.getMPSGraphTensorData() forKey:recurrentKernelWeight.getMPSGraphTensor()];
            [feeds setObject:bias.getMPSGraphTensorData() forKey:bias.getMPSGraphTensor()];
            [feeds setObject:recurrentBias.getMPSGraphTensorData() forKey:recurrentBias.getMPSGraphTensor()];
        }

        Tensor output = at::empty_like(input);
        Tensor grad_rec_weights = at::empty_like(recurrent_kernel_weights[0]);
        Tensor grad_weights = at::empty_like(kernel_weights[0]);
        Tensor grad_bias = at::empty_like(biases[0]);
        Tensor grad_state = at::empty_like(hx[0]);
        Tensor grad_cell_state = at::empty_like(hx[1]);
        Placeholder outputPlaceholder   = Placeholder(cachedGraph->outputTensors_[0], output);
        Placeholder gradRecWeightsPlaceholder   = Placeholder(cachedGraph->outputTensors_[1], grad_rec_weights);
        Placeholder gradWeightsPlaceholder   = Placeholder(cachedGraph->outputTensors_[2], grad_weights);
        Placeholder gradBiasPlaceholder   = Placeholder(cachedGraph->outputTensors_[3], grad_bias);
        Placeholder gradStatePlaceholder   = Placeholder(cachedGraph->outputTensors_[4], grad_state);
        Placeholder gradCellStatePlaceholder   = Placeholder(cachedGraph->outputTensors_[5], grad_cell_state);

        std::vector<Tensor> grad_hx = {grad_state, grad_cell_state};

        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*> *results = [[[NSMutableDictionary alloc] init] autorelease];
        NSMutableArray<MPSGraphTensor*> *gradOutputArray = cachedGraph->gradOutput_;
        NSMutableArray<MPSGraphTensor*> *gradRecWeightsArray = cachedGraph->gradRecWeights_;
        NSMutableArray<MPSGraphTensor*> *gradWeightsArray = cachedGraph->gradWeights_;
        NSMutableArray<MPSGraphTensor*> *gradBiasArray = cachedGraph->gradBias_;
        NSMutableArray<MPSGraphTensor*> *gradStateArray = cachedGraph->gradState_;
        NSMutableArray<MPSGraphTensor*> *gradCellStateArray = cachedGraph->gradCellState_;
        Placeholder gradOutPlaceholder;

        std::vector<Tensor> weights;
        for (int i = 0; i < num_layers; i++) {
            Tensor output = at::empty_like(input);
            Tensor grad_rec_weights = at::empty_like(recurrent_kernel_weights[i]);
            Tensor grad_weights = at::empty_like(kernel_weights[i]);
            Tensor grad_bias = at::empty_like(biases[i]);
            Tensor grad_state = at::empty_like(hx[0]);
            Tensor grad_cell_state = at::empty_like(hx[1]);
            weights.push_back(grad_weights);
            weights.push_back(grad_rec_weights);
            weights.push_back(grad_bias);
            weights.push_back(grad_bias);
            gradOutPlaceholder = Placeholder([gradOutputArray objectAtIndex:i], output);
            gradRecWeightsPlaceholder = Placeholder([gradRecWeightsArray objectAtIndex:i], grad_rec_weights);
            gradWeightsPlaceholder = Placeholder([gradWeightsArray objectAtIndex:i], grad_weights);
            gradBiasPlaceholder = Placeholder([gradBiasArray objectAtIndex:i], grad_bias);
            gradStatePlaceholder = Placeholder([gradStateArray objectAtIndex:i], grad_state);
            gradCellStatePlaceholder = Placeholder([gradCellStateArray objectAtIndex:i], grad_cell_state);

            [results setObject:gradOutPlaceholder.getMPSGraphTensorData() forKey:gradOutPlaceholder.getMPSGraphTensor()];
            [results setObject:gradRecWeightsPlaceholder.getMPSGraphTensorData() forKey:gradRecWeightsPlaceholder.getMPSGraphTensor()];
            [results setObject:gradBiasPlaceholder.getMPSGraphTensorData() forKey:gradBiasPlaceholder.getMPSGraphTensor()];
            [results setObject:gradStatePlaceholder.getMPSGraphTensorData() forKey:gradStatePlaceholder.getMPSGraphTensor()];
            [results setObject:gradCellStatePlaceholder.getMPSGraphTensorData() forKey:gradCellStatePlaceholder.getMPSGraphTensor()];
            [results setObject:gradWeightsPlaceholder.getMPSGraphTensorData() forKey:gradWeightsPlaceholder.getMPSGraphTensor()];
        }

        runMPSGraph(stream, cachedGraph->graph(), feeds, results);

        return std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> (output, grad_hx, weights);

    }
}
}}//at::native
