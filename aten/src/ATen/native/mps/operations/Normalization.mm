//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/TensorUtils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/layer_norm.h>
#include <torch/library.h>

namespace at {
namespace native {

void get_shapes(MPSShape* input_shape_readonly,
                NSMutableArray<NSNumber*>* &input_shape,
                NSMutableArray<NSNumber*>* &new_mean_shape,
                NSMutableArray<NSNumber*>* &axes,
                int num_input_dims, c10::MemoryFormat memory_format,
                bool isBackward) {
  // Modify the shape
  if(memory_format == at::MemoryFormat::Contiguous) {
    for(int i = 0; i < num_input_dims; i++)
      input_shape[i] = input_shape_readonly[i];
  }
  else { // ChannelsLast
    auto num_channels = input_shape_readonly[1];
    input_shape[0] = input_shape_readonly[0];
    for(int i = 1; i < num_input_dims-1; i++)
      input_shape[i] = input_shape_readonly[i+1];
    input_shape[num_input_dims-1] = num_channels;
  }

  // Mean shape should remain unchanged in backward
  if(memory_format == at::MemoryFormat::Contiguous || isBackward) {
    new_mean_shape[0] = @1;
    new_mean_shape[1] = input_shape_readonly[1];
    for(int i = 2; i < num_input_dims; i++)
      new_mean_shape[i] = @1;
  }
  else if(memory_format == at::MemoryFormat::ChannelsLast) {
    for(int i = 0; i < num_input_dims-1; i++)
      new_mean_shape[i] = @1;
    new_mean_shape[num_input_dims-1] = input_shape[num_input_dims-1];
  }

  // Set axes of reduction
  if(memory_format == at::MemoryFormat::Contiguous || isBackward) {
      axes[0] = @0;
      for(int i = 2; i < num_input_dims; i++)
        axes[i-1] = [NSNumber numberWithInt:i];
    }
    else {
      for(int i = 0; i < num_input_dims-1; i++)
        axes[i] = [NSNumber numberWithInt:i];
    }
}

// Inverse standard deviation now becomes variance (without epsilon)
std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_mps_out
                   (const Tensor& self,
                    const c10::optional<Tensor>& weight_opt,
                    const c10::optional<Tensor>& bias_opt,
                    const c10::optional<Tensor>& running_mean_opt,
                    const c10::optional<Tensor>& running_var_opt,
                    bool train, double momentum, double epsilon,
                    Tensor& output,
                    Tensor& save_mean,
                    Tensor& save_var) {

  namespace native_mps = at::native::mps;
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* biasTensor_ = nil;
    MPSGraphTensor* runningMeanTensor_ = nil;
    MPSGraphTensor* runningVarTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
    MPSGraphTensor* saveMeanTensor_ = nil;
    MPSGraphTensor* saveVarTensor_ = nil;
    MPSGraphTensor* runningMeanInplaceUpdate_ = nil;
    MPSGraphTensor* runningVarInplaceUpdate_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  auto stream = at::mps::getCurrentMPSStream();

  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK(has_running_mean == has_running_var);

  const bool has_weight = (weight_opt.has_value() && weight_opt->defined());
  const bool has_bias = (bias_opt.has_value() && bias_opt->defined());

  const auto memory_format = self.suggest_memory_format();

  if (output.numel() == 0) {
    return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_var);;
  }

  @autoreleasepool {

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    // Number of elements in one channel, needed for bessel correction term
    const int64_t N = self.numel() / save_mean.numel();
    MPSShape* input_shape_readonly = mps::getMPSShape(self);
    int num_input_dims = [input_shape_readonly count];
    // Input shape changes based on memory format
    NSMutableArray<NSNumber*>* input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
    // Shape which can be broadcasted with input
    NSMutableArray<NSNumber*>* new_mean_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
    // Reduction axes
    NSMutableArray<NSNumber*>* axes = [NSMutableArray<NSNumber*> arrayWithCapacity:(num_input_dims-1)];

    get_shapes(input_shape_readonly, input_shape, new_mean_shape, axes, num_input_dims, memory_format, false);

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    string key = "batch_norm_mps_out:" + mem_format_key + ":" + std::to_string(epsilon) + ":"
                      + std::to_string(momentum) + ":" + std::to_string(train) + ":"
                      + std::to_string(has_running_mean) + ":"
                      + std::to_string(has_weight) + ":" + std::to_string(has_bias) + ":"
                      + [ns_shape_key UTF8String] + ":" + native_mps::getMPSTypeString(self.scalar_type());
    auto input_mps_dtype = native_mps::getMPSDataType(self.scalar_type());
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    // Dim where channels are located
    int channelsDim;
    if(memory_format == at::MemoryFormat::Contiguous)
      channelsDim = 1;
    else
      channelsDim = num_input_dims - 1;

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);

            MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input_mps_dtype, input_shape);
            MPSGraphTensor* weightTensor = nil;
            // Should have shape of mean
            if(has_weight)
              weightTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(weight_opt.value().scalar_type()), new_mean_shape);
            MPSGraphTensor* biasTensor = nil;
            if(has_bias)
              biasTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(bias_opt.value().scalar_type()), new_mean_shape);
            MPSGraphTensor* runningMeanTensor = nil;
            MPSGraphTensor* runningVarTensor = nil;
            if(has_running_mean) {
              runningMeanTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(running_mean_opt.value().scalar_type()), new_mean_shape);
              runningVarTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(running_var_opt.value().scalar_type()), new_mean_shape);
            }

            // Mean and inv std tensors to be saved and returned
            MPSGraphTensor* saveMeanTensor = nil;
            MPSGraphTensor* saveVarTensor = nil;

            // Running stats inplace update
            MPSGraphTensor* runningMeanInplaceUpdate = nil;
            MPSGraphTensor* runningVarInplaceUpdate = nil;

            MPSGraphTensor* updatedRunningMeanTensor = nil;
            MPSGraphTensor* updatedRunningVarTensor = nil;

            /*
            If train:
              If has_running_mean:
                Update the running stats to be stored into save_mean and save_var,
                AND to be used in current batchnorm computation
              Else:
                Just calculate the var using batch variance
            If not train:
              Check if running mean exists (maybe do this check before making graph)
              Copy the running mean into the mean to be saved
              Calculate the save_var directly from the running variance

            Compute the batch norm output and stats to be saved
            */

            if(train) {
              // Compute mean and variance of the current batch
              MPSGraphTensor* batchMeanTensor = [mpsGraph meanOfTensor:inputTensor
                                                                  axes:axes
                                                                  name:nil];
              MPSGraphTensor* batchVarianceTensor = [mpsGraph varianceOfTensor:inputTensor
                                                                          axes:axes
                                                                          name:nil];
              if(has_running_mean) {
                // TODO: This is not the formula used in PyTorch, is this OK? Seems more robust
                // float besselCorrectionTerm = float(N) / std::max(N - 1.0f, 1.0f);
                float besselCorrectionTerm = float(N) / float(N - 1.0f);
                MPSGraphTensor* besselConstantTensor = [mpsGraph constantWithScalar:(double)besselCorrectionTerm
                                                                              shape:@[@1]
                                                                           dataType:input_mps_dtype];
                MPSGraphTensor* unbiasedVarianceTensor = [mpsGraph multiplicationWithPrimaryTensor:batchVarianceTensor
                                                                                   secondaryTensor:besselConstantTensor
                                                                                              name:nil];
                MPSGraphTensor* momentumTensor = [mpsGraph constantWithScalar:(double)momentum
                                                                        shape:@[@1]
                                                                     dataType:input_mps_dtype];
                MPSGraphTensor* oneMinusMomentum = [mpsGraph constantWithScalar:(double)(1.0 - momentum)
                                                                          shape:@[@1]
                                                                       dataType:input_mps_dtype];
                // Compute updated running mean
                MPSGraphTensor* scaledBatchMean = [mpsGraph multiplicationWithPrimaryTensor:batchMeanTensor
                                                                            secondaryTensor:momentumTensor
                                                                                       name:nil];
                MPSGraphTensor* scaledRunningMean = [mpsGraph multiplicationWithPrimaryTensor:runningMeanTensor
                                                                              secondaryTensor:oneMinusMomentum
                                                                                         name:nil];
                updatedRunningMeanTensor = [mpsGraph additionWithPrimaryTensor:scaledBatchMean
                                                                               secondaryTensor:scaledRunningMean
                                                                                          name:nil];
                // Compute updated running var
                MPSGraphTensor* scaledCorrectedBatchVar = [mpsGraph multiplicationWithPrimaryTensor:unbiasedVarianceTensor
                                                                                    secondaryTensor:momentumTensor
                                                                                               name:nil];
                MPSGraphTensor* scaledRunningVar = [mpsGraph multiplicationWithPrimaryTensor:runningVarTensor
                                                                             secondaryTensor:oneMinusMomentum
                                                                                        name:nil];
                updatedRunningVarTensor = [mpsGraph additionWithPrimaryTensor:scaledCorrectedBatchVar
                                                              secondaryTensor:scaledRunningVar
                                                                         name:nil];
                // Update saved mean and inverse std tensor
                saveMeanTensor = batchMeanTensor;
                saveVarTensor = batchVarianceTensor;
            }
            else {
              saveMeanTensor = batchMeanTensor;
              saveVarTensor = batchVarianceTensor;
            }
          }
          else { // Test
            TORCH_CHECK(has_running_mean);
            saveMeanTensor = [mpsGraph identityWithTensor:runningMeanTensor
                                                     name:nil];
            saveVarTensor = [mpsGraph identityWithTensor:runningVarTensor
                                                    name:nil];
          }

          // Compute output of batch norm
          MPSGraphTensor* outputTensor = [mpsGraph normalizationWithTensor:inputTensor
                                                                meanTensor:saveMeanTensor
                                                            varianceTensor:saveVarTensor
                                                               gammaTensor:weightTensor
                                                                betaTensor:biasTensor
                                                                   epsilon:(float)epsilon
                                                                      name:nil];

          // Reshape saved mean and var to fit output
          saveMeanTensor = [mpsGraph reshapeTensor:saveMeanTensor
                                         withShape:@[new_mean_shape[channelsDim]]
                                               name:nil];
          saveVarTensor = [mpsGraph reshapeTensor:saveVarTensor
                                        withShape:@[new_mean_shape[channelsDim]]
                                             name:nil];

          if(train && has_running_mean) {
            // Running stats inplace update
            runningMeanInplaceUpdate = [mpsGraph reshapeTensor:updatedRunningMeanTensor
                                                     withShape:@[input_shape[channelsDim]]
                                                          name:nil];
            runningVarInplaceUpdate = [mpsGraph reshapeTensor:updatedRunningVarTensor
                                                    withShape:@[input_shape[channelsDim]]
                                                         name:nil];
          }

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->weightTensor_ = weightTensor;
          newCachedGraph->biasTensor_ = biasTensor;
          newCachedGraph->runningMeanTensor_ = runningMeanTensor;
          newCachedGraph->runningVarTensor_ = runningVarTensor;
          newCachedGraph->outputTensor_ = outputTensor;
          newCachedGraph->saveMeanTensor_ = saveMeanTensor;
          newCachedGraph->saveVarTensor_ = saveVarTensor;
          newCachedGraph->runningMeanInplaceUpdate_ = runningMeanInplaceUpdate;
          newCachedGraph->runningVarInplaceUpdate_ = runningVarInplaceUpdate;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, self, input_shape);
    auto weightPlaceholder = native_mps::Placeholder();
    if(has_weight)
      weightPlaceholder = native_mps::Placeholder(cachedGraph->weightTensor_, weight_opt.value(), new_mean_shape);
    auto biasPlaceholder = native_mps::Placeholder();
    if(has_bias)
      biasPlaceholder = native_mps::Placeholder(cachedGraph->biasTensor_, bias_opt.value(), new_mean_shape);
    auto runningMeanPlaceholder = native_mps::Placeholder();
    auto runningVarPlaceholder = native_mps::Placeholder();
    if(has_running_mean) {
      runningMeanPlaceholder = native_mps::Placeholder(cachedGraph->runningMeanTensor_, running_mean_opt.value(), new_mean_shape);
      runningVarPlaceholder = native_mps::Placeholder(cachedGraph->runningVarTensor_, running_var_opt.value(), new_mean_shape);
    }

    auto runningMeanInplaceUpdatePlaceholder = native_mps::Placeholder();
    auto runningVarInplaceUpdatePlaceholder = native_mps::Placeholder();

    if(train && has_running_mean) {
      runningMeanInplaceUpdatePlaceholder = native_mps::Placeholder(cachedGraph->runningMeanInplaceUpdate_, running_mean_opt.value());
      runningVarInplaceUpdatePlaceholder = native_mps::Placeholder(cachedGraph->runningVarInplaceUpdate_, running_var_opt.value());
    }

    auto outputPlaceholder = native_mps::Placeholder(cachedGraph->outputTensor_, output, input_shape);
    auto saveMeanPlaceholder = native_mps::Placeholder(cachedGraph->saveMeanTensor_, save_mean);
    auto saveVarPlaceholder = native_mps::Placeholder(cachedGraph->saveVarTensor_, save_var);

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    if(has_weight)
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    if(has_bias)
      feeds[biasPlaceholder.getMPSGraphTensor()] = biasPlaceholder.getMPSGraphTensorData();
    if(has_running_mean) {
      feeds[runningMeanPlaceholder.getMPSGraphTensor()] = runningMeanPlaceholder.getMPSGraphTensorData();
      feeds[runningVarPlaceholder.getMPSGraphTensor()] = runningVarPlaceholder.getMPSGraphTensorData();
    }

    NSMutableDictionary *results = [[NSMutableDictionary new] autorelease];
    results[outputPlaceholder.getMPSGraphTensor()] = outputPlaceholder.getMPSGraphTensorData();
    results[saveMeanPlaceholder.getMPSGraphTensor()] = saveMeanPlaceholder.getMPSGraphTensorData();
    results[saveVarPlaceholder.getMPSGraphTensor()] = saveVarPlaceholder.getMPSGraphTensorData();

    // If train and has_running_mean, add updated running mean to the output
    if(train && has_running_mean) {
      results[runningMeanInplaceUpdatePlaceholder.getMPSGraphTensor()] = runningMeanInplaceUpdatePlaceholder.getMPSGraphTensorData();
      results[runningVarInplaceUpdatePlaceholder.getMPSGraphTensor()] = runningVarInplaceUpdatePlaceholder.getMPSGraphTensorData();
    }

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }

  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_var);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_mps
                  (const Tensor& self,
                   const c10::optional<Tensor>& weight_opt,
                   const c10::optional<Tensor>& bias_opt,
                   const c10::optional<Tensor>& running_mean_opt,
                   const c10::optional<Tensor>& running_var_opt,
                   bool train,
                   double momentum,
                   double epsilon) {

  const auto memory_format = self.suggest_memory_format();

  auto output = at::native::empty_mps(
          self.sizes(),
          self.scalar_type(),
          c10::nullopt,
          kMPS,
          c10::nullopt,
          memory_format);

  int64_t n_input = self.size(1);

  auto save_mean = at::native::empty_mps(
              {n_input},
              self.scalar_type(),
              // TODO: Accumulate type?
              // at::toAccumulateType(self.scalar_type(), /*is_cuda=*/false),
              c10::nullopt,
              kMPS,
              c10::nullopt,
              c10::nullopt);
  auto save_var = at::native::empty_mps(
              {n_input},
              self.scalar_type(),
              // TODO: Accumulate type?
              // at::toAccumulateType(self.scalar_type(), /*is_cuda=*/false),
              c10::nullopt,
              kMPS,
              c10::nullopt,
              c10::nullopt);

  at::native::batch_norm_mps_out(
      self,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      epsilon,
      output,
      save_mean,
      save_var);
  return std::make_tuple(output, save_mean, save_var);
}

string get_mem_string(c10::MemoryFormat memory_format) {
  string mem_format_key;
  switch(memory_format) {
    case at::MemoryFormat::Contiguous:
      mem_format_key = "Contiguous";
      break;
    case at::MemoryFormat::ChannelsLast:
      mem_format_key = "ChannelsLast";
      break;
    default:
      assert(0 && "Invalid memory format\n");
  }

  return mem_format_key;
}

// Batch norm backward
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_mps
                  (const Tensor& grad_out,
                   const Tensor& input,
                   const c10::optional<Tensor>& weight_opt,
                   const c10::optional<Tensor>& running_mean_opt,
                   const c10::optional<Tensor>& running_var_opt,
                   const c10::optional<Tensor>& save_mean_opt,
                   const c10::optional<Tensor>& save_var_opt,
                   bool train,
                   double epsilon,
                   std::array<bool,3> grad_input_mask) {

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  const auto memory_format = input.suggest_memory_format();

  if (grad_input_mask[0]) {
    grad_input = at::native::empty_mps(input.sizes(),
                                       input.scalar_type(),
                                       c10::nullopt,
                                       kMPS,
                                       c10::nullopt,
                                       memory_format);
  }
  // Assuming that if grad_input_mask of weight is 1, then the weight is available
  if (grad_input_mask[1]) {
    grad_weight = at::native::empty_mps(weight_opt.value().sizes(),
                                        weight_opt.value().scalar_type(),
                                        c10::nullopt,
                                        kMPS,
                                        c10::nullopt,
                                        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    grad_bias = at::native::empty_mps(weight_opt.value().sizes(),
                                      weight_opt.value().scalar_type(),
                                      c10::nullopt,
                                      kMPS,
                                      c10::nullopt,
                                      at::MemoryFormat::Contiguous);
  }

  namespace native_mps = at::native::mps;

  // Derive from MPSCachedGraph
  struct CachedGraph : public native_mps::MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* gradOutputTensor_ = nil;
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* weightTensor_ = nil;
    MPSGraphTensor* runningMeanTensor_ = nil;
    MPSGraphTensor* runningVarTensor_ = nil;
    MPSGraphTensor* saveMeanTensor_ = nil;
    MPSGraphTensor* saveVarTensor_ = nil;
    MPSGraphTensor* gradInputTensor_ = nil;
    MPSGraphTensor* gradWeightTensor_ = nil;
    MPSGraphTensor* gradBiasTensor_ = nil;
  };

  native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

  auto stream = at::mps::getCurrentMPSStream();

  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK(has_running_mean == has_running_var);
  const bool has_save_mean = (save_mean_opt.has_value() && save_mean_opt->defined());
  const bool has_save_var = (save_var_opt.has_value() && save_var_opt->defined());
  TORCH_CHECK(has_save_mean == has_save_var);

  const bool has_weight = (weight_opt.has_value() && weight_opt->defined());

  if (grad_input.numel() == 0) {
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  @autoreleasepool {

    string mem_format_key;
    switch(memory_format) {
      case at::MemoryFormat::Contiguous:
        mem_format_key = "Contiguous";
        break;
      case at::MemoryFormat::ChannelsLast:
        mem_format_key = "ChannelsLast";
        break;
      default:
        assert(0 && "Check should have been done earlier\n");
    }

    MPSShape* input_shape_readonly = mps::getMPSShape(input);
    int num_input_dims = [input_shape_readonly count];
    NSMutableArray<NSNumber*>* input_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
    // Broadcast with input
    NSMutableArray<NSNumber*>* new_mean_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:num_input_dims];
    // Reduction axes
    NSMutableArray<NSNumber*>* axes = [NSMutableArray<NSNumber*> arrayWithCapacity:(num_input_dims-1)];

    get_shapes(input_shape_readonly, input_shape, new_mean_shape, axes, num_input_dims, memory_format, true);

    NSString* ns_shape_key = [[input_shape valueForKey:@"description"] componentsJoinedByString:@","];

    string key = "batch_norm_backward_mps:" + mem_format_key + ":" + std::to_string(epsilon) + ":"
                      + std::to_string(train) + ":"
                      + std::to_string(has_running_mean) + ":"
                      + std::to_string(has_weight) + ":"
                      + [ns_shape_key UTF8String] + ":" + native_mps::getMPSTypeString(input.scalar_type());
    auto input_mps_dtype = native_mps::getMPSDataType(input.scalar_type());
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = native_mps::make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          // NCHW - Channels dim is 1
          int channelsDim = 1;

          MPSGraphTensor* inputTensorOriginal = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, input_mps_dtype, input_shape);
          // Shape is the ORIGINAL NCHW shape
          MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(grad_out.scalar_type()), input_shape_readonly);
          MPSGraphTensor* weightTensor = nil;
          if(has_weight)
            weightTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(weight_opt.value().scalar_type()), new_mean_shape);
          MPSGraphTensor* runningMeanTensor = nil;
          MPSGraphTensor* runningVarTensor = nil;
          if(has_running_mean) {
            runningMeanTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(running_mean_opt.value().scalar_type()), new_mean_shape);
            runningVarTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(running_var_opt.value().scalar_type()), new_mean_shape);
          }

          // Mean and inv std tensors to be saved and returned
          MPSGraphTensor* saveMeanTensor = nil;
          MPSGraphTensor* saveVarTensor = nil;
          if(has_save_mean) {
            saveMeanTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(save_mean_opt.value().scalar_type()), new_mean_shape);
            saveVarTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(save_var_opt.value().scalar_type()), new_mean_shape);
          }

          MPSGraphTensor* gradInputTensor = nil;
          MPSGraphTensor* gradWeightTensor = nil;
          MPSGraphTensor* gradBiasTensor = nil;
          MPSGraphTensor* inputTensor = nil;

          if(memory_format == at::MemoryFormat::Contiguous)
            inputTensor = inputTensorOriginal;
          else {
            // Reshape/transpose the input as needed
            auto N = input_shape[0];
            auto H = input_shape[1];
            auto W = input_shape[2];
            auto C = input_shape[3];

            inputTensor = [mpsGraph reshapeTensor:inputTensorOriginal
                                        withShape:@[N, ([NSNumber numberWithInt:[H intValue]* [W intValue]]), C]
                                             name:nil];
            inputTensor = [mpsGraph transposeTensor:inputTensor
                                          dimension:1
                                      withDimension:2
                                               name:nil];
            inputTensor = [mpsGraph reshapeTensor:inputTensor
                                        withShape:@[N, C, H, W]
                                             name:nil];
          }

          if(train) {
            // Use save_mean and save_var
            if(grad_input_mask[1]) {
              gradWeightTensor = [mpsGraph normalizationGammaGradientWithIncomingGradientTensor:gradOutputTensor
                                                                                   sourceTensor:inputTensor
                                                                                     meanTensor:saveMeanTensor
                                                                                 varianceTensor:saveVarTensor
                                                                                  reductionAxes:axes
                                                                                        epsilon:(float)epsilon
                                                                                           name:nil];
            }
            if(grad_input_mask[2]) {
              gradBiasTensor = [mpsGraph normalizationBetaGradientWithIncomingGradientTensor:gradOutputTensor
                                                                                sourceTensor:inputTensor
                                                                               reductionAxes:axes
                                                                                        name:nil];
            }
            if(grad_input_mask[0]) {
              gradInputTensor = [mpsGraph normalizationGradientWithIncomingGradientTensor:gradOutputTensor
                                                                             sourceTensor:inputTensor
                                                                               meanTensor:saveMeanTensor
                                                                           varianceTensor:saveVarTensor
                                                                              gammaTensor:weightTensor
                                                                      gammaGradientTensor:gradWeightTensor
                                                                       betaGradientTensor:gradBiasTensor
                                                                            reductionAxes:axes
                                                                                  epsilon:(float) epsilon
                                                                                     name:nil];
            }
          }
          else {
            // Use running mean and running var
            MPSGraphTensor* rsqrtTensor = nil;
            MPSGraphTensor* epsilonTensor = nil;
            if(grad_input_mask[1]) {
              epsilonTensor = [mpsGraph constantWithScalar:(float)epsilon
                                                     shape:@[@1]
                                                  dataType:input_mps_dtype];
              MPSGraphTensor* xMinusMean = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                                  secondaryTensor:runningMeanTensor
                                                                             name:nil];
              MPSGraphTensor* varianceEpsTensor = [mpsGraph additionWithPrimaryTensor:runningVarTensor
                                                                      secondaryTensor:epsilonTensor
                                                                                 name:nil];
              rsqrtTensor = [mpsGraph reverseSquareRootWithTensor:varianceEpsTensor
                                                             name:nil];
              MPSGraphTensor* bnForwardTensor = [mpsGraph multiplicationWithPrimaryTensor:xMinusMean
                                                                          secondaryTensor:rsqrtTensor
                                                                                     name:nil];
              MPSGraphTensor* gradBnMulTensor = [mpsGraph multiplicationWithPrimaryTensor:bnForwardTensor
                                                                          secondaryTensor:gradOutputTensor
                                                                                     name:nil];
              gradWeightTensor = [mpsGraph reductionSumWithTensor:gradBnMulTensor
                                                             axes:axes
                                                             name:nil];
            }
            if(grad_input_mask[2]) {
              gradBiasTensor = [mpsGraph normalizationBetaGradientWithIncomingGradientTensor:gradOutputTensor
                                                                                sourceTensor:inputTensor
                                                                               reductionAxes:axes
                                                                                        name:nil];
            }
            if(grad_input_mask[0]) {

              MPSGraphTensor* unitTensor = [mpsGraph constantWithScalar:1.0
                                                                  shape:input_shape_readonly
                                                               dataType:input_mps_dtype];
              if(!epsilonTensor)
                epsilonTensor = [mpsGraph constantWithScalar:(float)epsilon
                                                       shape:@[@1]
                                                    dataType:input_mps_dtype];
              if(!rsqrtTensor) {
                MPSGraphTensor* varianceEpsTensor = [mpsGraph additionWithPrimaryTensor:runningVarTensor
                                                                      secondaryTensor:epsilonTensor
                                                                                 name:nil];
                rsqrtTensor = [mpsGraph reverseSquareRootWithTensor:varianceEpsTensor
                                                               name:nil];
              }

              gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:unitTensor
                                                          secondaryTensor:rsqrtTensor
                                                                     name:nil];
              if(has_weight)
                gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradInputTensor
                                                            secondaryTensor:weightTensor
                                                                       name:nil];
              gradInputTensor = [mpsGraph multiplicationWithPrimaryTensor:gradInputTensor
                                                          secondaryTensor:gradOutputTensor
                                                                     name:nil];
            }
          }

          if(grad_input_mask[1]) {
            gradWeightTensor = [mpsGraph reshapeTensor:gradWeightTensor
                                             withShape:@[input_shape_readonly[channelsDim]]
                                                  name:nil];
          }
          if(grad_input_mask[2]) {
            gradBiasTensor = [mpsGraph reshapeTensor:gradBiasTensor
                                           withShape:@[input_shape_readonly[channelsDim]]
                                                name:nil];
          }

          MPSGraphTensor* gradInputTensorFinal = nil;

          if(memory_format == at::MemoryFormat::Contiguous)
            gradInputTensorFinal = gradInputTensor;
          else {
            // Reshape/transpose the input as needed
            auto N = input_shape[0];
            auto H = input_shape[1];
            auto W = input_shape[2];
            auto C = input_shape[3];

            gradInputTensorFinal = [mpsGraph reshapeTensor:gradInputTensor
                                                 withShape:@[N, C, ([NSNumber numberWithInt:[H intValue]* [W intValue]])]
                                                      name:nil];
            gradInputTensorFinal = [mpsGraph transposeTensor:gradInputTensorFinal
                                                   dimension:1
                                               withDimension:2
                                                        name:nil];
            gradInputTensorFinal = [mpsGraph reshapeTensor:gradInputTensorFinal
                                                 withShape:@[N, H, W, C]
                                                      name:nil];
          }

          newCachedGraph->gradOutputTensor_ = gradOutputTensor;
          newCachedGraph->inputTensor_ = inputTensorOriginal;
          newCachedGraph->weightTensor_ = weightTensor;
          newCachedGraph->runningMeanTensor_ = runningMeanTensor;
          newCachedGraph->runningVarTensor_ = runningVarTensor;
          newCachedGraph->saveMeanTensor_ = saveMeanTensor;
          newCachedGraph->saveVarTensor_ = saveVarTensor;
          newCachedGraph->gradInputTensor_ = gradInputTensorFinal;
          newCachedGraph->gradWeightTensor_ = gradWeightTensor;
          newCachedGraph->gradBiasTensor_ = gradBiasTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, input, input_shape);
    auto gradOutputPlaceholder = native_mps::Placeholder(cachedGraph->gradOutputTensor_, grad_out, input_shape_readonly);
    auto weightPlaceholder = native_mps::Placeholder();
    if(has_weight)
      weightPlaceholder = native_mps::Placeholder(cachedGraph->weightTensor_, weight_opt.value(), new_mean_shape);
    auto runningMeanPlaceholder = native_mps::Placeholder();
    auto runningVarPlaceholder = native_mps::Placeholder();
    if(has_running_mean) {
      runningMeanPlaceholder = native_mps::Placeholder(cachedGraph->runningMeanTensor_, running_mean_opt.value(), new_mean_shape);
      runningVarPlaceholder = native_mps::Placeholder(cachedGraph->runningVarTensor_, running_var_opt.value(), new_mean_shape);
    }
    auto saveMeanPlaceholder = native_mps::Placeholder();
    auto saveVarPlaceholder = native_mps::Placeholder();
    if(has_save_mean) {
      saveMeanPlaceholder = native_mps::Placeholder(cachedGraph->saveMeanTensor_, save_mean_opt.value(), new_mean_shape);
      saveVarPlaceholder = native_mps::Placeholder(cachedGraph->saveVarTensor_, save_var_opt.value(), new_mean_shape);
    }

    auto gradInputPlaceholder = native_mps::Placeholder();
    if(grad_input_mask[0])
      gradInputPlaceholder = native_mps::Placeholder(cachedGraph->gradInputTensor_, grad_input, input_shape);
    auto gradWeightPlaceholder = native_mps::Placeholder();
    if(grad_input_mask[1])
      gradWeightPlaceholder = native_mps::Placeholder(cachedGraph->gradWeightTensor_, grad_weight);
    auto gradBiasPlaceholder = native_mps::Placeholder();;
    if(grad_input_mask[2])
      gradBiasPlaceholder = native_mps::Placeholder(cachedGraph->gradBiasTensor_, grad_bias);

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
    feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
    if(has_weight)
      feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
    if(has_running_mean) {
      feeds[runningMeanPlaceholder.getMPSGraphTensor()] = runningMeanPlaceholder.getMPSGraphTensorData();
      feeds[runningVarPlaceholder.getMPSGraphTensor()] = runningVarPlaceholder.getMPSGraphTensorData();
    }
    if(has_save_mean) {
      feeds[saveMeanPlaceholder.getMPSGraphTensor()] = saveMeanPlaceholder.getMPSGraphTensorData();
      feeds[saveVarPlaceholder.getMPSGraphTensor()] = saveVarPlaceholder.getMPSGraphTensorData();
    }

    NSMutableDictionary *results = [[NSMutableDictionary new] autorelease];
    if(grad_input_mask[0])
      results[gradInputPlaceholder.getMPSGraphTensor()] = gradInputPlaceholder.getMPSGraphTensorData();
    if(grad_input_mask[1])
      results[gradWeightPlaceholder.getMPSGraphTensor()] = gradWeightPlaceholder.getMPSGraphTensorData();
    if(grad_input_mask[2])
      results[gradBiasPlaceholder.getMPSGraphTensor()] = gradBiasPlaceholder.getMPSGraphTensorData();

    native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);

}

// Layer norm forward for MPS
std::tuple<Tensor, Tensor, Tensor> layer_norm_mps(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    double eps) {

  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();

  auto input_shape = input.sizes();
  const auto input_ndim = input.dim();
  const int normalized_ndim = normalized_shape.size();
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  const int axis = input_ndim - normalized_ndim;
  at::Tensor input_reshaped = input.reshape({1, M, -1});
  // Unlike Batch Normalization, which applies scalar scale and bias for each
  // entire channel/plane with the affine option, Layer Normalization applies
  // per-element scale and bias. E.g. For input {N, C, H, W}, weight for
  // batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
  auto outputs = at::native_batch_norm(
      input_reshaped, /*weight=*/{}, /*bias=*/{}, /*running_mean=*/{},
      /*running_var=*/{}, /*training=*/true, /*momentum=*/0, eps);
  at::Tensor out = std::get<0>(outputs);
  out = out.view(input_shape);
  if (weight.defined() && bias.defined()) {
    out = bias.addcmul(out, weight, 1);
  } else if (weight.defined()) {
    out = out.mul(weight);
  } else if (bias.defined()) {
    out = out.add(bias);
  }
  at::Tensor mean = std::get<1>(outputs);
  at::Tensor variance = std::get<2>(outputs);

  at::Tensor rstd = at::rsqrt(at::add(variance, eps));

  std::vector<int64_t> stat_shape;
  for (const auto idx : c10::irange(axis)) {
    stat_shape.push_back(input_shape[idx]);
  }
  for (const auto idx : c10::irange(axis, input.dim())) {
    (void)idx; // Suppress unused variable
    stat_shape.push_back(1);
  }
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
  return std::make_tuple(out, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_mps(
    const Tensor& grad_out,
    const Tensor& input,
    IntArrayRef normalized_shape,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& weight_opt /* optional */,
    const c10::optional<Tensor>& bias_opt /* optional */,
    std::array<bool, 3> grad_input_mask) {

  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();
  auto dOut = grad_out.expect_contiguous();

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::native::empty_like(
        *X,
        c10::nullopt /* dtype */,
        c10::nullopt /* layout */,
        kMPS /* device */,
        c10::nullopt /* pin_memory */,
        at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[1]) {
    grad_weight = M > 0 ? at::native::empty_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         kMPS /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous)
                   : at::native::zeros_like(
                         *gamma,
                         c10::nullopt /* dtype */,
                         c10::nullopt /* layout */,
                         kMPS /* device */,
                         c10::nullopt /* pin_memory */,
                         at::MemoryFormat::Contiguous);
  }
  if (grad_input_mask[2]) {
    grad_bias = M > 0 ? at::native::empty_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        kMPS /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous)
                  : at::native::zeros_like(
                        *beta,
                        c10::nullopt /* dtype */,
                        c10::nullopt /* layout */,
                        kMPS /* device */,
                        c10::nullopt /* pin_memory */,
                        at::MemoryFormat::Contiguous);
  }
  if (M > 0) {

    namespace native_mps = at::native::mps;

    // Derive from MPSCachedGraph
    struct CachedGraph : public native_mps::MPSCachedGraph
    {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor* gradOutputTensor_ = nil;
      MPSGraphTensor* inputTensor_ = nil;
      MPSGraphTensor* weightTensor_ = nil;
      MPSGraphTensor* meanTensor_ = nil;
      MPSGraphTensor* rstdTensor_ = nil;
      MPSGraphTensor* gradInputTensor_ = nil;
      MPSGraphTensor* gradWeightTensor_ = nil;
      MPSGraphTensor* gradBiasTensor_ = nil;
    };

    native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

    auto stream = at::mps::getCurrentMPSStream();

    const bool has_weight = (weight_opt.has_value() && weight_opt->defined());

    if (grad_input.numel() == 0) {
      return std::make_tuple(grad_input, grad_weight, grad_bias);
    }

    // const auto memory_format = input.suggest_memory_format();

    @autoreleasepool {

      MPSShape* input_shape = mps::getMPSShape(*X);
      MPSShape* gamma_shape = mps::getMPSShape(normalized_shape);

      auto num_normalized_dims = [gamma_shape count];
      auto num_channel_dims = [input_shape count] - num_normalized_dims;

      NSMutableArray<NSNumber*>* gamma_axes = [NSMutableArray<NSNumber*> arrayWithCapacity:num_channel_dims];

      for(int i = 0; i < num_channel_dims; i++)
        gamma_axes[i] = [NSNumber numberWithInt:i];

      // Axes along which to reduce to get "batch norm" gradient
      // This will be applied on shape [1, M, -1]
      NSMutableArray<NSNumber*>* bn_axes = [NSMutableArray<NSNumber*> arrayWithCapacity:num_normalized_dims];
      for(int i = 0; i < num_normalized_dims; i++)
        bn_axes[i] = [NSNumber numberWithInt:(1+1+i)];

      // Shape of input to do "batch norm" backward
      // This is [1, M, -1]
      NSMutableArray<NSNumber*>* bn_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:(num_normalized_dims+2)];
      bn_shape[0] = [NSNumber numberWithInt:1];
      bn_shape[1] = [NSNumber numberWithInt:M];
      for(int i = 0; i < num_normalized_dims; i++)
        bn_shape[i+2] = input_shape[i+num_channel_dims];

      // Shape of mean to do "batch norm" backward
      // This is [1, M, [1,1,1..1]]
      NSMutableArray<NSNumber*>* bn_mean_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:(num_normalized_dims+2)];
      bn_mean_shape[0] = [NSNumber numberWithInt:1];
      bn_mean_shape[1] = [NSNumber numberWithInt:M];
      for(int i = 0; i < num_normalized_dims; i++)
        bn_mean_shape[i+2] = [NSNumber numberWithInt:1];

      // Shape of gamma to multiply with "batch norm" backward
      // This is [1, 1, -1]
      NSMutableArray<NSNumber*>* bn_gamma_shape = [NSMutableArray<NSNumber*> arrayWithCapacity:(num_normalized_dims+2)];
      bn_gamma_shape[0] = [NSNumber numberWithInt:1];
      bn_gamma_shape[1] = [NSNumber numberWithInt:1];
      for(int i = 0; i < num_normalized_dims; i++)
        bn_gamma_shape[i+2] = input_shape[i+num_channel_dims];

      string key = "layer_norm_backward_mps:"
                        + std::to_string(has_weight) + ":"
                        + native_mps::getArrayRefString(normalized_shape) + ":"
                        + native_mps::getArrayRefString((*X).sizes()) + ":"
                        + native_mps::getMPSTypeString((*X).scalar_type());

      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

      if(!cachedGraph) {
        native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
            MPSGraph* mpsGraph = native_mps::make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);

            MPSGraphTensor* inputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, *X);
            MPSGraphTensor* gradOutputTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, *dOut);
            MPSGraphTensor* weightTensor = nil;
            if(has_weight)
              weightTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, *gamma);

            // Mean and inv std tensors to be saved and returned
            MPSGraphTensor* meanTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, mean);
            MPSGraphTensor* rstdTensor = native_mps::mpsGraphRankedPlaceHolder(mpsGraph, rstd);

            MPSGraphTensor* gradInputTensor = nil;
            MPSGraphTensor* gradWeightTensor = nil;
            MPSGraphTensor* gradBiasTensor = nil;

            if(grad_input_mask[1]) {
              MPSGraphTensor* xMinusMean = [mpsGraph subtractionWithPrimaryTensor:inputTensor
                                                                  secondaryTensor:meanTensor
                                                                             name:nil];
              MPSGraphTensor* bnForwardTensor = [mpsGraph multiplicationWithPrimaryTensor:xMinusMean
                                                                          secondaryTensor:rstdTensor
                                                                                     name:nil];
              MPSGraphTensor* gradBnMulTensor = [mpsGraph multiplicationWithPrimaryTensor:bnForwardTensor
                                                                          secondaryTensor:gradOutputTensor
                                                                                     name:nil];
              gradWeightTensor = [mpsGraph reductionSumWithTensor:gradBnMulTensor
                                                             axes:gamma_axes
                                                             name:nil];
            }
            if(grad_input_mask[2]) {
              gradBiasTensor = [mpsGraph reductionSumWithTensor:gradOutputTensor
                                                           axes:gamma_axes
                                                           name:nil];
            }
            if(grad_input_mask[0]) {

              // Reshape input to [1, M, -1]
              // Reshape mean and rstd to [1, M, -1]
              // Reshape gamma to [1, 1, -1] (-1 has N dims)

              MPSGraphTensor* bnInputTensor = [mpsGraph reshapeTensor:inputTensor
                                                            withShape:bn_shape
                                                                 name:nil];
              MPSGraphTensor* bnGradOutputTensor = [mpsGraph reshapeTensor:gradOutputTensor
                                                                 withShape:bn_shape
                                                                      name:nil];
              // Do this at the end
              if(has_weight) {
                MPSGraphTensor* bnGammaTensor = [mpsGraph reshapeTensor:weightTensor
                                                              withShape:bn_gamma_shape
                                                                   name:nil];
                bnGradOutputTensor = [mpsGraph multiplicationWithPrimaryTensor:bnGradOutputTensor
                                                               secondaryTensor:bnGammaTensor
                                                                          name:nil];
              }
              MPSGraphTensor* bnMeanTensor = [mpsGraph reshapeTensor:meanTensor
                                                           withShape:bn_mean_shape
                                                                name:nil];
              MPSGraphTensor* bnRstdTensor = [mpsGraph reshapeTensor:rstdTensor
                                                           withShape:bn_mean_shape
                                                                name:nil];

              MPSGraphTensor* mulTensor = [mpsGraph constantWithScalar:N
                                                   shape:@[@1]
                                                dataType:MPSDataTypeInt32];

              MPSGraphTensor* numberToReduceTensor = mulTensor;

              MPSGraphTensor* cast2Tensor = [mpsGraph castTensor:numberToReduceTensor
                                                          toType:bnInputTensor.dataType
                                                            name:@"cast2Tensor"];

              MPSGraphTensor* sizeReciprocalTensor = [mpsGraph reciprocalWithTensor:cast2Tensor
                                                                               name:nil];

              // TODO: Reduce redundant computation
              MPSGraphTensor* xMinusMean = [mpsGraph subtractionWithPrimaryTensor:bnInputTensor
                                                                  secondaryTensor:bnMeanTensor
                                                                             name:nil];

              MPSGraphTensor* normalizedTensor = [mpsGraph multiplicationWithPrimaryTensor:xMinusMean
                                                                           secondaryTensor:bnRstdTensor
                                                                                      name:nil];

              MPSGraphTensor* bnGradMulTensor = [mpsGraph multiplicationWithPrimaryTensor:bnGradOutputTensor
                                                                          secondaryTensor:normalizedTensor
                                                                                     name:nil];

              MPSGraphTensor* gammaGradient = [mpsGraph reductionSumWithTensor:bnGradMulTensor
                                                                          axes:bn_axes
                                                                          name:nil];

              MPSGraphTensor* betaGradient = [mpsGraph reductionSumWithTensor:bnGradOutputTensor
                                                                         axes:bn_axes
                                                                         name:nil];

              MPSGraphTensor* gradient1 = [mpsGraph multiplicationWithPrimaryTensor:bnGradOutputTensor
                                                                    secondaryTensor:bnRstdTensor
                                                                               name:nil];

              MPSGraphTensor* gradient2_1 = [mpsGraph multiplicationWithPrimaryTensor:sizeReciprocalTensor
                                                                      secondaryTensor:xMinusMean
                                                                                 name:nil];

              // reverseVariance is square of rstd
              MPSGraphTensor* reverseVariance = [mpsGraph squareWithTensor:bnRstdTensor
                                                                      name:nil];
              MPSGraphTensor* gradient2_2 = [mpsGraph multiplicationWithPrimaryTensor:gammaGradient
                                                                      secondaryTensor:reverseVariance
                                                                                 name:nil];

              MPSGraphTensor* gradient2 = [mpsGraph multiplicationWithPrimaryTensor:gradient2_1
                                                                secondaryTensor:gradient2_2
                                                                           name:nil];

              MPSGraphTensor* gradient3_1 = [mpsGraph multiplicationWithPrimaryTensor:sizeReciprocalTensor
                                                                      secondaryTensor:betaGradient
                                                                                 name:nil];

              MPSGraphTensor* gradient3 = [mpsGraph multiplicationWithPrimaryTensor:gradient3_1
                                                                    secondaryTensor:bnRstdTensor
                                                                               name:nil];

              MPSGraphTensor* gradient4 = [mpsGraph subtractionWithPrimaryTensor:gradient1
                                                                 secondaryTensor:gradient2
                                                                            name:nil];

              MPSGraphTensor* gradient = [mpsGraph subtractionWithPrimaryTensor:gradient4
                                                                secondaryTensor:gradient3
                                                                           name:nil];

              gradInputTensor = [mpsGraph reshapeTensor:gradient
                                              withShape:input_shape
                                                   name:nil];

            }

            if(grad_input_mask[1]) {
              gradWeightTensor = [mpsGraph reshapeTensor:gradWeightTensor
                                               withShape:gamma_shape
                                                    name:nil];
            }
            if(grad_input_mask[2]) {
              gradBiasTensor = [mpsGraph reshapeTensor:gradBiasTensor
                                             withShape:gamma_shape
                                                  name:nil];
            }

            newCachedGraph->gradOutputTensor_ = gradOutputTensor;
            newCachedGraph->inputTensor_ = inputTensor;
            newCachedGraph->weightTensor_ = weightTensor;
            newCachedGraph->meanTensor_ = meanTensor;
            newCachedGraph->rstdTensor_ = rstdTensor;
            newCachedGraph->gradInputTensor_ = gradInputTensor;
            newCachedGraph->gradWeightTensor_ = gradWeightTensor;
            newCachedGraph->gradBiasTensor_ = gradBiasTensor;
          }
          return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
      }

      auto inputPlaceholder = native_mps::Placeholder(cachedGraph->inputTensor_, *X);
      auto gradOutputPlaceholder = native_mps::Placeholder(cachedGraph->gradOutputTensor_, *dOut);
      auto weightPlaceholder = native_mps::Placeholder();
      if(has_weight)
        weightPlaceholder = native_mps::Placeholder(cachedGraph->weightTensor_, *gamma);
      auto saveMeanPlaceholder = native_mps::Placeholder(cachedGraph->meanTensor_, mean);
      auto saveVarPlaceholder = native_mps::Placeholder(cachedGraph->rstdTensor_, rstd);

      auto gradInputPlaceholder = native_mps::Placeholder();
      if(grad_input_mask[0])
        gradInputPlaceholder = native_mps::Placeholder(cachedGraph->gradInputTensor_, grad_input);
      auto gradWeightPlaceholder = native_mps::Placeholder();
      if(grad_input_mask[1])
        gradWeightPlaceholder = native_mps::Placeholder(cachedGraph->gradWeightTensor_, grad_weight);
      auto gradBiasPlaceholder = native_mps::Placeholder();;
      if(grad_input_mask[2])
        gradBiasPlaceholder = native_mps::Placeholder(cachedGraph->gradBiasTensor_, grad_bias);

      NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
      feeds[inputPlaceholder.getMPSGraphTensor()] = inputPlaceholder.getMPSGraphTensorData();
      feeds[gradOutputPlaceholder.getMPSGraphTensor()] = gradOutputPlaceholder.getMPSGraphTensorData();
      if(has_weight)
        feeds[weightPlaceholder.getMPSGraphTensor()] = weightPlaceholder.getMPSGraphTensorData();
      feeds[saveMeanPlaceholder.getMPSGraphTensor()] = saveMeanPlaceholder.getMPSGraphTensorData();
      feeds[saveVarPlaceholder.getMPSGraphTensor()] = saveVarPlaceholder.getMPSGraphTensorData();

      NSMutableDictionary *results = [[NSMutableDictionary new] autorelease];
      if(grad_input_mask[0])
        results[gradInputPlaceholder.getMPSGraphTensor()] = gradInputPlaceholder.getMPSGraphTensorData();
      if(grad_input_mask[1])
        results[gradWeightPlaceholder.getMPSGraphTensor()] = gradWeightPlaceholder.getMPSGraphTensorData();
      if(grad_input_mask[2])
        results[gradBiasPlaceholder.getMPSGraphTensor()] = gradBiasPlaceholder.getMPSGraphTensorData();

      native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }

  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));

}

} // namespace native
} // namespace at
