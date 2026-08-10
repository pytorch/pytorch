#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSGeneratorImpl.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/_gru_mps_native.h>
#include <ATen/ops/gru_mps_backward_native.h>
#import <MetalPerformanceShadersGraph/MPSGraphRNNOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphRandomOps.h>
#include <c10/util/bit_cast.h>
#include <algorithm>

namespace at::native {
namespace {

std::vector<long long> get_tensor_shape(MPSGraphTensor* tensor) {
  std::vector<long long> dimensions;
  for (NSNumber* dimension in tensor.shape) {
    dimensions.push_back(dimension.longLongValue);
  }
  return dimensions;
}

struct GRULayerTensors {
  MPSGraphTensor* state;
  MPSGraphTensor* recurrent_weight;
  MPSGraphTensor* input_weight;
  MPSGraphTensor* bias;
  MPSGraphTensor* secondary_bias;
};

MPSGraphTensor* slice_gate(MPSGraph* graph, MPSGraphTensor* tensor, int64_t start, int64_t length) {
  return [graph sliceTensor:tensor dimension:0 start:start length:length name:nil];
}

std::pair<MPSGraphTensor*, MPSGraphTensor*> make_gru_biases(MPSGraph* graph,
                                                            MPSGraphTensor* input_bias,
                                                            MPSGraphTensor* recurrent_bias,
                                                            int64_t hidden_size) {
  auto input_rz = slice_gate(graph, input_bias, 0, hidden_size * 2);
  auto recurrent_rz = slice_gate(graph, recurrent_bias, 0, hidden_size * 2);
  auto input_new = slice_gate(graph, input_bias, hidden_size * 2, hidden_size);
  auto recurrent_new = slice_gate(graph, recurrent_bias, hidden_size * 2, hidden_size);
  auto primary_rz = [graph additionWithPrimaryTensor:input_rz secondaryTensor:recurrent_rz name:nil];
  auto primary = [graph concatTensor:primary_rz withTensor:input_new dimension:0 name:nil];
  return {primary, recurrent_new};
}

GRULayerTensors get_gru_layer_tensors(MPSGraph* graph,
                                      MPSGraphTensor* state,
                                      NSArray<MPSGraphTensor*>* recurrent_weights,
                                      NSArray<MPSGraphTensor*>* input_weights,
                                      NSArray<MPSGraphTensor*>* input_biases,
                                      NSArray<MPSGraphTensor*>* recurrent_biases,
                                      bool has_biases,
                                      bool bidirectional,
                                      int64_t layer,
                                      int64_t hidden_size) {
  const int64_t direction_count = bidirectional ? 2 : 1;
  const int64_t first = layer * direction_count;
  MPSGraphTensor* layer_state = [graph sliceTensor:state dimension:0 start:first length:direction_count name:nil];
  MPSGraphTensor* recurrent_weight = recurrent_weights[first];
  MPSGraphTensor* input_weight = input_weights[first];
  MPSGraphTensor* bias = nil;
  MPSGraphTensor* secondary_bias = nil;

  if (bidirectional) {
    layer_state = [graph transposeTensor:layer_state dimension:0 withDimension:1 name:nil];
    layer_state = [graph flatten2DTensor:layer_state axis:1 name:nil];
    recurrent_weight = [graph concatTensor:[graph expandDimsOfTensor:recurrent_weight axis:0 name:nil]
                                withTensor:[graph expandDimsOfTensor:recurrent_weights[first + 1] axis:0 name:nil]
                                 dimension:0
                                      name:nil];
    input_weight = [graph concatTensor:input_weight withTensor:input_weights[first + 1] dimension:0 name:nil];
    if (has_biases) {
      auto forward_biases = make_gru_biases(graph, input_biases[first], recurrent_biases[first], hidden_size);
      auto reverse_biases = make_gru_biases(graph, input_biases[first + 1], recurrent_biases[first + 1], hidden_size);
      bias = [graph concatTensor:forward_biases.first withTensor:reverse_biases.first dimension:0 name:nil];
      secondary_bias = [graph concatTensor:forward_biases.second withTensor:reverse_biases.second dimension:0 name:nil];
    }
  } else {
    layer_state = [graph squeezeTensor:layer_state axis:0 name:nil];
    if (has_biases) {
      std::tie(bias, secondary_bias) =
          make_gru_biases(graph, input_biases[first], recurrent_biases[first], hidden_size);
    }
  }

  return {layer_state, recurrent_weight, input_weight, bias, secondary_bias};
}

MPSGraphTensor* apply_dropout(MPSGraph* graph,
                              MPSGraphTensor* input,
                              MPSGraphTensor*& random_state,
                              double dropout_p,
                              MPSDataType dtype,
                              NSMutableArray<MPSGraphTensor*>* masks) {
  MPSGraphRandomOpDescriptor* descriptor =
      [MPSGraphRandomOpDescriptor descriptorWithDistribution:MPSGraphRandomDistributionUniform
                                                    dataType:MPSDataTypeFloat32];
  descriptor.min = 0.0f;
  descriptor.max = 1.0f;
  auto shape = [graph shapeOfTensor:input name:nil];
  auto random_results = [graph randomTensorWithShapeTensor:shape
                                                descriptor:descriptor
                                               stateTensor:random_state
                                                      name:nil];
  auto threshold = [graph constantWithScalar:dropout_p dataType:MPSDataTypeFloat32];
  auto keep = [graph greaterThanOrEqualToWithPrimaryTensor:random_results[0] secondaryTensor:threshold name:nil];
  auto keep_cast = [graph castTensor:keep toType:dtype name:nil];
  auto scale = [graph constantWithScalar:(dropout_p >= 1.0 ? 0.0 : 1.0 / (1.0 - dropout_p)) dataType:dtype];
  auto mask = [graph multiplicationWithPrimaryTensor:keep_cast secondaryTensor:scale name:nil];
  random_state = random_results[1];
  [masks addObject:[graph expandDimsOfTensor:mask axis:0 name:nil]];
  return [graph multiplicationWithPrimaryTensor:input secondaryTensor:mask name:nil];
}

} // namespace

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> _gru_mps(const Tensor& input,
                                                            const Tensor& hx,
                                                            TensorList params,
                                                            bool has_biases,
                                                            int64_t num_layers,
                                                            double dropout_p,
                                                            bool train,
                                                            bool bidirectional,
                                                            bool batch_first) {
  using namespace mps;

  const int64_t directions = bidirectional ? 2 : 1;
  const int64_t total_layers = num_layers * directions;
  const int64_t param_stride = has_biases ? 4 : 2;
  const int64_t hidden_size = hx.size(2);
  const bool needs_grad = GradMode::is_enabled() &&
      (input.requires_grad() || hx.requires_grad() ||
       std::any_of(params.begin(), params.end(), [](const Tensor& param) { return param.requires_grad(); }));

  std::vector<Tensor> input_weights;
  std::vector<Tensor> recurrent_weights;
  std::vector<Tensor> input_biases;
  std::vector<Tensor> recurrent_biases;
  input_weights.reserve(total_layers);
  recurrent_weights.reserve(total_layers);
  input_biases.reserve(total_layers);
  recurrent_biases.reserve(total_layers);
  for (const auto i : c10::irange(total_layers)) {
    input_weights.push_back(params[i * param_stride]);
    recurrent_weights.push_back(params[i * param_stride + 1]);
    if (has_biases) {
      input_biases.push_back(params[i * param_stride + 2]);
      recurrent_biases.push_back(params[i * param_stride + 3]);
    }
  }

  struct CachedGraph : public MPSCachedGraph {
    explicit CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    std::vector<MPSGraphTensor*> inputs;
    MPSGraphTensor* output = nil;
    MPSGraphTensor* hy = nil;
    MPSGraphTensor* z_state = nil;
    MPSGraphTensor* output_fwd = nil;
    MPSGraphTensor* layer_inputs = nil;
    NSMutableArray<MPSGraphTensor*>* input_weights = nil;
    NSMutableArray<MPSGraphTensor*>* recurrent_weights = nil;
    NSMutableArray<MPSGraphTensor*>* input_biases = nil;
    NSMutableArray<MPSGraphTensor*>* recurrent_biases = nil;
    MPSGraphTensor* dropout_state = nil;
  };

  const bool use_dropout = dropout_p > 0.0 && train && num_layers > 1;
  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto key = "gru_" + getTensorsStringKey({input, hx}) + getMPSTypeString(input) + "_layers_" +
        std::to_string(num_layers) + "_bidirectional_" + std::to_string(bidirectional) + "_bias_" +
        std::to_string(has_biases) + "_dropout_" + std::to_string(c10::bit_cast<uint64_t>(dropout_p)) +
        "_batch_first_" + std::to_string(batch_first) + "_train_" + std::to_string(train) + "_needs_grad_" +
        std::to_string(needs_grad);
    auto cached_graph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](MPSGraph* graph, CachedGraph* cache) {
      auto graph_input_weights = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto graph_recurrent_weights = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto graph_input_biases = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto graph_recurrent_biases = [[NSMutableArray alloc] initWithCapacity:total_layers];
      for (const auto i : c10::irange(total_layers)) {
        [graph_input_weights
            addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(input_weights[i]))];
        [graph_recurrent_weights
            addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(recurrent_weights[i]))];
        if (has_biases) {
          [graph_input_biases
              addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(input_biases[i]))];
          [graph_recurrent_biases
              addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(recurrent_biases[i]))];
        }
      }

      auto input_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(input));
      auto state_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(hx), getMPSShape(hx));
      cache->inputs = {input_tensor, state_tensor};
      if (batch_first) {
        input_tensor = [graph transposeTensor:input_tensor dimension:0 withDimension:1 name:nil];
      }

      MPSGraphTensor* random_state = nil;
      if (use_dropout) {
        cache->dropout_state =
            mpsGraphRankedPlaceHolder(graph, MPSDataTypeInt32, @[ @(at::mps::detail::PHILOX_STATE_N) ]);
        random_state = cache->dropout_state;
      }

      auto descriptor = [MPSGraphGRUDescriptor descriptor];
      descriptor.training = needs_grad;
      descriptor.bidirectional = bidirectional;
      descriptor.resetGateFirst = true;
      descriptor.resetAfter = true;
      descriptor.flipZ = false;

      auto layer_inputs = [[NSMutableArray alloc] initWithCapacity:std::max<int64_t>(1, num_layers - 1)];
      auto dropout_masks = [[NSMutableArray alloc] initWithCapacity:std::max<int64_t>(1, num_layers - 1)];
      auto output_states = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto z_states = [[NSMutableArray alloc] initWithCapacity:num_layers];
      auto forward_outputs = [[NSMutableArray alloc] initWithCapacity:num_layers];

      MPSGraphTensor* layer_input = input_tensor;
      for (const auto layer : c10::irange(num_layers)) {
        auto tensors = get_gru_layer_tensors(graph,
                                             state_tensor,
                                             graph_recurrent_weights,
                                             graph_input_weights,
                                             graph_input_biases,
                                             graph_recurrent_biases,
                                             has_biases,
                                             bidirectional,
                                             layer,
                                             hidden_size);
        auto results = [graph GRUWithSourceTensor:layer_input
                                  recurrentWeight:tensors.recurrent_weight
                                      inputWeight:tensors.input_weight
                                             bias:tensors.bias
                                        initState:tensors.state
                                             mask:nil
                                    secondaryBias:tensors.secondary_bias
                                       descriptor:descriptor
                                             name:nil];
        auto raw_output = results[0];
        if (needs_grad) {
          [forward_outputs addObject:[graph expandDimsOfTensor:raw_output axis:0 name:nil]];
          [z_states addObject:[graph expandDimsOfTensor:results[1] axis:0 name:nil]];
        }

        if (bidirectional) {
          auto last = [graph sliceTensor:raw_output dimension:0 start:-1 length:1 name:nil];
          auto first = [graph sliceTensor:raw_output dimension:0 start:0 length:1 name:nil];
          [output_states addObject:[graph sliceTensor:last dimension:-1 start:0 length:hidden_size name:nil]];
          [output_states addObject:[graph sliceTensor:first
                                            dimension:-1
                                                start:hidden_size
                                               length:hidden_size
                                                 name:nil]];
        } else {
          [output_states addObject:[graph sliceTensor:raw_output dimension:0 start:-1 length:1 name:nil]];
        }

        layer_input = raw_output;
        if (use_dropout && layer != num_layers - 1) {
          layer_input =
              apply_dropout(graph, layer_input, random_state, dropout_p, getMPSDataType(input), dropout_masks);
        }
        if (needs_grad && layer != num_layers - 1) {
          [layer_inputs addObject:[graph expandDimsOfTensor:layer_input axis:0 name:nil]];
        }
      }

      auto output = layer_input;
      if (batch_first) {
        output = [graph transposeTensor:output dimension:0 withDimension:1 name:nil];
      }
      auto hy = [graph concatTensors:output_states dimension:0 name:nil];
      cache->output = output;
      cache->hy = hy;
      if (needs_grad) {
        cache->z_state = [graph concatTensors:z_states dimension:0 name:nil];
        cache->output_fwd = [graph concatTensors:forward_outputs dimension:0 name:nil];
        if (use_dropout) {
          [layer_inputs addObjectsFromArray:dropout_masks];
        }
        if (num_layers > 1) {
          cache->layer_inputs = [graph concatTensors:layer_inputs dimension:0 name:nil];
        }
      }
      cache->input_weights = graph_input_weights;
      cache->recurrent_weights = graph_recurrent_weights;
      cache->input_biases = graph_input_biases;
      cache->recurrent_biases = graph_recurrent_biases;
    });

    auto feeds = [[[NSMutableDictionary alloc] init] autorelease];
    Placeholder input_placeholder(cached_graph->inputs[0], input);
    Placeholder state_placeholder(cached_graph->inputs[1], hx);
    feeds[input_placeholder.getMPSGraphTensor()] = input_placeholder.getMPSGraphTensorData();
    feeds[state_placeholder.getMPSGraphTensor()] = state_placeholder.getMPSGraphTensorData();
    for (const auto i : c10::irange(total_layers)) {
      Placeholder input_weight(cached_graph->input_weights[i], input_weights[i]);
      Placeholder recurrent_weight(cached_graph->recurrent_weights[i], recurrent_weights[i]);
      feeds[input_weight.getMPSGraphTensor()] = input_weight.getMPSGraphTensorData();
      feeds[recurrent_weight.getMPSGraphTensor()] = recurrent_weight.getMPSGraphTensorData();
      if (has_biases) {
        Placeholder input_bias(cached_graph->input_biases[i], input_biases[i]);
        Placeholder recurrent_bias(cached_graph->recurrent_biases[i], recurrent_biases[i]);
        feeds[input_bias.getMPSGraphTensor()] = input_bias.getMPSGraphTensorData();
        feeds[recurrent_bias.getMPSGraphTensor()] = recurrent_bias.getMPSGraphTensorData();
      }
    }
    if (cached_graph->dropout_state) {
      auto generator =
          get_generator_or_default<at::MPSGeneratorImpl>(std::nullopt, at::mps::detail::getDefaultMPSGenerator());
      auto descriptor = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeInt32
                                                               shape:@[ @(at::mps::detail::PHILOX_STATE_N) ]];
      auto array = [[[MPSNDArray alloc] initWithDevice:stream->device() descriptor:descriptor] autorelease];
      {
        std::lock_guard<std::mutex> lock(generator->mutex_);
        generator->update_philox_counters();
        [array writeBytes:generator->state_data() strideBytes:nil];
      }
      feeds[cached_graph->dropout_state] = [[[MPSGraphTensorData alloc] initWithMPSNDArray:array] autorelease];
    }

    Tensor output = at::empty(get_tensor_shape(cached_graph->output), input.options());
    Tensor hy = at::empty_like(hx);
    Tensor z_state = needs_grad ? at::empty(get_tensor_shape(cached_graph->z_state), input.options())
                                : at::empty({0}, input.options());
    Tensor output_fwd = needs_grad ? at::empty(get_tensor_shape(cached_graph->output_fwd), input.options())
                                   : at::empty({0}, input.options());
    Tensor layer_inputs = needs_grad && num_layers > 1
        ? at::empty(get_tensor_shape(cached_graph->layer_inputs), input.options())
        : at::empty({0}, input.options());

    auto results = [[[NSMutableDictionary alloc] init] autorelease];
    Placeholder output_placeholder(cached_graph->output, output);
    Placeholder hy_placeholder(cached_graph->hy, hy);
    results[output_placeholder.getMPSGraphTensor()] = output_placeholder.getMPSGraphTensorData();
    results[hy_placeholder.getMPSGraphTensor()] = hy_placeholder.getMPSGraphTensorData();
    if (needs_grad) {
      Placeholder z_placeholder(cached_graph->z_state, z_state);
      Placeholder output_fwd_placeholder(cached_graph->output_fwd, output_fwd);
      results[z_placeholder.getMPSGraphTensor()] = z_placeholder.getMPSGraphTensorData();
      results[output_fwd_placeholder.getMPSGraphTensor()] = output_fwd_placeholder.getMPSGraphTensorData();
      if (num_layers > 1) {
        Placeholder layer_inputs_placeholder(cached_graph->layer_inputs, layer_inputs);
        results[layer_inputs_placeholder.getMPSGraphTensor()] = layer_inputs_placeholder.getMPSGraphTensorData();
      }
    }

    runMPSGraph(stream, cached_graph->graph(), feeds, results);
    return {output, hy, z_state, output_fwd, layer_inputs};
  }
}

std::tuple<Tensor, Tensor, std::vector<Tensor>> gru_mps_backward(const std::optional<Tensor>& grad_y_opt,
                                                                 const std::optional<Tensor>& grad_hy_opt,
                                                                 const Tensor& z_state,
                                                                 const Tensor& output_fwd,
                                                                 const Tensor& input,
                                                                 const Tensor& layer_inputs,
                                                                 const Tensor& hx,
                                                                 TensorList params,
                                                                 bool has_biases,
                                                                 int64_t num_layers,
                                                                 double dropout_p,
                                                                 bool train,
                                                                 bool bidirectional,
                                                                 bool batch_first) {
  using namespace mps;

  const int64_t directions = bidirectional ? 2 : 1;
  const int64_t total_layers = num_layers * directions;
  const int64_t param_stride = has_biases ? 4 : 2;
  const int64_t hidden_size = hx.size(2);
  const int64_t batch_size = hx.size(1);
  const int64_t sequence_length = input.size(batch_first ? 1 : 0);
  const bool use_dropout = dropout_p > 0.0 && train && num_layers > 1;

  const auto grad_y = grad_y_opt.has_value() && grad_y_opt->defined()
      ? *grad_y_opt
      : at::zeros({batch_first ? batch_size : sequence_length,
                   batch_first ? sequence_length : batch_size,
                   hidden_size * directions},
                  input.options());
  const auto grad_hy = grad_hy_opt.has_value() && grad_hy_opt->defined() ? *grad_hy_opt : at::zeros_like(hx);

  std::vector<Tensor> input_weights;
  std::vector<Tensor> recurrent_weights;
  std::vector<Tensor> input_biases;
  std::vector<Tensor> recurrent_biases;
  input_weights.reserve(total_layers);
  recurrent_weights.reserve(total_layers);
  input_biases.reserve(total_layers);
  recurrent_biases.reserve(total_layers);
  for (const auto i : c10::irange(total_layers)) {
    input_weights.push_back(params[i * param_stride]);
    recurrent_weights.push_back(params[i * param_stride + 1]);
    if (has_biases) {
      input_biases.push_back(params[i * param_stride + 2]);
      recurrent_biases.push_back(params[i * param_stride + 3]);
    }
  }

  struct CachedGraph : public MPSCachedGraph {
    explicit CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
    std::vector<MPSGraphTensor*> inputs;
    NSMutableArray<MPSGraphTensor*>* input_weights = nil;
    NSMutableArray<MPSGraphTensor*>* recurrent_weights = nil;
    NSMutableArray<MPSGraphTensor*>* input_biases = nil;
    NSMutableArray<MPSGraphTensor*>* recurrent_biases = nil;
    MPSGraphTensor* grad_input = nil;
    MPSGraphTensor* grad_hx = nil;
    NSMutableArray<MPSGraphTensor*>* grad_input_weights = nil;
    NSMutableArray<MPSGraphTensor*>* grad_recurrent_weights = nil;
    NSMutableArray<MPSGraphTensor*>* grad_input_biases = nil;
    NSMutableArray<MPSGraphTensor*>* grad_recurrent_biases = nil;
  };

  auto stream = getCurrentMPSStream();
  @autoreleasepool {
    auto key = "gru_backward_" + getTensorsStringKey({input, z_state, output_fwd, grad_y, grad_hy}) +
        getMPSTypeString(input) + "_layers_" + std::to_string(num_layers) + "_bidirectional_" +
        std::to_string(bidirectional) + "_bias_" + std::to_string(has_biases) + "_dropout_" +
        std::to_string(c10::bit_cast<uint64_t>(dropout_p)) + "_batch_first_" + std::to_string(batch_first) + "_train_" +
        std::to_string(train);
    auto cached_graph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](MPSGraph* graph, CachedGraph* cache) {
      auto graph_input_weights = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto graph_recurrent_weights = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto graph_input_biases = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto graph_recurrent_biases = [[NSMutableArray alloc] initWithCapacity:total_layers];
      for (const auto i : c10::irange(total_layers)) {
        [graph_input_weights
            addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(input_weights[i]))];
        [graph_recurrent_weights
            addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(recurrent_weights[i]))];
        if (has_biases) {
          [graph_input_biases
              addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(input_biases[i]))];
          [graph_recurrent_biases
              addObject:mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(recurrent_biases[i]))];
        }
      }

      auto input_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(input), getMPSShape(input));
      auto state_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(hx), getMPSShape(hx));
      auto z_state_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(z_state), getMPSShape(z_state));
      auto output_fwd_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(output_fwd), getMPSShape(output_fwd));
      auto grad_y_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(grad_y), getMPSShape(grad_y));
      auto grad_hy_tensor = mpsGraphRankedPlaceHolder(graph, getMPSDataType(grad_hy), getMPSShape(grad_hy));
      auto layer_inputs_tensor =
          mpsGraphRankedPlaceHolder(graph, getMPSDataType(layer_inputs), getMPSShape(layer_inputs));
      cache->inputs = {input_tensor,
                       state_tensor,
                       z_state_tensor,
                       output_fwd_tensor,
                       grad_y_tensor,
                       grad_hy_tensor,
                       layer_inputs_tensor};

      if (batch_first) {
        input_tensor = [graph transposeTensor:input_tensor dimension:0 withDimension:1 name:nil];
        grad_y_tensor = [graph transposeTensor:grad_y_tensor dimension:0 withDimension:1 name:nil];
      }

      auto descriptor = [MPSGraphGRUDescriptor descriptor];
      descriptor.training = true;
      descriptor.bidirectional = bidirectional;
      descriptor.resetGateFirst = true;
      descriptor.resetAfter = true;
      descriptor.flipZ = false;

      auto grad_input_weights = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto grad_recurrent_weights = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto grad_input_biases = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto grad_recurrent_biases = [[NSMutableArray alloc] initWithCapacity:total_layers];
      auto grad_states = [[NSMutableArray alloc] initWithCapacity:total_layers];

      MPSGraphTensor* layer_gradient = grad_y_tensor;
      for (int64_t layer = num_layers - 1; layer >= 0; --layer) {
        auto saved_z = [graph sliceTensor:z_state_tensor dimension:0 start:layer length:1 name:nil];
        saved_z = [graph squeezeTensor:saved_z axis:0 name:nil];
        auto saved_output = [graph sliceTensor:output_fwd_tensor dimension:0 start:layer length:1 name:nil];
        saved_output = [graph squeezeTensor:saved_output axis:0 name:nil];

        auto tensors = get_gru_layer_tensors(graph,
                                             state_tensor,
                                             graph_recurrent_weights,
                                             graph_input_weights,
                                             graph_input_biases,
                                             graph_recurrent_biases,
                                             has_biases,
                                             bidirectional,
                                             layer,
                                             hidden_size);
        MPSGraphTensor* source = input_tensor;
        if (layer > 0) {
          source = [graph sliceTensor:layer_inputs_tensor dimension:0 start:layer - 1 length:1 name:nil];
          source = [graph squeezeTensor:source axis:0 name:nil];
        }

        MPSGraphTensor* state_gradient = nil;
        if (bidirectional) {
          state_gradient = [graph sliceTensor:grad_hy_tensor dimension:0 start:layer * 2 length:2 name:nil];
          state_gradient = [graph transposeTensor:state_gradient dimension:0 withDimension:1 name:nil];
          state_gradient = [graph flatten2DTensor:state_gradient axis:1 name:nil];
        } else {
          state_gradient = [graph sliceTensor:grad_hy_tensor dimension:0 start:layer length:1 name:nil];
          state_gradient = [graph squeezeTensor:state_gradient axis:0 name:nil];
        }

        auto gradients = [graph GRUGradientsWithSourceTensor:source
                                             recurrentWeight:tensors.recurrent_weight
                                              sourceGradient:layer_gradient
                                                      zState:saved_z
                                                   outputFwd:saved_output
                                               stateGradient:state_gradient
                                                 inputWeight:tensors.input_weight
                                                        bias:tensors.bias
                                                   initState:tensors.state
                                                        mask:nil
                                               secondaryBias:tensors.secondary_bias
                                                  descriptor:descriptor
                                                        name:nil];
        layer_gradient = gradients[0];
        if (use_dropout && layer > 0) {
          auto mask = [graph sliceTensor:layer_inputs_tensor
                               dimension:0
                                   start:(num_layers - 1) + (layer - 1)
                                  length:1
                                    name:nil];
          mask = [graph squeezeTensor:mask axis:0 name:nil];
          layer_gradient = [graph multiplicationWithPrimaryTensor:layer_gradient secondaryTensor:mask name:nil];
        }

        auto recurrent_gradient = gradients[1];
        auto input_gradient = gradients[2];
        int64_t output_index = 3;
        MPSGraphTensor* primary_bias_gradient = has_biases ? gradients[output_index++] : nil;
        auto initial_state_gradient = gradients[output_index++];
        MPSGraphTensor* secondary_bias_gradient = has_biases ? gradients[output_index] : nil;

        if (has_biases) {
          primary_bias_gradient = [graph reshapeTensor:primary_bias_gradient
                                             withShape:@[ @(directions * hidden_size * 3) ]
                                                  name:nil];
          secondary_bias_gradient = [graph reshapeTensor:secondary_bias_gradient
                                               withShape:@[ @(directions * hidden_size) ]
                                                    name:nil];
        }

        if (bidirectional) {
          for (int64_t direction = directions - 1; direction >= 0; --direction) {
            auto grad_recurrent = [graph sliceTensor:recurrent_gradient dimension:0 start:direction length:1 name:nil];
            grad_recurrent = [graph squeezeTensor:grad_recurrent axis:0 name:nil];
            auto grad_weight = [graph sliceTensor:input_gradient
                                        dimension:0
                                            start:direction * hidden_size * 3
                                           length:hidden_size * 3
                                             name:nil];
            [grad_recurrent_weights insertObject:grad_recurrent atIndex:0];
            [grad_input_weights insertObject:grad_weight atIndex:0];

            auto grad_state = [graph sliceTensor:initial_state_gradient
                                       dimension:1
                                           start:direction * hidden_size
                                          length:hidden_size
                                            name:nil];
            [grad_states insertObject:[graph expandDimsOfTensor:grad_state axis:0 name:nil] atIndex:0];

            if (has_biases) {
              auto grad_primary = [graph sliceTensor:primary_bias_gradient
                                           dimension:0
                                               start:direction * hidden_size * 3
                                              length:hidden_size * 3
                                                name:nil];
              auto grad_secondary = [graph sliceTensor:secondary_bias_gradient
                                             dimension:0
                                                 start:direction * hidden_size
                                                length:hidden_size
                                                  name:nil];
              auto grad_primary_rz = [graph sliceTensor:grad_primary
                                              dimension:0
                                                  start:0
                                                 length:hidden_size * 2
                                                   name:nil];
              auto grad_recurrent_bias = [graph concatTensor:grad_primary_rz
                                                  withTensor:grad_secondary
                                                   dimension:0
                                                        name:nil];
              [grad_input_biases insertObject:grad_primary atIndex:0];
              [grad_recurrent_biases insertObject:grad_recurrent_bias atIndex:0];
            }
          }
        } else {
          [grad_recurrent_weights insertObject:recurrent_gradient atIndex:0];
          [grad_input_weights insertObject:input_gradient atIndex:0];
          [grad_states insertObject:[graph expandDimsOfTensor:initial_state_gradient axis:0 name:nil] atIndex:0];
          if (has_biases) {
            auto grad_primary_rz = [graph sliceTensor:primary_bias_gradient
                                            dimension:0
                                                start:0
                                               length:hidden_size * 2
                                                 name:nil];
            auto grad_recurrent_bias = [graph concatTensor:grad_primary_rz
                                                withTensor:secondary_bias_gradient
                                                 dimension:0
                                                      name:nil];
            [grad_input_biases insertObject:primary_bias_gradient atIndex:0];
            [grad_recurrent_biases insertObject:grad_recurrent_bias atIndex:0];
          }
        }
      }

      cache->grad_input =
          batch_first ? [graph transposeTensor:layer_gradient dimension:0 withDimension:1 name:nil] : layer_gradient;
      cache->grad_hx = [graph concatTensors:grad_states dimension:0 name:nil];
      cache->input_weights = graph_input_weights;
      cache->recurrent_weights = graph_recurrent_weights;
      cache->input_biases = graph_input_biases;
      cache->recurrent_biases = graph_recurrent_biases;
      cache->grad_input_weights = grad_input_weights;
      cache->grad_recurrent_weights = grad_recurrent_weights;
      cache->grad_input_biases = grad_input_biases;
      cache->grad_recurrent_biases = grad_recurrent_biases;
    });

    auto feeds = [[[NSMutableDictionary alloc] init] autorelease];
    std::array<Tensor, 7> input_tensors = {input, hx, z_state, output_fwd, grad_y, grad_hy, layer_inputs};
    for (const auto i : c10::irange(input_tensors.size())) {
      Placeholder placeholder(cached_graph->inputs[i], input_tensors[i]);
      feeds[placeholder.getMPSGraphTensor()] = placeholder.getMPSGraphTensorData();
    }
    for (const auto i : c10::irange(total_layers)) {
      Placeholder input_weight(cached_graph->input_weights[i], input_weights[i]);
      Placeholder recurrent_weight(cached_graph->recurrent_weights[i], recurrent_weights[i]);
      feeds[input_weight.getMPSGraphTensor()] = input_weight.getMPSGraphTensorData();
      feeds[recurrent_weight.getMPSGraphTensor()] = recurrent_weight.getMPSGraphTensorData();
      if (has_biases) {
        Placeholder input_bias(cached_graph->input_biases[i], input_biases[i]);
        Placeholder recurrent_bias(cached_graph->recurrent_biases[i], recurrent_biases[i]);
        feeds[input_bias.getMPSGraphTensor()] = input_bias.getMPSGraphTensorData();
        feeds[recurrent_bias.getMPSGraphTensor()] = recurrent_bias.getMPSGraphTensorData();
      }
    }

    Tensor grad_input = at::empty_like(input);
    Tensor grad_hx = at::empty_like(hx);
    auto results = [[[NSMutableDictionary alloc] init] autorelease];
    Placeholder grad_input_placeholder(cached_graph->grad_input, grad_input);
    Placeholder grad_hx_placeholder(cached_graph->grad_hx, grad_hx);
    results[grad_input_placeholder.getMPSGraphTensor()] = grad_input_placeholder.getMPSGraphTensorData();
    results[grad_hx_placeholder.getMPSGraphTensor()] = grad_hx_placeholder.getMPSGraphTensorData();

    std::vector<Tensor> grad_params;
    grad_params.reserve(params.size());
    for (const auto i : c10::irange(total_layers)) {
      Tensor grad_input_weight = at::empty_like(input_weights[i]);
      Tensor grad_recurrent_weight = at::empty_like(recurrent_weights[i]);
      grad_params.push_back(grad_input_weight);
      grad_params.push_back(grad_recurrent_weight);
      Placeholder grad_input_weight_placeholder(cached_graph->grad_input_weights[i], grad_input_weight);
      Placeholder grad_recurrent_weight_placeholder(cached_graph->grad_recurrent_weights[i], grad_recurrent_weight);
      results[grad_input_weight_placeholder.getMPSGraphTensor()] =
          grad_input_weight_placeholder.getMPSGraphTensorData();
      results[grad_recurrent_weight_placeholder.getMPSGraphTensor()] =
          grad_recurrent_weight_placeholder.getMPSGraphTensorData();
      if (has_biases) {
        Tensor grad_input_bias = at::empty_like(input_biases[i]);
        Tensor grad_recurrent_bias = at::empty_like(recurrent_biases[i]);
        grad_params.push_back(grad_input_bias);
        grad_params.push_back(grad_recurrent_bias);
        Placeholder grad_input_bias_placeholder(cached_graph->grad_input_biases[i], grad_input_bias);
        Placeholder grad_recurrent_bias_placeholder(cached_graph->grad_recurrent_biases[i], grad_recurrent_bias);
        results[grad_input_bias_placeholder.getMPSGraphTensor()] = grad_input_bias_placeholder.getMPSGraphTensorData();
        results[grad_recurrent_bias_placeholder.getMPSGraphTensor()] =
            grad_recurrent_bias_placeholder.getMPSGraphTensorData();
      }
    }

    runMPSGraph(stream, cached_graph->graph(), feeds, results);
    return {grad_input, grad_hx, grad_params};
  }
}

} // namespace at::native
