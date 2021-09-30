#include <torch/csrc/jit/codegen/cuda/partition.h>

#include <ATen/core/jit_type.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

bool hasNonElementWiseOperation(const Node* node) {
  if (node->kind() == prim::CudaFusionGroup) {
    for (auto n : node->g(attr::Subgraph)->nodes()) {
      if (hasNonElementWiseOperation(n)) {
        return true;
      }
    }
  } else {
    // prim::Constant is not parsible, but it is also not nonElementWise
    if (node->kind() != prim::Constant && !isElementWiseNode(node)) {
      return true;
    }
  }
  return false;
}

// Check all outputs are:
//   1. TensorType
//   2. on the same device;
// TODO: update this when codegen can output scalar
static c10::optional<c10::Device> getDevice(const Value* value) {
  if (!value->type()->isSubtypeOf(TensorType::get())) {
    // not tensor type, return false as the op is not outputing scalar.
    return c10::nullopt;
  }
  return value->type()->expectRef<TensorType>().device();
}

static c10::optional<c10::Device> getDevice(const Node* node) {
  auto outputs = node->outputs();
  for (auto output : outputs) {
    auto device = getDevice(output);
    if (device.has_value()) {
      return device;
    }
  }
  return c10::nullopt;
}

static bool isFusibleDevice(const Node* node, const c10::Device device) {
  for (auto value : node->outputs()) {
    auto output_device = getDevice(value);
    if (output_device.has_value() && output_device.value() != device) {
      return false;
    }
  }
  return true;
}

// TODO: we need to check input type when we handle `to()`
static bool isFusibleDevice(const Node* node) {
  auto device = getDevice(node);
  if (!device.has_value()) {
    return true;
  }
  return device->is_cuda() &&
      (at::cuda::getDeviceProperties(device->index())->major >= 7 ||
       !hasNonElementWiseOperation(node));
}

bool compatibleType(const torch::jit::Value* val) {
  if (auto tensor_type = val->type()->cast<c10::TensorType>()) {
    if (tensor_type->scalarType().has_value()) {
      if (aten_to_data_type(tensor_type->scalarType().value()) ==
          DataType::Null) {
        return false;
      }
    }
  }
  return true;
}

bool checkInputTensorTypes(const Node* node) {
  for (size_t i = 0; i < node->inputs().size(); i++) {
    const auto& val = node->inputs()[i];
    if (!compatibleType(val)) {
      // special case on aten::_batch_norm_impl_index_backward, the 11th output
      // is going to be discarded, so no need to check data type there.
      if (node->kind() ==
              c10::Symbol::fromQualString(
                  "aten::_batch_norm_impl_index_backward") &&
          i == 11) {
        continue;
      }
      return false;
    }
  }
  return true;
}

bool checkOutputTensorTypes(const Node* node) {
  for (size_t i = 0; i < node->outputs().size(); i++) {
    const auto& val = node->outputs()[i];
    if (!compatibleType(val)) {
      // special case on aten::_batch_norm_impl_index, the 4th output
      // is going to be discarded, so no need to check data type there.
      if (node->kind() ==
              c10::Symbol::fromQualString("aten::_batch_norm_impl_index") &&
          i == 3) {
        continue;
      }
      return false;
    }
  }
  return true;
}

inline bool isFusibleNode(const Node* node) {
  if (node->kind() == prim::CudaFusionGroup)
    return true;
  // Check we have a parsing rule
  bool isFusible = isNodeParsible(node);
  // Check if we have a tensor type it's one we support
  isFusible = isFusible && checkInputTensorTypes(node);
  isFusible = isFusible && checkOutputTensorTypes(node);
  // Check if already part of a fusion group
  return isFusible;
}

bool maybeBroadcast(
    const TensorTypePtr& type,
    const std::vector<c10::optional<int64_t>>& shape) {
  if (type->dim()) {
    if (type->dim().value() < shape.size()) {
      // no broadcast for reduction operation;
      return false;
    } else if (type->dim().value() > shape.size()) {
      // increased rank means there is reduction;
      return true;
    } else {
      // same rank, we need to iterate through sizes and check if size-1
      // exists in input `shape`
      for (const auto& opt_size : shape) {
        // TODO: not sure if we need to check for output size != 1, since we
        // are currently marking all size-1 dimension as broadcast in codegen.
        if (opt_size.has_value() && opt_size.value() == 1) {
          return true;
        }
      }
    }
  }
  return false;
}

// utility function to check if the node implies broadcast on a given shape (
// assumed to be shape of an input tensor)
// limitations:
//   1. we rely on shape information to judge this. so we would require output
//      shape to be available;
//   2. we basically compares given shape to the shape of the only output of
//      the node and return true if it implies broadcast from the former to the
//      latter.
bool maybeBroadcastOnShape(
    const Node* n,
    const std::vector<c10::optional<int64_t>>& shape) {
  // TODO: we are only checking output 0. This means that our current check for
  // normalization is not complete.
  // assumes that if output is not a tensor type, it's not broadcasting
  if (auto out_type = n->output(0)->type()->cast<TensorType>()) {
    return maybeBroadcast(out_type, shape);
  }
  return false;
};

// return true if node is pointwise operation and input tensors all have
// identical shape.
bool isNonBroadcastElementWise(const Node* n) {
  if (hasNonElementWiseOperation(n)) {
    return false;
  }

  for (const auto output : n->outputs()) {
    const auto& n_output_type = output->type()->cast<TensorType>();

    // TODO: we need to stay on safer side instead of "default to return true
    // when shape information is not available.", Change that when we enable
    // profiling on autodiff FW execution.
    if (n_output_type != nullptr && n_output_type->sizes().sizes()) {
      const std::vector<c10::optional<int64_t>>& n_output_shape =
          n_output_type->sizes().sizes().value();

      for (auto input : n->inputs()) {
        if (auto t_type = input->type()->cast<TensorType>()) {
          if (maybeBroadcast(t_type, n_output_shape)) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

//! [ Note - tricky broadcasting ]
//!
//! github issue # 190
//!
//! To extend the issue further, we consider two difficult broadcasting cases
//! that is difficult to naively schedule:
//!   scenario 1: single tensor with multiple broadcasting semantics;
//!               ```
//!                   %t = op(...)
//!                   %t0_o = op0(%t, %t0)
//!                   %t1_o = op1(%t, %t1)
//!               ```
//!               It's hard to check/validate whether `%t0` and `%t1` implies
//!               identical broadcasting for `%t` so that we can simply
//!               broadcast it to their common shape and use the broadcasted
//!               tensor view in both `op0` and `op1`; or, if `%t0` and `%t1`
//!               has different shapes, we would need differently broadcasted
//!               `%t` for the two ops. Even with this condition sorted out,
//!               scheduling is challenging. As we cannot inline the computation
//!               of `%t` to the downstream consumer of `%t0_o` and `%t1_o`
//!               easily, because `computeAt` could propagate contradicting
//!               transformations on the common ancestor `%t`. See footnote*;
//!   scenario 2: output tensor_view which is broadcasted later;
//!               ```
//!                   %t = op(...)
//!                   %t0_o = op0(%t, %t0)
//!                   return (%t, %t0_o)
//!               ```
//!               Similarly, if we need to broadcast `%t` to `%t0` for `op0`,
//!               and use it as output, it also complicates schedule.
//!
//! Currently we just avoid the two cases in our graph partitioning.
//!
//! We bake the implementation along with our partition, where we merge nodes
//! from producer to consumer. In the example down, we list all "type"s of edges
//! among producer/consumer and the out side world.
//!
//!   %input_t0, %input_t1, %input_t2 # inputs from outside world feeding
//!                                   # producer/consumer pair
//!   %p_out_t0, %p_out_t1 = producer(%input_t0, %input_t1)
//!   %c_out_t, ... = consumer(%input_t0, %input_t2, %p_out_t0)
//!
//! producer/consumer : the nodes that we are trying to merge, each node could
//! be
//!                     a parsible real operation or a `CudaFusionGroup`.
//! %input_t0         : inputs shared by both producer & consumer
//! %input_t1         : inputs feed only to producer, but not to consumer
//! %input_t2         : inputs feed only to consumer, but not to producer
//! %p_put_t0         : outputs of producer that is fed to consumer
//! %p_put_t1         : outputs of producer that is not fed to consumer
//! %c_put_t0         : outputs of consumer
//!
//! We can see that after merging consumer & producer, we will have:
//!   %input_t0, %input_t1, %input_t2 # inputs from outside world feeding
//!                                   # producer/consumer pair
//!   %p_out_t, %c_out_t = group(%input_t0, %input_t1, %input_t2)
//!
//! Under the assumption that any existing `CudaFusionGroup` does not have
//! violating broadcasting semantics mentioned above.
//!
//! If we examine the `group`, new cases of scenario 1 (multiple broadcast)
//! could only be created by merging new edges in the new `group`, that is:
//!   case 1. `%input_t0`, shared by `producer` and `consumer`
//!   case 2. `%p_out_t0`, produced by `producer` and fed to `consumer`
//!
//! new cases of scenario 2 (output was broadcasted later) could only be added
//! via:
//!   case 3. `%p_out_t0`, produced by `producer` and fed to `consumer`, which
//!           could be broadcasted in the consumer subgraph.
//!
//! footnote*:
//! We are only disabling multiple broadcast right on the tensor, instead of
//! tracing all the broadcast further down.
//! I don't think we need to worry about broadcasting further down the
//! dependency chain, as those would create new IterDomain, which doesn't have
//! th problem of conflicting broadcasting.
bool createTrickyBroadcast(const Node* consumer, const Node* producer) {
  auto count_broadcasting_in_node =
      [](const Node* node,
         const std::vector<c10::optional<int64_t>>& shape,
         size_t offset) {
        int num_broadcasting = 0;
        if (node->kind() == prim::CudaFusionGroup) {
          // be careful here as `subgraph_input`, as its name suggests, is in a
          // different fraph from `node`.
          const auto& subgraph_input =
              node->g(attr::Subgraph)->inputs()[offset];
          for (const auto& use : subgraph_input->uses()) {
            if (maybeBroadcastOnShape(use.user, shape)) {
              num_broadcasting++;
            }
          }
        } else {
          if (maybeBroadcastOnShape(node, shape)) {
            num_broadcasting++;
          }
        }
        return num_broadcasting;
      };

  // case 1. We check shared inputs to `producer` & `consumer`;
  for (const auto i : c10::irange(producer->inputs().size())) {
    auto n_input = producer->input(i);
    auto n_input_type = n_input->type()->cast<TensorType>();
    if (n_input_type != nullptr && n_input_type->sizes().sizes()) {
      std::vector<c10::optional<int64_t>> n_input_shape =
          n_input_type->sizes().sizes().value();
      int num_broadcasting = 0;

      // check broadcasting for the n_input inside `consumer`;
      for (const auto& use : n_input->uses()) {
        if (use.user == consumer) {
          num_broadcasting +=
              count_broadcasting_in_node(consumer, n_input_shape, use.offset);
        }
      }

      // if no broadcasting happened for consumer, there's no point check
      // multiple broadcasting in producer alone;
      if (num_broadcasting == 0) {
        continue;
      }

      // check broadcasting for n_input inside `producer`;
      num_broadcasting +=
          count_broadcasting_in_node(producer, n_input_shape, i);

      // encounted multiple broadcasting scheme for a single TV, we will not be
      // able to schedule this, prevent the fusion; (case 1)
      if (num_broadcasting > 1) {
        return true;
      }
    }
  }

  // case 2. We check input to `consumer` that is also the output from
  // `producer`
  for (const auto i : c10::irange(producer->outputs().size())) {
    auto n_output = producer->output(i);
    auto n_output_type = n_output->type()->cast<TensorType>();
    if (n_output_type != nullptr && n_output_type->sizes().sizes()) {
      std::vector<c10::optional<int64_t>> n_output_shape =
          n_output_type->sizes().sizes().value();
      int num_broadcasting = 0;
      // If we only look at case 1 & case 2, we need to check broadcast of
      // `n_output` inside `producer`, if it is a `prim::CudaFusionGroup`.
      // this is actually not necessary when we consider case 3, as we avoid
      // broadcasting on outputs already;

      // TODO: merge this code with case 1.
      // check broadcasting for the n_output inside `consumer`;
      bool use_as_output = false;
      for (const auto& use : n_output->uses()) {
        if (use.user == consumer) {
          num_broadcasting +=
              count_broadcasting_in_node(consumer, n_output_shape, use.offset);
        } else {
          // case 3. output is used by other nodes not the consumer, no
          //         broadcasting is allowed;
          use_as_output = true;
        }
      }

      // encounted multiple broadcasting scheme for a single TV, we will not be
      // able to schedule this, prevent the fusion; (case 2)
      // Alternatively, if use_as_output is true, we would not permit broadcast
      // at all. (case 3)
      if (num_broadcasting > (use_as_output ? 0 : 1)) {
        return true;
      }
    }
  }

  return false;
}

} // namespace

bool isFusibleCudaFusionGroup(const Node* node) {
  FUSER_PERF_SCOPE("isFusibleCudaFusionGroup");

  if (isFusibleNode(node)) {
    auto ret = isFusibleDevice(node);
    return ret;
  }
  return false;
}

bool isFusibleCudaFusionGroup(const Node* fusion, const Node* node) {
  FUSER_PERF_SCOPE("isFusibleCudaFusionGroup");
  bool fused = false;
  // TODO: lift the restriction of not fusing producer containing reduction when
  //       we have proper scheduling.
  if (isFusibleCudaFusionGroup(node)) {
    // ensure if the node has a designated device, it's on the same device with
    // fusion.
    // TODO: is there a danger of us fusing operations that's supposed to be on
    //       separate GPUs? And is that necessarily bad?
    auto device = getDevice(fusion);
    fused = (!device.has_value() || isFusibleDevice(node, device.value()));
  }

  return fused;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
