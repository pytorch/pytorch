#include <torch/csrc/jit/codegen/cuda/interface.h>

#include <ATen/core/dispatch/OperatorOptions.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<bool> cuda_fusion_guard_mode{true};

std::atomic<bool>& getCudaFusionGuardMode() {
  return cuda_fusion_guard_mode;
}

CudaFuserInterface* getFuserInterface() {
  static CudaFuserInterface fuser_interface_;
  return &fuser_interface_;
}

void compileFusionGroup(Node* fusion_node) {
  TORCH_CHECK(
      getFuserInterface()->fn_compile_n_ != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_compile_n_(fusion_node);
}

void runFusionGroup(const Node* fusion_node, Stack& stack) {
  TORCH_CHECK(
      getFuserInterface()->fn_run_n_s_ != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_run_n_s_(fusion_node, stack);
}

void fuseGraph(std::shared_ptr<Graph>& graph) {
  TORCH_CHECK(
      getFuserInterface()->fn_fuse_graph_ != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_fuse_graph_(graph);
}

bool canFuseNode(const Node* node) {
  return getFuserInterface()->fn_can_fuse_n_ != nullptr &&
      getFuserInterface()->fn_can_fuse_n_(node);
}

//! [ Note -- type guard logic in CudaFusionGuard ]
//!
//! CudaFusionGuard is used to Guard input tensor to `CudaFusionGroup` so that
//! we would not feed inputs that violates the graph defined in `GraphCache`.
//!
//! see [ Note -- 2 level cache implementation ] for definition of unique
//! computational graph.
//! see [ Note -- CudaFusionGuard implementation] for details on how guard works
//! in profiling executor
//!
//! Type guard logic is used to query whether a runtime input `tensor` compiles
//! with profiled `guard_tensor_type`. `guard_tensor_type` is the observed
//! tensor type during profiling runs.
//!
//! At this moment, we only do single profiling run, so `guard_tensor_type` has
//! static shape / stride / scalarType. *This might be a little confusing as our
//! implementation is actually more relaxed.
//!
//! Things that we check:
//!   a. identical rank & scalar type
//!   b. stride check:
//!        b.1. identical stride order
//!        b.2. identical contiguity
//!             note that contiguity here is used for tensor collapsing. So
//!             extra attention should be paid to contiguity across size-1
//!             dimensions.
//!   c. size check:
//!        making sure that broadcast semantics are identical. So we want to
//!        make sure a given dimension either are both size-1 for `tensor` &
//!        `guard_tensor_type`, or are both non-size-1.
//!        This is due to the fact that we specialize size-1 dimension as
//!        broadcasted dimension while translating PyTorch tensor to Fusion IR.
//!
bool complyWith(
    const at::Tensor& tensor,
    const c10::TensorTypePtr& guard_tensor_type) {
  // guard broadcast semantics, contiguity & stride order;
  TORCH_INTERNAL_ASSERT(
      guard_tensor_type && guard_tensor_type->dim().has_value());

  // check a. if num_dimension check fails or scalar type check fails
  if (*guard_tensor_type->dim() != static_cast<size_t>(tensor.ndimension()) ||
      (guard_tensor_type->scalarType().has_value() &&
       (guard_tensor_type->scalarType().value() != tensor.scalar_type())) ||
      tensor.requires_grad()) {
    return false;
  }

  // TODO: should we get symbolic_size instead and check for size
  // consistency across tensors as well?
  const auto& sizes = guard_tensor_type->sizes();
  const auto& stride_properties = guard_tensor_type->stride_properties();

  const auto& t_sizes = tensor.sizes();
  const auto& t_strides = tensor.strides();
  int inner_dim = -1;
  for (size_t j = 0; j < *guard_tensor_type->dim(); j++) {
    // check b. for stride check, we go along dimensions from fastest stride to
    // slowest stride
    int sorted_index = stride_properties[j]->stride_index_
        ? static_cast<int>(*stride_properties[j]->stride_index_)
        : -1;

    // only apply stride check when we have stride_properties
    if (sorted_index != -1) {
      // check b.1. stride order [current dimension has stride larger
      // than its inner dimension(s)], check only applies when both:
      //     i. already encountered an inner dimension
      //    ii. not at the fastest dimension
      if (j != 0 && inner_dim != -1) {
        // we are not looking at dim-j, but dim-sorted_index, which
        // is the j-th fastest dim;
        // TODO: merge this with above and put a long comment there
        if (t_strides[sorted_index] < t_strides[inner_dim]) {
          return false;
        }
      }

      // check b.2. contiguity, we only check when it's marked as
      // contiguous.
      if (stride_properties[j]->contiguous_ &&
          *stride_properties[j]->contiguous_) {
        if (j != 0) {
          // we use contiguity to collapse dimension, if size == 1, it is
          // always collapsible
          if (t_sizes[sorted_index] != 1) {
            TORCH_INTERNAL_ASSERT(
                stride_properties[j - 1]->stride_index_.has_value(),
                "Counknown index is meaningless");
            // TODO: merge this check up
            if (t_strides[sorted_index] !=
                t_strides[inner_dim] * t_sizes[inner_dim]) {
              return false;
            }
          }
        } else {
          // TODO: merge this check up
          if (t_strides[sorted_index] != 1) {
            return false;
          }
        }
      }

      // update inner_dim to be current dim. Note that we try to skip update
      // when current `t_size[sorted_index] == 1`, because:
      //   1. stride comparison on a size-1 dimension is meaningless
      //      [check b.1]
      //   2. contiguity on a size-1 dimension is misleading. For collapsing,
      //      we should actually look at the next non-size-1 dimension
      //      [check b.2]
      if (inner_dim == -1 || t_sizes[sorted_index] != 1) {
        inner_dim = sorted_index;
      }
    }

    // check c, we go along semantic ordered dimensions
    // check broadcast / size-1:
    bool guard_bcast = sizes[j].has_value() && sizes[j].value() == 1;
    if (guard_bcast != (t_sizes[j] == 1)) {
      return false;
    }
  }

  return true;
}

} // namespace cuda
} // namespace fuser

namespace {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterOperators reg_fusion({
    Operator(
        prim::CudaFusionGroup,
        [](const Node* node) -> Operation {
          return [node](Stack* stack) {
            fuser::cuda::runFusionGroup(node, *stack);
          };
        },
        aliasAnalysisSpecialCase()),
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
RegisterOperators reg_guard({
    Operator(
        "prim::CudaFusionGuard(...) -> bool",
        // prim::CudaFusionGuard returns a fresh Boolean type without aliasing.
        // if we would ever return refined tensor, which would change aliasing
        // analysis, we should update aliasdb pass.
        [](const Node* node) -> Operation {
          return [node](Stack* stack) {
            // TODO: check latency here!!!!
            std::vector<TypePtr> types = node->tys(attr::types);
            const auto num_inputs = types.size();
            at::ArrayRef<IValue> inputs = last(stack, num_inputs);
            drop(stack, num_inputs);

            if (!fuser::cuda::getCudaFusionGuardMode()) {
              push(stack, IValue(true));
              return;
            }

            for (size_t i = 0; i < num_inputs; i++) {
              const c10::TensorTypePtr& guard_tensor_type =
                  types[i]->cast<TensorType>();

              // TODO: maybe we should just push false and fallback
              TORCH_INTERNAL_ASSERT(inputs[i].isTensor());
              const at::Tensor& tensor = inputs[i].toTensor();

              if (!fuser::cuda::complyWith(tensor, guard_tensor_type)) {
                push(stack, IValue(false));
                return;
              }
            }

            // TODO: check type and return the right flag
            // naively return true;
            push(stack, IValue(true));
            return;
          };
        },
        aliasAnalysisFromSchema()),
});
} // namespace

} // namespace jit
} // namespace torch
