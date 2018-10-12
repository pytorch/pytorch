#include "torch/csrc/jit/fusers/common/fusion_handle_impl.h"

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/fusion_arg_spec.h"
#include "torch/csrc/jit/fusers/common/annotated_graph.h"
#include "torch/csrc/jit/fusers/common/tensor_desc.h"
#include "torch/csrc/jit/fusers/cpu/fused_kernel.h"
#include "torch/csrc/jit/fusers/cpu/fusion_compiler.h"
#include "torch/csrc/jit/fusers/cuda/fused_kernel.h"

#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/custom_operator.h"

#include "torch/csrc/utils/functional.h" //fmap

#include "ATen/ATen.h"
#include "ATen/ExpandUtils.h"

#include <unordered_set>
#include <tuple>
#include <algorithm>
#include <exception>

namespace torch { namespace jit {

////////////////////////////////////////////////////////////////////////////////
// FusedKernelCache

// Note [Run-time shape checking code]
// There are multiple assumptions that our codegen makes, which we can't check
// in the fusion pass, because we don't have the shape information. Most notably,
// that all values (post-input-chunk, and pre-output-concat) have the same shape
// (hereinafter referred to as map size). One way to check this would be to run
// shape propagation for every size configuration we get as an input, but that
// requires a full graph traversal, and might incur unnecessary overhead. The code
// below uses a few nice properties of broadcasting rules and their interactions with
// pointwise operations, and takes a smarter approach, to quickly verify validity of
// the kernel.
//
// Notation:
//   - a.s when a is a tensor is a shorthand for a.shape.
//   - B is a shorthand for the broadcasting/expanding function. It is used as a
//     vararg function.
//   - E is a shorthand for expand function.
//   - Every pointwise operation can be equivalently rewritten as
//     f(a, b) = f^(E(a, B(a.s, b.s)), E(b, B(a.s, b.s))),
//     where f^ is a non-broadcasting verison of f.
//   - A set of inputs that are used to produce a certain graph output is referred to
//     as the output's broadcasting group (see Lemma 2. for explanation why).
//
// Lemma 1. Set of lists of integers (shapes) + { _|_ (bottom/error marker) }, with the
//          operation of broadcasting (returning bottom upon shape mismatch) forms a monoid.
//          In simpler terms: broadcasting is associative, i.e. B(a, B(b, c)) == B(B(a, b), c).
//
// Proof.   Satisfies all monoid laws:
//            - Closed under broadcasting (trivial)
//            - Empty shape is the identity element: B(a, []) == B([], a) == a
//            - Associativity: A simple visual proof is that you can expand 3 tensors
//                at the same time by stacking their sizes (with alignment to the right),
//                just as you'd do in the case of 2 tensors, but with an intermediate
//                (the algorithm ends up being pretty much the same).
//
// Lemma 2. Shape of an output of an arbitrary DAG of pointwise ops depends only on the set
//          of inputs used in this DAG and is equal to B([i.shape for i in used_inputs]).
//
// Proof.   Let G be any DAG of pointwise ops and < be any valid topological
//          ordering on nodes of G. Proof by induction over <.
//          Base case (graph input):
//            Trivial (input is also an output).
//          Step (n = f(q, r)):
//            Let QS (RS) be the set of shapes of inputs that q (r) depends on.
//            Note that the set of inputs that n depends on is exactly QS + RS.
//            shape(n) == shape(f(q, r))
//                          (def of f)
//                     == shape(f^(E(q, B(q.s, r.s)), E(r, B(q.s, r.s))))
//                          (output shape of f^ is equal to either of argument shapes)
//                     == shape(E(q, B(q.s, r.s)))
//                          (property of expand)
//                     == B(q.s, r.s)
//                          (induction assumption)
//                     == B(B(QS...), B(RS...))
//                          (Lemma 1.)
//                     == B(QS..., RS...)
//                          (repeated shapes don't matter for broadcasting)
//                     == B((QS + RS)...)
//
// Lemma 3. Expands are distributive over pointwise ops, i.e. E(f(a, b), s) = f(E(a, s), E(b, s))
// Lemma 4. Expands can be collapsed, i.e. E(E(x, s1), s2) = E(x, B(s1, s2)).
// Proof.   A simple exercise for the reader :)
//
// Theorem. If all (pre-concat-)outputs have equal shapes, then we can push the expands to
//          (post-chunk-)inputs, and have all intermediates of the same shape
//          (no broadcasting happening in the body).
//
// Proof.   Using the above lemmas we can easily show that a graph with a single output
//          can be easily rewritten by taking the shape given by B applied to all input
//          shapes, expanding inputs to it, and using only non-broadcasting operations.
//          Example:
//
//          let d = f(a, b) in
//          let e = h(b, c) in
//          g(d, e)
//
//          (By def. of broadcasting pointwise ops applied to g, f and h)
//          (Lemma 2. for a closed formula for the size of g = gs)
//
//          let gs = B(a.s, b.s, c.s) in
//          let d' = E(f^(E(a, B(a.s, b.s)), E(b, B(a.s, b.s))), gs) in
//          let e' = E(h^(E(b, B(b.s, c.s)), E(c, B(b.s, c.s))), gs) in
//          g^(d', e')
//
//          (Lemma 3.)
//
//          let gs = B(a.s, b.s, c.s) in
//          let d' = f^(E(E(a, B(a.s, b.s)), gs), E(E(b, B(a.s, b.s)), gs)) in
//          let e' = h^(E(E(b, B(b.s, c.s)), gs), E(E(c, B(b.s, c.s)), gs)) in
//          g^(d', e')
//
//          (Lemma 4. + Lemma 1. to simplify broadcasting function)
//
//          let gs = B(a.s, b.s, c.s) in
//          let d' = f^(E(a, gs), E(b, gs)) in
//          let e' = h^(E(b, gs), E(c, gs)) in
//          g^(d', e')
//
//          (Simple rewrite)
//
//          let gs = B(a.s, b.s, c.s) in
//          let a' = E(a, gs) in
//          let b' = E(b, gs) in
//          let c' = E(c, gs) in
//          let d' = f^(a', b') in
//          let e' = h^(b', c') in
//          g^(d', e')
//
//          This example can be easily formalized to arbitrary DAGs using induction
//          over topological ordering, similar to Lemma 2. Now, if broadcasting groups
//          for all outputs have the same shape, then performing an expand to this size
//          on all inputs will ensure that all intermediates on all paths to outputs
//          will have the same shape, proving that the body of the kernel is valid.
//
//          This shows the part until post-chunk-inputs. Extending it to pre-chunk-inputs
//          is straightforward (needs a simple lemma for moving expands through chunks).

// Register implementations of fused operators, so that we can reuse the fused graph
// to generate fallback code.
RegisterOperators reg_fused_operators({
  Operator(
    prim::FusedConcat,
    [](Node* node) {
      int64_t dim = node->i(attr::dim);
      int64_t num_inputs = node->inputs().size();
      return [dim, num_inputs](Stack& stack) {
        auto result = at::cat(
          fmap(last(stack, num_inputs), [](const IValue& i) { return i.toTensor(); }),
          dim
        );
        drop(stack, num_inputs);
        pack(stack, std::move(result));
        return 0;
      };
    })
});

FusionHandleImpl::FusionHandleImpl(
  std::shared_ptr<Graph> _graph
, int device)
: device(device)
, fallback_code(_graph)
, graph(std::move(_graph))
, input_broadcast_groups(getInputBroadcastGroups())
, input_chunks(getInputChunkDescriptors())
, kernels() { }

std::atomic<size_t> FusionHandleImpl::next_kernel_id {0};

static Node* usedInFusedChunk(Value* input) {
  auto uses = input->uses();
  if (uses.size() == 1) {
    Node *user = uses[0].user;
    if (user->kind() == prim::ConstantChunk) {
      return user;
    }
  }
  return nullptr;
}

auto FusionHandleImpl::getInputChunkDescriptors() -> std::vector<PartitionInfo> {
  std::vector<PartitionInfo> descs;
  descs.reserve(graph->inputs().size());
  for (Value* input : graph->inputs()) {
    if (Node* chunk = usedInFusedChunk(input)) {
      descs.emplace_back(chunk->i(attr::chunks), chunk->i(attr::dim));
    } else {
      descs.emplace_back(1, 0);
    }
  }
  return descs;
}

// NB: this vector is really a set, but we want to keep it contiguous in memory for faster access
static std::vector<int64_t> getInputDependencies(Value* output) {
  // Run a DFS traversal to find all inputs that affect a given output value
  std::vector<Value*> queue { output };
  std::unordered_set<Value*> inputs;
  std::unordered_set<Value*> seen;
  while (!queue.empty()) {
    Value* val = queue.back(); queue.pop_back();
    Node* producer = val->node();
    if (producer->kind() == prim::Param) {
      inputs.insert(val);
      continue;
    }
    for (Value* input : producer->inputs()) {
      if (/*bool inserted = */seen.insert(input).second) {
        queue.push_back(input);
      }
    }
  }

  // Convert Value* into offsets into the graph's input list
  std::vector<int64_t> offsets;
  offsets.reserve(inputs.size());
  for (Value* input : inputs) {
    offsets.push_back(input->offset());
  }

  std::sort(offsets.begin(), offsets.end());
  return offsets;
}

// See Note [Run-time shape checking code] for more explanation on the algorithm.
at::optional<std::vector<int64_t>> FusionHandleImpl::canRunKernel(at::TensorList args) {
  AT_CHECK(args.size() == input_chunks.size(),
           "Expected ", input_chunks.size(), " arguments, but got ", args.size());

  at::optional<std::vector<int64_t>> map_size;
  for (const auto & broadcast_group : input_broadcast_groups) {
    if (!map_size) {
      map_size = getMapSize(args, broadcast_group);
      if (!map_size) {
        return at::nullopt;
      }
    } else {
      auto group_map_size = getMapSize(args, broadcast_group);
      // NB: this checks that group_map_size is defined AND equal to map_size
      if (map_size != group_map_size) {
        return at::nullopt;
      }
    }
  }
  return map_size;
}

std::unique_ptr<FusedKernel> FusionHandleImpl::compileSpec(
  const FusionArgSpec& spec
, const std::vector<int64_t>& map_size) {
  AnnotatedGraph agraph{*graph, device};

  agraph.input_desc = spec.descs();
  // XXX: this assumes that fused kernels only operate on floating-point values inside
  at::optional<at::ScalarType> scalar_type;
  for (TensorDesc& desc : agraph.input_desc) {
    if (isFloatingType(desc.scalar_type)) {
      scalar_type = desc.scalar_type;
      break;
    }
  }
  JIT_ASSERT(scalar_type);

  for (Value * output : graph->outputs()) {
    std::vector<int64_t> sizes = map_size;
    if (output->node()->kind() == prim::FusedConcat) {
      sizes.at(output->node()->i(attr::dim)) *= output->node()->inputs().size();
    }
    auto type = CompleteTensorType::create(*scalar_type, device, sizes);
    agraph.output_desc.emplace_back(std::move(type));
  }

  std::string name = "kernel_" + std::to_string(next_kernel_id++);
  FusedKernel* raw_func;
  if (device != kCPUDevice) {
    #if USE_CUDA_FUSER
      raw_func = new cudafuser::CUDAFusedKernel(name, agraph);
    #else
      throw std::runtime_error("CUDA Fusion is not supported on this build.");
    #endif // USE_CUDA_FUSER
  } else {
    raw_func = new cpufuser::CPUFusedKernel(
      name
    , agraph
    , cpufuser::getFusionCompiler().getConfig());
  }
  return std::unique_ptr<FusedKernel>(raw_func);
}

// NB: args are mutated in this call. map_size is mutated too, but is restored to its original
// value before this function returns (it's an optimization).
void FusionHandleImpl::expandArgs(std::vector<at::Tensor>& args, std::vector<int64_t>& map_size) {
  for (size_t i = 0; i < args.size(); ++i) {
    auto& arg = args[i];
    auto& pdesc = input_chunks[i];
    if (pdesc.nSubtensors == 1) {
      if (arg.sizes().equals(map_size)) continue;
      arg = arg.expand(map_size);
    } else {
      map_size.at(pdesc.dim) *= pdesc.nSubtensors;
      if (!arg.sizes().equals(map_size)) {
        arg = arg.expand(map_size);
      }
      map_size.at(pdesc.dim) /= pdesc.nSubtensors;
    }
  }
}

std::vector<std::vector<int64_t>> FusionHandleImpl::getInputBroadcastGroups() {
  std::unordered_set<std::vector<int64_t>, torch::hash<std::vector<int64_t>>> broadcast_groups;
  for (Value* output : graph->outputs()) {
    broadcast_groups.insert(getInputDependencies(output));
  }
  return std::vector<std::vector<int64_t>>{broadcast_groups.begin(), broadcast_groups.end()};
}

void FusionHandleImpl::run(Stack& stack) {
  int64_t num_inputs = graph->inputs().size();
  auto args = fmap(last(stack, num_inputs), [](const IValue& i) {
    return i.toTensor();
  });

  auto maybe_map_size = canRunKernel(args);
  if (!maybe_map_size) {
    return runFallback(stack);
  }
  expandArgs(args, *maybe_map_size);

  FusionArgSpec spec{args};
  auto it = kernels.find(spec);
  if (it == kernels.end()) {
    std::tie(it, std::ignore) = kernels.emplace(spec, compileSpec(spec, *maybe_map_size));
  }
  auto& fn = it->second;

  std::vector<at::Tensor> outputs;
  fn->launch(args, outputs);
  drop(stack, num_inputs);
  stack.insert(
    stack.end()
  , std::make_move_iterator(outputs.begin())
  , std::make_move_iterator(outputs.end()));
}

at::optional<std::vector<int64_t>> FusionHandleImpl::getMapSize(
  at::TensorList args
, at::IntList arg_subset) {
  int64_t dim_after_broadcast = 0;
  for (int64_t arg_idx : arg_subset) {
    dim_after_broadcast = std::max(dim_after_broadcast, args[arg_idx].dim());
  }
  // TODO: this keeps reallocating map_size at every iteration, but we know
  // exactly how much storage do we need, so this could be fixed in-place at
  // every step. We're just missing a few functions for ATen, but the fix
  // should be straightforward.
  // NB: we leave this uninitialized, because an empty size is trivially
  // broadcastable to any other size.
  std::vector<int64_t> map_size;
  for (size_t i = 0; i < arg_subset.size(); ++i) {
    auto& arg = args.at(arg_subset[i]);
    auto& chunk_desc = input_chunks.at(arg_subset[i]);
    if (chunk_desc.nSubtensors == 1) {
      try {
        map_size = at::infer_size(map_size, arg.sizes());
      } catch (std::exception& e) {
        return at::nullopt;
      }
    } else {
      auto tensor_sizes = arg.sizes().vec();
      int64_t num_chunks = chunk_desc.nSubtensors;
      int64_t dim = at::maybe_wrap_dim(chunk_desc.dim, tensor_sizes.size());
      if (tensor_sizes[dim] % num_chunks != 0) {
        return at::nullopt;
      }
      tensor_sizes[dim] /= num_chunks;
      try {
        map_size = at::infer_size(map_size, tensor_sizes);
      } catch (std::exception& e) {
        return at::nullopt;
      }
    }
  }

  return {map_size};
}

void FusionHandleImpl::runFallback(Stack& stack) {
  InterpreterState(fallback_code).run(stack);
}

} // namespace jit
} // namespace torch
