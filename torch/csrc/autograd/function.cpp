#include "torch/csrc/autograd/function.h"

#include "torch/csrc/autograd/functions/special.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <deque>

namespace torch { namespace autograd {

thread_local uint64_t Function::next_sequence_nr_ = 0;

auto Function::name() -> std::string {
  return std::string(typeid(*this).name());
}

// This function is analogous to make_trace which operates on PythonOp, but this
// function instead works for C++ implemented autograd Functions, which don't
// actually have any backing Python class. We still need to trace them!
variable_list Function::traced_apply(variable_list inputs) {
  using namespace torch::jit;
  // Traceable Functions are completely transparent to the JIT.
  if (is_traceable()) {
    return apply(inputs);
  }
  auto state = tracer::getTracingState(inputs);
  auto state_lock = state->lock();

  // Insert a CppOp in the trace.
  auto& graph = state->graph;
  std::vector<VariableFlags> var_flags;
  for(auto & input: inputs) {
    var_flags.push_back(VariableFlags::of(input));
  }
  auto* this_node = graph->createCppOp(get_shared_ptr(), std::move(var_flags));
#ifndef NO_PYTHON
  this_node->setSourceLocation(std::make_shared<StringSourceLocation>(
        jit::tracer::getPythonInterpreterStackTrace()
  ));
#endif
  for (auto& input: inputs) {
    this_node->addInput(tracer::getValueTrace(state, input));
  }
  graph->appendNode(this_node);

  // Finally apply this Function.
  state_lock.unlock();
  variable_list outputs = apply(inputs);
  state_lock.lock();

  // Set up output traces.
  int num_outputs = outputs.size();
  for (int i = 0; i < num_outputs; ++i) {
    auto& output = outputs[i];
    auto sel = this_node->addOutput();
    // TODO: At the moment, C++ does not track shared storage.  It
    // should.  Update this when that happens.
    if (output.defined()) {
      sel->inferTypeFrom(output.data());
      tracer::setValueTrace(state, output, sel);
    }
  }

  if (!passes_state_transparently()) {
    auto this_eval = dynamic_cast<Eval*>(this);
    // Evals consume handle from a context edge of forward node
    if (this_eval)
      this_node->addInput(this_eval->forward_ctx_select);
    // There's no point in wrapping functions in Eval, if we know they already are
    // part of another Eval subgraph. This is both a small optimization, and
    // it allows us to not implement saved_variables() in many functions.
    const bool should_trace_backward = tracing_state_->in_eval_subgraph;
    if (!should_trace_backward) {
      auto saved_vars = saved_variables();
      if (!saved_vars)
        throw std::runtime_error("saved_variables() needed but not implemented in " + name());
      variable_list bw_subgraph_inputs(inputs);
      for (auto& saved_var : *saved_vars) {
        bw_subgraph_inputs.emplace_back(saved_var.unpack(get_shared_ptr()));
      }
      tracer::nontraceableBackwardSubgraph(bw_subgraph_inputs, outputs);
    }
    bool has_backwards_eval = !should_trace_backward || this_eval;
    if (has_backwards_eval)
      set_up_context_edge(this_node, inputs, outputs);
  }
  return outputs;
}

void Function::set_up_context_edge(
    jit::Node* this_node,
    const variable_list& inputs,
    const variable_list& outputs) {
  auto ctx_select = this_node->addOutput();
  ctx_select->setType(jit::HandleType::get());
  auto backward_eval = Eval::getBackwardEval(inputs, outputs);
  if (backward_eval)
    backward_eval->forward_ctx_select = ctx_select;
}

/*
 * Fix for #5534: prevent stack overflow on deletion of deep computation graph
 *
 * Sometimes one can end up with a very big computation graph of Functions
 * and Edges. Each std::shared_ptr<Function> contains a list of Edge, and
 * each Edge contains a std::shared_ptr<Function>. Deleting a
 * std::shared_ptr<Function> can trigger the recursive deletion of other
 * std::shared_ptr<Function>'s: this can stack overflow if the graph
 * is deep enough. Here is an example of such a graph:
 *
 * shared_ptr<Function> -> Edge -> shared_ptr<Function> -> Edge -> ... -> shared_ptr<Function>
 *
 * The solution here is to use a custom deleter with each
 * std::shared_ptr<Function>. The custom deleter keeps track of how many
 * nested deleters it is in. When this number exceeds the maximum allowed
 * depth, the Function* to be deleted are accumulated in a per-thread
 * delete queue and handled by one of the deleters.
 *
 * Note that these custom deleters are NOT necessary for deleting PyFunction.
 * This is because a THPFunction Python object owns a PyFunction that is in a
 * computation graph. When Python objects get recursively destroyed, they
 * are also queued into a delete list. This happens very early for them
 * (at 50 deleters): https://github.com/python/cpython/blob/f320be77ffb73e3b9e7fc98c37b8df3975d84b40/Include/object.h#L1024-L1063
 * so we don't need to worry about them.
 */

thread_local std::deque<Function*> deleteFunctionQueue;
thread_local size_t deleteFunctionRecursionDepth = 0;

/*
 * If this number is set too high, a deep computation graph can still
 * stack overflow. The procedure for setting this number was to
 * 1) find the smallest value that would not guard against stack overflows
 *    on various machines
 * 2) Take the minimum of all such values and subtract some leeway because
 *    the memory of these stack frames will probably grow as time passes.
 * Testing on a few machines machines, the magic numbers were:
 * - Mac OSX (Macbook Pro 15) : ~60000
 * - A beefy Ubuntu 16.04 box : ~15000
 * - Windows AWS instance (g3.4xlarge): variable. My two attempts at different
 *   times have gotten the following numbers: ~8300, 3669
 */
#ifdef _WIN32
constexpr size_t kDeleteFunctionMaxRecursionDepth = 3000;
#else
constexpr size_t kDeleteFunctionMaxRecursionDepth = 10000;
#endif

struct RecursionDepthCounter {
 public:
  explicit RecursionDepthCounter() {
    ++deleteFunctionRecursionDepth;
  }
  ~RecursionDepthCounter() {
    --deleteFunctionRecursionDepth;
  }

  size_t value() {
    return deleteFunctionRecursionDepth;
  }
};

/*
 * Note that the custom deleter deletes in BFS style. Without using
 * the custom deleter, the computation graph is deleted in a DFS style.
 * The BFS deletion is valid (and safe) because if a shared_ptr<Function>
 * 's reference count hits 0, nothing else will access it.
 */
void deleteFunction(Function* function) {
  RecursionDepthCounter recursion_depth;

  if (recursion_depth.value() > kDeleteFunctionMaxRecursionDepth) {
    deleteFunctionQueue.push_back(function);
    return;
  }

  delete function;

  if (deleteFunctionQueue.size() == 0) {
    return;
  }
  if (recursion_depth.value() != kDeleteFunctionMaxRecursionDepth) {
    AT_ERROR("Only one deleter per thread should be able to process "
             "the delete queue. Please open an issue.");
  }
  while (deleteFunctionQueue.size() > 0) {
    auto queued_function = deleteFunctionQueue.front();
    deleteFunctionQueue.pop_front();
    delete queued_function;
  }
}

}} // namespace torch::autograd
