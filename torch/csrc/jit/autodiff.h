#pragma once

#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch { namespace jit {

using value_list = std::vector<Value*>;
// Example showcasing how Gradient is constructed:
//
// Let's assume we have a function f, `m` and `n` do not require grad
// (`n` can depend only on `m`):
//   y, n = f(x, m)
//
// Now, let's assume that the reverse of f (called f') needs to use values of `x`, `t` and `y`.
// `t` is an intermediate value produced in the body of f, and let's assume that it requires
// grad too.
//
// In this case differentiate(f) will return this:
//   y, n, t = f(x, m)        // `t` is appended to the output list
//   dx = f'(dy, dt, x, t, y) // No `dm` or `dn` because they do not require gradient
//                            // All needed values from f are prepended to the input list
//
//   f_real_outputs = 2       // Only first two outputs were present in f originally
//   df_input_vjps = {0, 2}   // i.e. connect grad_fn of y and t variables produced by f,
//                    y  t    // with y's output_nr = 0 and t's output_nr = 1
//   df_input_captures = {I0, O2, O0} // Order matches the prefix of inputs to df
//                        x   t   y
//   df_output_vjps = {0}     // i.e. connect next_edge[0] of grad_fn to x's (grad_fn, output_nr).
struct Gradient {
  operator bool() const {
    return df != nullptr;
  }
  std::shared_ptr<Graph> f;
  std::shared_ptr<Graph> df;

  // Describes how to construct outputs of f from what its graph will return.
  // This is necessary because some trailing outputs are intermediates produced
  // only to be saved for df (and should be ignored).
  std::size_t f_real_outputs;

  // df inputs are split into two sections: vjps (aka grad_outputs) and captures.
  // VJPs are "seeds" for the gradient computation given for each input capture
  // of an Output kind.
  // Captures are values the need to be saved when f is run. We handle inputs
  // specially, because this allows us to avoid adding extra vjps as df inputs.

  std::vector<std::size_t> df_input_vjps; // Offsets into f's outputs.
  // capture can come from inputs or outputs
  std::vector<std::size_t> df_input_captured_inputs; // Offsets into f's inputs
  std::vector<std::size_t> df_input_captured_outputs; // Offsets into f's outputs


  // df will produce vjps for a subset of inputs of f that required grad.
  // df_output_vjps[idx] == inp_idx means that idx-th output of df produces a vjp
  // for inp_idx-th input of f.
  std::vector<std::size_t> df_output_vjps; // Offsets into f's inputs.

  // How to use gradient to implement a differentiable autograd function:
  // When running f:
  //   - Unwrap input Variables
  //   - Run f's graph
  //   - Create grad_fn
  //   - Wrap outputs in Variables (assume we have a tensor_outputs array):
  //       outputs = map(Variable, tensor_output)
  //       for i, offset in enumerate(df_input_vjps):
  //         outputs[offset].set_grad_fn(grad_fn, output_nr=i)
  //   - Use df_output_vjps to connect next_edges of grad_fn:
  //       for idx in df_output_vjps:
  //         grad_fn.add_next_edge(inputs[idx].gradient_edge())
  //   - Save captures for df (care needs to be taken to use SavedVariables for inputs and
  //                           outputs that we will actually return)
  //   - Return outputs[:f_real_outputs]
  //
  // When running df:
  //   - Concatenate received vjps and captured Variables
  //   - Interpret df
  //   - Wrap outputs of df into Variables (that don't require grad)
};
// XXX: When calling this function, graph should have complete type information.
// Use the shape analysis pass to fill in the gaps if it doesn't.
Gradient differentiate(std::shared_ptr<Graph>& graph, const std::vector<bool>& requires_grad);

// can we take a derivative of this node symbolically?
bool isDifferentiable(Node * n);
bool isDifferentiable(Graph & g);
bool isZero(Value * v);

}}
