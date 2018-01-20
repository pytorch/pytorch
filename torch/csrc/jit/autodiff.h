#pragma once

#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch { namespace jit {

// This function mutates the given graph (which should only contain a single stage)
// by appending nodes of a next stage, which computes the Jacobian-vector product
// (aka backward) of inputs to the first stage w.r.t. the outputs of the first stage.
// TODO: expand the comment
void differentiate(std::shared_ptr<Graph>& graph,
                   ArrayRef<bool> requires_grad_inputs,
                   ArrayRef<bool> requires_grad_outputs);

using value_list = std::vector<Value*>;
struct LiftedReverse {
  std::shared_ptr<Graph> f;
  std::shared_ptr<Graph> df;

  // See liftLambdaReverse for a detailed overview of the structure of
  // f's outputs and df's inputs. The gist of it is that:
  // - f_output_intermediates describes how many temporaries were appended to outputs of f
  //   (because they are needed by df; they should be discarded before returning f's outputs)
  // - df_input_captures describes which outputs of f to capture and later provide
  //   as first df_input_captures.size() inputs to df (with trailing inputs being
  //   vjps aka grad outputs).
  std::size_t f_output_intermediates;
  std::vector<std::size_t> df_input_captures;
};
LiftedReverse lambdaLiftReverse(std::shared_ptr<Graph>& graph);

}}
