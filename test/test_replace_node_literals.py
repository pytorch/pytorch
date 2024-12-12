import itertools 
from typing import List
import logging

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.ao.quantization.pt2e.utils import _replace_node_literals_with_existing_placeholders
from torch.fx.subgraph_rewriter import replace_pattern
from torch.testing._internal.common_utils import TestCase

log = logging.getLogger('REWRITE')
# logging.basicConfig(level = logging.INFO)

def to_svg(graph_module, filename: str):
    """
    Export GM as SVG.
    """
    g_dot = FxGraphDrawer(graph_module, "Model")
    g_dot.get_dot_graph().write_svg(filename)

def ln_example_inputs():
    """
    Layer Norm Example Inputs.
    """
    N, C, H, W = 20, 5, 10, 10
    return (
      torch.randn(N, C, H, W),  
      torch.Size([C, H, W]), 
      torch.randn(C, H, W), 
      torch.randn(C, H, W),  
      0.0001
    )

class LayerNormPattern(torch.nn.Module):

    def forward(self, x: torch.Tensor, normalized_shape: List[int], 
                weight: torch.Tensor, bias: torch.Tensor, 
                eps: float) -> torch.Tensor:

        # match LayerNorm
        y = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

        # ... optionally, match other functionality ...

        return y

# a mapping of literals (as specified using node/argument idx pairs) to a
# corresponding placeholder name for the entire Op.
# - for example, here we connect the argument index #1 (which corresponds
#   to a literal argument) of the aten::layer_norm op node to the
#   placeholder called "normalized_shape" in the containing graph module.
# 
# This corresponds to LayerNormPattern.
LN_PATTERN_LITERAL_MAP = {
    'aten::layer_norm': {           # the node for which this replacement applies
        1: 'normalized_shape',      # <node_arg_idx> -> <op_placeholder_name> mapping.
        4: 'eps',
    }
}

@torch.library.custom_op("my_lib::my_layer_norm", mutates_args=())
def my_layer_norm(x: torch.Tensor, normalized_shape: List[int],
                  weight: torch.Tensor, bias: torch.Tensor, 
                  eps: float) -> torch.Tensor:

    # for now, returns corresponding ATEN operator output, but could contain a
    # customized version of layer_norm.
    y = torch.ops.aten.layer_norm(x, normalized_shape, weight, bias, eps)

    # ... other optional functionality ...

    return y

@torch.library.register_fake("my_lib::my_layer_norm")
def _(x: torch.Tensor, normalized_shape: List[int], 
      weight: torch.Tensor, bias:torch.Tensor, 
      eps: float):

    return torch.empty_like(x)

class LayerNormReplacement(torch.nn.Module):

    def forward(self, x: torch.Tensor, normalized_shape: List[int], 
                weight: torch.Tensor, bias: torch.Tensor, 
                eps: float) -> torch.Tensor:

        # match LayerNorm
        y = torch.ops.my_lib.my_layer_norm(x, normalized_shape, weight, bias, eps)

        # ... optionally, match other functionality ...

        return y
#
# Define the mapping between a node's arguments and its corresponding placeholder nodes.
#
LN_REPL_LITERAL_MAP = {
    'my_lib::my_layer_norm': {     # the node (call_function) for which this replacement applies
        1: 'normalized_shape',     # <node_arg_idx> -> <op_placeholder_name> mapping.
        4: 'eps',
    }
}

class TestLayerNormRepl(TestCase):
    
    @classmethod
    def setUpClass(cls): 

        cls.example_inputs = ln_example_inputs()

        # create and compile patterns and replacements.
        pattern_model = LayerNormPattern()
        cls.pattern_gm = torch.export.export(pattern_model, cls.example_inputs).module()
        _replace_node_literals_with_existing_placeholders(cls.pattern_gm, LN_PATTERN_LITERAL_MAP)
        # to_svg(cls.pattern_gm, 'pattern_before.svg')
        
        repl_model = LayerNormReplacement()
        cls.repl_gm = torch.export.export(repl_model, cls.example_inputs).module()
        _replace_node_literals_with_existing_placeholders(cls.repl_gm, LN_REPL_LITERAL_MAP)
        # to_svg(cls.repl_gm, 'replacement.svg')

    def run_layernorm_repl(self, model_gm: torch.nn.Module):

        # replace literals with existing placeholders.
        # to_svg(self.pattern_gm, 'pattern_after.svg')

        # do the op replacement.
        matches = replace_pattern(model_gm, self.pattern_gm, self.repl_gm)
        # to_svg(model_gm, 'model_after_repl.svg')

        return matches, model_gm

    def test_layernorm_models(self):
        
        shape_arr = [ (5, 10, 10) ]
        eps_arr = [ 1e-5 ]
        elementwise_affine_arr = [ False, True ]
        bias_arr = [ False, True ]
        dtype_arr = [ torch.float32, torch.float64 ]

        for shape, eps, affine, bias, dtype in itertools.product(
                shape_arr, eps_arr, elementwise_affine_arr, bias_arr, dtype_arr):

            # example input.
            x_in = torch.randn(shape, dtype=dtype)

            log.info(f'Running {shape}, {eps}, {affine}, {bias}, {dtype} ...')
            model = torch.nn.LayerNorm(shape, eps, affine, bias, dtype=dtype)
            model_gm = torch.export.export(model, (x_in,)).module()

            # check if match succeeded.
            matches, model_repl = self.run_layernorm_repl(model_gm)
            self.assertEqual(len(matches), 1)

            # since our op does the same as the original layernorm, do a sanity
            # compare of the original model's output with our replacement).
            self.assertEqual(torch.all(model(x_in) == model_repl(x_in)), True)
