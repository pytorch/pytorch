from typing import List, Tuple, Any

import torch
from torch.jit._recursive import wrap_cpp_module

class OrderLogger(torch.nn.Module):
    op_idxs: List[int]

    def __init__(self):
        super().__init__()
        self.op_idxs = []

    def forward(self, op_idx: int, data: torch.Tensor):
        self.op_idxs.append(op_idx)
        return data


def trace_and_record_order(
    scripted_module: torch.nn.Module,
    example_inputs: Tuple[Any, ...],
) -> torch.nn.Module:
    """
    This function takes a scripted model and does the following:
    1. adds an OrderLogger to each module in the model
    2. for each forward, routes the output of each quantizeable op in the
       graph through the OrderLogger instance with an op_idx
    3. traces the annotated model with `example_inputs` to create
       a mapping of op_idx and execution order

    For example, given a model with graph

      def forward(self, x):
          x = x + x
          if self.flag:
              x = x + x
          else:
              x = x + x
          return x

    The output of step 2 will be

      def forward(self, x):
          x = x + x
          x = self._logger_0(0, x)
          if self.flag:
              x = x + x
              x = self._logger_0(1, x)
          else:
              x = x + x
              x = self._logger_0(2, x)
          return x

    And the contents of `self._logger_0.op_idxs` after step 3 will be

      [0, 2]

    if `self.flag` is False.
    """
    module_c = scripted_module._c
    logger = torch.jit.script(OrderLogger())
    logger_c = logger._c
    module_c = torch._C._jit_pass_dbr_quant_annotate_with_order_logger(
        module_c, logger_c)
    scripted_module = wrap_cpp_module(module_c)
    scripted_module(*example_inputs)
    return scripted_module
