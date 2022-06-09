import typing as t

import torch
import torch.fx
from torch.fx.passes.operator_support import OperatorSupport
# from torch.fx.passes.tools_common import CALLABLE_NODE_OPS, get_node_target
from torch._C._nvfuser import FusionDefinition as fd


class NvFuserOperatorSupport(OperatorSupport):
    """
    Operator support for nvFuser backend.

    Note: When adding a rule, please add it to the corresponding secion and follow the
    alphabetical order.
    """

    def __init__(self):

        support_dict = {
        #     # ===============================================================
        #     # call_function torch.nn.functional
        #     # ===============================================================
        #     "torch.nn.functional.avg_pool1d": None,
        #     "torch.nn.functional.avg_pool2d": None,
        #     "torch.nn.functional.avg_pool3d": None,
        #     "torch.nn.functional.batch_norm": None,
        #     "torch.nn.functional.dropout": None,
        #     "torch.nn.functional.embedding": (
        #         (),
        #         {
        #             "input": [torch.int64],
        #             "weight": [torch.float32],
        #         },
        #     ),
        #     "torch.nn.functional.gelu": (
        #         (),
        #         {
        #             "input": [torch.float32],
        #         },
        #     ),
        #     "torch.nn.functional.layer_norm": (
        #         (),
        #         {
        #             "input": [torch.float32],
        #             "weight": [torch.float32],
        #             "bias": [torch.float32],
        #         },
        #     ),
        #     "torch.nn.functional.log_softmax": None,
        #     "torch.nn.functional.max_pool2d": None,
        #     "torch.nn.functional.prelu": None,
        #     "torch.nn.functional.relu": None,
        #     "torch.nn.functional.sigmoid": None,
        #     "torch.nn.functional.silu": None,
        #     "torch.nn.functional.softmax": (
        #         (),
        #         {
        #             "input": [torch.float32],
        #         },
        #     ),
        #     "torch.nn.functional.tanh": None,
        #     # ===============================================================
        #     # call_function torch
        #     # ===============================================================
        #     "torch.addmm": None,
        #     "torch.arange": None,
        #     "torch.argmax": None,
        #     "torch.argmin": None,
        #     "torch.bmm": ((), {"input": [torch.float32], "mat2": [torch.float32]}),
        #     "torch.clamp": None,
        #     "torch.clone": None,
        #     "torch.dequantize": None,
        #     "torch.empty_like": None,
        #     "torch.eq": None,
        #     "torch.exp": None,
        #     "torch.flatten": None,
        #     "torch.floor": None,
        #     "torch.floor_divide": None,
        #     "torch.fmod": None,
        #     "torch.full_like": None,
        #     "torch.ge": None,
        #     "torch.gt": None,
        #     "torch.le": None,
        #     "torch.lt": None,
        #     "torch.log": None,
        #     "torch.matmul": None,
        #     "torch.max": None,
        #     "torch.min": None,
        #     "torch.mm": None,
        #     "torch.ne": None,
        #     "torch.ones_like": None,
        #     "torch.pow": None,
        #     "torch.quantize_per_tensor": None,
        #     "torch.reshape": None,
        #     "torch.sigmoid": None,
        #     "torch.split": None,
        #     "torch.sqrt": None,
        #     "torch.squeeze": None,
        #     "torch.sum": None,
        #     "torch.tanh": None,
        #     "torch.topk": None,
        #     "torch.transpose": None,
        #     "torch.unsqueeze": None,
        #     "torch.zeros": None,
        #     "torch.zeros_like": None,
        #     "torch._shape_as_tensor": None,
            # ===============================================================
            # call_function builtins and operator
            # ===============================================================
            "getattr": None,
        #     "_operator.add": None,
        #     "_operator.div": None,
        #     "_operator.getitem": None,
        #     "_operator.mul": None,
        #     "_operator.sub": None,
        #     "_operator.truediv": None,
        #     # ===============================================================
        #     # call_module torch.nn.modules
        #     # ===============================================================
        #     "torch.nn.modules.activation.GELU": None,
        #     "torch.nn.modules.activation.PReLU": None,
        #     "torch.nn.modules.activation.ReLU": None,
        #     "torch.nn.modules.activation.Sigmoid": None,
        #     "torch.nn.modules.activation.Softmax": None,
        #     "torch.nn.modules.activation.Tanh": None,
        #     "torch.nn.modules.batchnorm.BatchNorm1d": None,
        #     "torch.nn.modules.batchnorm.BatchNorm2d": None,
        #     "torch.nn.modules.conv.Conv2d": None,
        #     "torch.nn.modules.dropout.Dropout": None,
        #     "torch.nn.modules.linear.Linear": None,
        #     "torch.nn.modules.normalization.LayerNorm": None,
        #     "torch.nn.modules.pooling.AdaptiveAvgPool2d": None,
        #     "torch.nn.modules.pooling.AvgPool1d": None,
        #     "torch.nn.modules.pooling.AvgPool2d": None,
        #     "torch.nn.modules.pooling.AvgPool3d": None,
        #     "torch.nn.modules.pooling.MaxPool2d": None,
        #     "torch.nn.modules.rnn.LSTM": None,
        #     "torch.nn.modules.sparse.Embedding": None,
        #     "torch.nn.modules.sparse.EmbeddingBag": None,
        #     # ===============================================================
        #     # call_method (Tensor)
        #     # ===============================================================
        #     "addmm": None,
        #     "argmax": None,
        #     "argmin": None,
        #     "bmm": None,
        #     "chunk": (([torch.float32],), {}),
        #     "contiguous": (([torch.float32],), {}),
        #     "clamp": None,
        #     "clone": None,
        #     "dequantize": None,
        #     "detach": None,
        #     "eq": (([torch.float32], [torch.float32]), {}),
        #     "exp": None,
        #     "flatten": None,
        #     "float": None,
        #     "fmod": None,
        #     "ge": None,
        #     "gt": None,
        #     "index_put": None,
        #     "int": None,
        #     "le": None,
        #     "lt": None,
        #     "log": None,
        #     "masked_fill": (([torch.float32], [torch.bool]), {}),
        #     "matmul": None,
        #     "max": None,
        #     "min": None,
        #     "mm": None,
        #     "ne": (([torch.float32], [torch.float32]), {}),
        #     "permute": None,
        #     "pow": None,
        #     "reshape": None,
        #     "sigmoid": None,
        #     "split": None,
        #     "sqrt": None,
        #     "sum": None,
        #     "tanh": None,
        #     "to": None,
        #     "topk": None,
        #     "transpose": (([torch.float32],), {}),
        #     "type_as": (([torch.bool, torch.float32], [torch.float32]), {}),
        #     "size": (([torch.float32, torch.bool],), {}),
        #     "squeeze": None,
        #     "unsqueeze": (([torch.bool],), {}),
        #     "view": (([torch.float32],), {}),
        }

        prim_nvfuser_ops = set(torch._prims.__all__).intersection(dir(fd.Ops))

        ops_with_nvfuser_impl = {
            "torch.ops.prims." + name + ".default" : None
            for name in prim_nvfuser_ops
            if getattr(torch.ops.prims, name).default.impl_nvfuser is not None
        }

        merged_support_dict = {**support_dict, **ops_with_nvfuser_impl}

        super().__init__(merged_support_dict)

    # Extension point: can further override operator_support() to skip node if input/ouput tensor types is scalar
    #
    # def is_node_supported(
    #     self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    # ) -> bool: