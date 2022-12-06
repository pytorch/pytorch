import json
import os

import torch
from torch.fx.experimental.proxy_tensor import make_fx

aten = torch.ops.aten


class ConvArgsAnalysis(torch.fx.Interpreter):
    """
    Log arguments like input shape (input, bias, weights shape)
    and options(padding/stride/kernel size/dilation/etc) for
    aten.convolution
    """

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)

        self.nodes_conv_args = {}
        self.conv_arg_names = [
            arg.name for arg in aten.convolution.default._schema.arguments
        ]

    def run(self, *args):
        run_result = super().run(*args)
        if self.nodes_conv_args:
            filename = "tmp/conv_args.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "a") as fd:
                json.dump(self.nodes_conv_args, fd)
                fd.write("\n")
        return run_result

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)

        if n.op == "call_function":
            if n.target == aten.convolution.default:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                assert len(args) == len(
                    self.conv_arg_names
                ), f"aten.convolution should have {len(self.conv_arg_names)} args"
                conv_args = {}
                # collect tensor's shape, stride (channel first or last), dtype
                for i in range(3):
                    arg_name = self.conv_arg_names[i]
                    if args[i] is None:
                        conv_args[arg_name] = {
                            "shape": None,
                            "stride": None,
                            "dtype": None,
                        }
                    else:
                        conv_args[arg_name] = {
                            "shape": args[i].shape,
                            "stride": args[i].stride(),
                            "dtype": str(args[i].dtype),
                        }
                # collect stride/padding/dilation/transposed/output_padding/groups
                for i in range(3, len(args)):
                    arg_name = self.conv_arg_names[i]
                    conv_args[arg_name] = args[i]

                self.nodes_conv_args[n.name.replace("_default", "")] = conv_args
        return result


def conv_args_analysis(gm: torch.fx.GraphModule, example_inputs):
    # lowering graph
    gm = make_fx(gm)(*example_inputs)
    # use Interpreter to logs the args of conv
    ConvArgsAnalysis(gm).run(*example_inputs)
    return gm
