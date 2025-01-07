import os
import sys
import uuid

import graphviz

from torch import Tensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from padded_tensor import *


def to_shape_str(arg):
    if isinstance(arg, Tensor):
        return [i for i in arg.shape]
    else:
        return arg


def to_original_shape_str(arg):
    if isinstance(arg, PaddedTensor):
        return [i for i in arg.original_shape]
    elif isinstance(arg, Tensor):
        return [i for i in arg.shape]
    else:
        return arg


class SimpleLoggingTensorMode(TorchDispatchMode):
    def __init__(self):
        self.ops = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        func_name_str = str(func)

        arg_shapes = []
        for arg in args:
            arg_shapes.append(str(pytree.tree_map(to_shape_str, arg)))
        arg_shapes_str = "[" + ", ".join(arg_shapes) + "]"
        out_shape_str = str(pytree.tree_map(to_shape_str, out))

        print(
            "{0:40} {1:60} {2:40}".format(func_name_str, arg_shapes_str, out_shape_str)
        )

        if func_name_str not in self.ops:
            self.ops[func_name_str] = []
        self.ops[func_name_str].append((arg_shapes_str, out_shape_str))

        return out


def log_aten_ops(model, inputs):
    with SimpleLoggingTensorMode() as mode:
        model(*inputs)
        return mode.ops


def get_shape_str(args, out):
    arg_shapes = []
    for arg in args:
        arg_shapes.append(str(pytree.tree_map(to_shape_str, arg)))
    arg_shapes_str = "[" + ", ".join(arg_shapes) + "]"
    out_shape_str = str(pytree.tree_map(to_shape_str, out))

    return arg_shapes_str + " -> " + out_shape_str


class DotMode(TorchDispatchMode):
    def __init__(self, max_nodes=sys.maxsize):
        self.max_nodes = max_nodes

        self.g = graphviz.Digraph()
        self.g.attr("node", shape="rectangle")

        self.n_nodes = 0

        self.out_tensor_to_op = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        node_id = str(self.n_nodes)
        self.n_nodes += 1

        if self.n_nodes > self.max_nodes:
            raise Exception("Max nodes reached")

        node_str = node_id + ":" + str(func)
        self.g.node(node_id, node_str)

        if type(out) is list:
            for o in out:
                self.out_tensor_to_op[id(o)] = node_id
        else:
            self.out_tensor_to_op[id(out)] = node_id

        for arg_idx, arg in enumerate(args):
            if isinstance(arg, Tensor):
                arg_id = id(arg)
                # edge_str = "%d\n%s" % (arg_idx, pytree.tree_map(to_size_str, arg))
                edge_str = "o:%s\np:%s" % (
                    pytree.tree_map(to_original_shape_str, arg),
                    pytree.tree_map(to_shape_str, arg),
                )
                if arg_id in self.out_tensor_to_op:
                    self.g.edge(self.out_tensor_to_op[arg_id], node_id, edge_str)
                else:
                    node_str = "Input"
                    self.g.node(str(arg_id), node_str, color="red")
                    self.g.edge(str(arg_id), node_id, edge_str)

                    self.out_tensor_to_op[arg_id] = str(arg_id)

            else:
                arg_str = ""
                if any([type(l) is Tensor for l in pytree.tree_leaves(arg)]):
                    arg_str = "list of tensors"
                else:
                    arg_str = str(arg)

                node_str = "%s\n%s" % (type(arg).__name__, arg_str)
                arg_id = str(uuid.uuid4())
                self.g.node(arg_id, node_str, color="blue")
                self.g.edge(arg_id, node_id)

        return out

    def save_pdf(self, output_file):
        dot_str = self.g.source
        with open(output_file + ".dot", "w") as f:
            f.write(dot_str)
        os.system("dot -Tpdf %s -o %s" % (output_file + ".dot", output_file + ".pdf"))


def dot_aten_graph(model, inputs, output_file, max_nodes=sys.maxsize):
    with DotMode(max_nodes) as mode:
        try:
            model(*inputs)
        except Exception as e:
            mode.g.render(output_file)
            return

        mode.save_pdf(output_file)


def log(*msgs):
    msg = " ".join([str(x) for x in msgs])
    print(msg)
