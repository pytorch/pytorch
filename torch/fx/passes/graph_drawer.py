from __future__ import absolute_import, division, print_function, unicode_literals

import hashlib
import torch
import torch.fx
from typing import Dict, Any, TYPE_CHECKING
from torch.fx.node import _get_qualified_name, _format_arg
from torch.fx.passes.shape_prop import TensorMetadata
from torch.fx._compatibility import compatibility
from itertools import chain

__all__ = ['FxGraphDrawer']
try:
    import pydot
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

_COLOR_MAP = {
    "placeholder": '"AliceBlue"',
    "call_module": "LemonChiffon1",
    "get_param": "Yellow2",
    "get_attr": "LightGrey",
    "output": "PowderBlue",
}

_HASH_COLOR_MAP = [
    "CadetBlue1",
    "Coral",
    "DarkOliveGreen1",
    "DarkSeaGreen1",
    "GhostWhite",
    "Khaki1",
    "LavenderBlush1",
    "LightSkyBlue",
    "MistyRose1",
    "MistyRose2",
    "PaleTurquoise2",
    "PeachPuff1",
    "Salmon",
    "Thistle1",
    "Thistle3",
    "Wheat1",
]

_WEIGHT_TEMPLATE = {
    "shape": "record",
    "fillcolor": "Salmon",
    "style": '"filled,rounded"',
    "fontcolor": "#000000",
}

if HAS_PYDOT:
    @compatibility(is_backward_compatible=False)
    class FxGraphDrawer:
        """
        Visualize a torch.fx.Graph with graphviz
        Basic usage:
            g = FxGraphDrawer(symbolic_traced, "resnet18")
            with open("a.svg", "w") as f:
                f.write(g.get_dot_graph().create_svg())

        log_node_meta_keys can be enabled to have generated dot node also
        contains all keys in the node's meta that has a corresponding non-empty
        value or a boolean field equals to True
        """

        def __init__(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            ignore_getattr: bool = False,
            ignore_parameters_and_buffers: bool = False,
            skip_node_names_in_args: bool = True,
            log_node_meta_keys: bool = False,
        ):
            self._name = name
            self._dot_graphs = {
                name: self._to_dot(
                    graph_module, name, ignore_getattr, ignore_parameters_and_buffers, skip_node_names_in_args, log_node_meta_keys
                )
            }

            for node in graph_module.graph.nodes:
                if node.op != "call_module":
                    continue

                leaf_node = self._get_leaf_node(graph_module, node)

                if not isinstance(leaf_node, torch.fx.GraphModule):
                    continue

                self._dot_graphs[f"{name}_{node.target}"] = self._to_dot(
                    leaf_node,
                    f"{name}_{node.target}",
                    ignore_getattr,
                    ignore_parameters_and_buffers,
                    skip_node_names_in_args,
                )

        def get_dot_graph(self, submod_name=None) -> pydot.Dot:
            if submod_name is None:
                return self.get_main_dot_graph()
            else:
                return self.get_submod_dot_graph(submod_name)

        def get_main_dot_graph(self) -> pydot.Dot:
            return self._dot_graphs[self._name]

        def get_submod_dot_graph(self, submod_name) -> pydot.Dot:
            return self._dot_graphs[f"{self._name}_{submod_name}"]

        def get_all_dot_graphs(self) -> Dict[str, pydot.Dot]:
            return self._dot_graphs

        def _get_node_style(self, node: torch.fx.Node) -> Dict[str, str]:
            template = {
                "shape": "record",
                "fillcolor": "#CAFFE3",
                "style": '"filled,rounded"',
                "fontcolor": "#000000",
            }
            if node.op in _COLOR_MAP:
                template["fillcolor"] = _COLOR_MAP[node.op]
            else:
                # Use a random color for each node; based on its name so it's stable.
                target_name = node._pretty_print_target(node.target)
                target_hash = int(hashlib.md5(target_name.encode()).hexdigest()[:8], 16)
                template["fillcolor"] = _HASH_COLOR_MAP[target_hash % len(_HASH_COLOR_MAP)]
            return template

        def _get_leaf_node(
            self, module: torch.nn.Module, node: torch.fx.Node
        ) -> torch.nn.Module:
            py_obj = module
            assert isinstance(node.target, str)
            atoms = node.target.split(".")
            for atom in atoms:
                if not hasattr(py_obj, atom):
                    raise RuntimeError(
                        str(py_obj) + " does not have attribute " + atom + "!"
                    )
                py_obj = getattr(py_obj, atom)
            return py_obj

        def _typename(self, target: Any) -> str:
            if isinstance(target, torch.nn.Module):
                ret = torch.typename(target)
            elif isinstance(target, str):
                ret = target
            else:
                ret = _get_qualified_name(target)

            # Escape "{" and "}" to prevent dot files like:
            # https://gist.github.com/SungMinCho/1a017aab662c75d805c5954d62c5aabc
            # which triggers `Error: bad label format (...)` from dot
            return ret.replace("{", r"\{").replace("}", r"\}")

        def _get_node_label(
            self,
            module: torch.fx.GraphModule,
            node: torch.fx.Node,
            skip_node_names_in_args: bool,
            log_node_meta_keys: bool = False,
        ) -> str:
            def _get_str_for_args_kwargs(arg):
                if isinstance(arg, tuple):
                    prefix, suffix = r"|args=(\l", r",\n)\l"
                    arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
                elif isinstance(arg, dict):
                    prefix, suffix = r"|kwargs={\l", r",\n}\l"
                    arg_strs_list = [
                        f"{k}: {_format_arg(v, max_list_len=8)}"
                        for k, v in arg.items()
                    ]
                else:  # Fall back to nothing in unexpected case.
                    return ""

                # Strip out node names if requested.
                if skip_node_names_in_args:
                    arg_strs_list = [a for a in arg_strs_list if "%" not in a]
                if len(arg_strs_list) == 0:
                    return ""
                arg_strs = prefix + r",\n".join(arg_strs_list) + suffix
                return arg_strs.replace("{", r"\{").replace("}", r"\}")

            meta_keys_str = ""
            if log_node_meta_keys and hasattr(node, "meta"):
                meta_keys_info = ",\n".join(
                    [k for k, v in node.meta.items() if v]
                ) if isinstance(node.meta, dict) else ""
                if meta_keys_info:
                    meta_keys_str = f"|meta_keys={meta_keys_info}"

            label = "{" + f"name=%{node.name}{meta_keys_str}|op_code={node.op}\n"

            if node.op == "call_module":
                leaf_module = self._get_leaf_node(module, node)
                label += r"\n" + self._typename(leaf_module) + r"\n|"
                extra = ""
                if hasattr(leaf_module, "__constants__"):
                    extra = r"\n".join(
                        [f"{c}: {getattr(leaf_module, c)}" for c in leaf_module.__constants__]  # type: ignore[union-attr]
                    )
                label += extra + r"\n"
            else:
                label += f"|target={self._typename(node.target)}" + r"\n"
                if len(node.args) > 0:
                    label += _get_str_for_args_kwargs(node.args)
                if len(node.kwargs) > 0:
                    label += _get_str_for_args_kwargs(node.kwargs)
                label += f"|num_users={len(node.users)}" + r"\n"

            tensor_meta = node.meta.get('tensor_meta')
            label += self._tensor_meta_to_label(tensor_meta)

            return label + "}"

        def _tensor_meta_to_label(self, tm) -> str:
            if tm is None:
                return ""
            elif isinstance(tm, TensorMetadata):
                return self._stringify_tensor_meta(tm)
            elif isinstance(tm, list):
                result = ""
                for item in tm:
                    result += self._tensor_meta_to_label(item)
                return result
            elif isinstance(tm, dict):
                result = ""
                for k, v in tm.items():
                    result += self._tensor_meta_to_label(v)
                return result
            elif isinstance(tm, tuple):
                result = ""
                for item in tm:
                    result += self._tensor_meta_to_label(item)
                return result
            else:
                raise RuntimeError(f"Unsupported tensor meta type {type(tm)}")

        def _stringify_tensor_meta(self, tm: TensorMetadata) -> str:
            result = ""
            if not hasattr(tm, "dtype"):
                print("tm", tm)
            result += "|" + "dtype" + "=" + str(tm.dtype) + r"\n"
            result += "|" + "shape" + "=" + str(tuple(tm.shape)) + r"\n"
            result += "|" + "requires_grad" + "=" + str(tm.requires_grad) + r"\n"
            result += "|" + "stride" + "=" + str(tm.stride) + r"\n"
            if tm.is_quantized:
                assert tm.qparams is not None
                assert "qscheme" in tm.qparams
                qscheme = tm.qparams["qscheme"]
                if qscheme in {
                        torch.per_tensor_affine,
                        torch.per_tensor_symmetric,
                }:
                    result += "|" + "q_scale" + "=" + str(tm.qparams["scale"]) + r"\n"
                    result += "|" + "q_zero_point" + "=" + str(tm.qparams["zero_point"]) + r"\n"
                elif qscheme in {
                        torch.per_channel_affine,
                        torch.per_channel_symmetric,
                        torch.per_channel_affine_float_qparams,
                }:
                    result += "|" + "q_per_channel_scale" + "=" + str(tm.qparams["scale"]) + r"\n"
                    result += "|" + "q_per_channel_zero_point" + "=" + str(tm.qparams["zero_point"]) + r"\n"
                    result += "|" + "q_per_channel_axis" + "=" + str(tm.qparams["axis"]) + r"\n"
                else:
                    raise RuntimeError(f"Unsupported qscheme: {qscheme}")
                result += "|" + "qscheme" + "=" + str(tm.qparams["qscheme"]) + r"\n"
            return result

        def _get_tensor_label(self, t: torch.Tensor) -> str:
            return str(t.dtype) + str(list(t.shape)) + r"\n"

        def _to_dot(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            ignore_getattr: bool,
            ignore_parameters_and_buffers: bool,
            skip_node_names_in_args: bool,
            log_node_meta_keys: bool = False
        ) -> pydot.Dot:
            """
            Actual interface to visualize a fx.Graph. Note that it takes in the GraphModule instead of the Graph.
            If ignore_parameters_and_buffers is True, the parameters and buffers
            created with the module will not be added as nodes and edges.
            """
            dot_graph = pydot.Dot(name, rankdir="TB")

            for node in graph_module.graph.nodes:
                if ignore_getattr and node.op == "get_attr":
                    continue

                style = self._get_node_style(node)
                dot_node = pydot.Node(
                    node.name, label=self._get_node_label(graph_module, node, skip_node_names_in_args, log_node_meta_keys), **style
                )
                dot_graph.add_node(dot_node)

                def get_module_params_or_buffers():
                    for pname, ptensor in chain(
                        leaf_module.named_parameters(), leaf_module.named_buffers()
                    ):
                        pname1 = node.name + "." + pname
                        label1 = (
                            pname1 + "|op_code=get_" + "parameter"
                            if isinstance(ptensor, torch.nn.Parameter)
                            else "buffer" + r"\l"
                        )
                        dot_w_node = pydot.Node(
                            pname1,
                            label="{" + label1 + self._get_tensor_label(ptensor) + "}",
                            **_WEIGHT_TEMPLATE,
                        )
                        dot_graph.add_node(dot_w_node)
                        dot_graph.add_edge(pydot.Edge(pname1, node.name))

                if node.op == "call_module":
                    leaf_module = self._get_leaf_node(graph_module, node)

                    if not ignore_parameters_and_buffers and not isinstance(leaf_module, torch.fx.GraphModule):
                        get_module_params_or_buffers()

            for node in graph_module.graph.nodes:
                if ignore_getattr and node.op == "get_attr":
                    continue

                for user in node.users:
                    dot_graph.add_edge(pydot.Edge(node.name, user.name))

            return dot_graph

else:
    if not TYPE_CHECKING:
        @compatibility(is_backward_compatible=False)
        class FxGraphDrawer:
            def __init__(self, graph_module: torch.fx.GraphModule, name: str, ignore_getattr: bool = False):
                raise RuntimeError('FXGraphDrawer requires the pydot package to be installed. Please install '
                                   'pydot through your favorite Python package manager.')
