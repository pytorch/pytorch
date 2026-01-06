# mypy: allow-untyped-defs
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.versions_pb2 import VersionDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def load_onnx_graph(fname):
    import onnx
    m = onnx.load(fname)  # type: ignore[attr-defined]
    g = m.graph
    return parse(g)


def parse(graph):
    nodes = []
    import itertools

    nodes_proto = list(itertools.chain(graph.input, graph.output))

    for node in nodes_proto:
        print(node.name)
        shapeproto = TensorShapeProto(
            dim=[
                # pyrefly: ignore [missing-attribute]
                TensorShapeProto.Dim(size=d.dim_value)
                for d in node.type.tensor_type.shape.dim
            ]
        )
        nodes.append(
            NodeDef(
                name=node.name.encode(encoding="utf_8"),
                op="Variable",
                input=[],
                attr={
                    "dtype": AttrValue(type=node.type.tensor_type.elem_type),
                    "shape": AttrValue(shape=shapeproto),
                },
            )
        )

    for node in graph.node:
        _attr = [" = ".join([str(f[1]) for f in s.ListFields()]) for s in node.attribute]
        attr = ", ".join(_attr).encode(encoding="utf_8")
        print(node.output[0])
        nodes.append(
            NodeDef(
                name=node.output[0].encode(encoding="utf_8"),
                op=node.op_type,
                input=node.input,
                attr={"parameters": AttrValue(s=attr)},
            )
        )

    # two pass token replacement, appends opname to object id
    mapping = {}
    for node in nodes:
        mapping[node.name] = node.op + "_" + node.name

    return GraphDef(node=nodes, versions=VersionDef(producer=22))
