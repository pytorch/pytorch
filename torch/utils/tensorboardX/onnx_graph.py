from .proto.graph_pb2 import GraphDef
from .proto.node_def_pb2 import NodeDef
from .proto.versions_pb2 import VersionDef
from .proto.attr_value_pb2 import AttrValue
from .proto.tensor_shape_pb2 import TensorShapeProto
# from .proto.onnx_pb2 import ModelProto


def gg(fname):
    import onnx  # 0.2.1
    m = onnx.load(fname)
    g = m.graph
    return parse(g)


def parse(graph):
    nodes_proto = []
    nodes = []
    import itertools
    for node in itertools.chain(graph.input, graph.output):
        nodes_proto.append(node)

    for node in nodes_proto:
        print(node.name)
        shapeproto = TensorShapeProto(
            dim=[TensorShapeProto.Dim(size=d.dim_value) for d in node.type.tensor_type.shape.dim])
        nodes.append(NodeDef(
            name=node.name.encode(encoding='utf_8'),
            op='Variable',
            input=[],
            attr={
                'dtype': AttrValue(type=node.type.tensor_type.elem_type),
                'shape': AttrValue(shape=shapeproto),
            })
        )

    for node in graph.node:
        attr = []
        for s in node.attribute:
            attr.append(' = '.join([str(f[1]) for f in s.ListFields()]))
        attr = ', '.join(attr).encode(encoding='utf_8')
        print(node.output[0])
        nodes.append(NodeDef(
            name=node.output[0].encode(encoding='utf_8'),
            op=node.op_type,
            input=node.input,
            attr={'parameters': AttrValue(s=attr)},
        ))
    # two pass token replacement, appends opname to object id
    mapping = {}
    for node in nodes:
        mapping[node.name] = node.op + '_' + node.name

    return GraphDef(node=nodes, versions=VersionDef(producer=22))


def updatenodes(nodes, mapping):
    for node in nodes:
        newname = mapping[node.name]
        node.name = newname
        newinput = []
        for inputnode in list(node.input):
            newinput.append(mapping[inputnode])
            node.input.remove(inputnode)
        node.input.extend(newinput)
    newmap = {}
    for k, v in mapping.items():
        newmap[v] = v
    return nodes, newmap


def findnode(nodes, name):
    """ input: node name
        returns: node object
    """
    for n in nodes:
        if n.name == name:
            return n


def parser(s, nodes, node):
    print(s)
    if len(s) == 0:
        return
    if len(s) > 0:
        if s[0] == node.op:
            print(s[0], node.name, s[1], node.input)
            for n in node.input:
                print(n, s[1])
                parser(s[1], nodes, findnode(nodes, n))
        else:
            return False
