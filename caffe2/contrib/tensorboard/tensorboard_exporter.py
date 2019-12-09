from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import bytes
import copy
import logging
import os
import six

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

try:
    # tensorboard>=1.14.0
    from tensorboard.compat.proto import tensor_shape_pb2
    from tensorboard.compat.proto.node_def_pb2 import NodeDef
    from tensorboard.compat.proto.graph_pb2 import GraphDef
except ImportError:
    from tensorflow.core.framework import tensor_shape_pb2
    try:
        # tensorflow>=1.0.0
        from tensorflow import NodeDef, GraphDef
    except ImportError:
        # tensorflow<=0.12.1
        from tensorflow.core.framework.graph_pb2 import NodeDef, GraphDef


def _make_unique_name(seen, name, min_version=0):
    assert name is not None
    i = min_version
    x = '%s_%d' % (name, i) if i else name
    while x in seen:
        i += 1
        x = '%s_%d' % (name, i)
    seen.add(x)
    return x


def _convert_to_ssa(shapes, track_blob_names, ops):
    """
    Convert an operator graph to SSA (i.e. out-of-place).

    I.e. blobs will be renamed so that each blob is produced only once.
    """
    ir = core.IR(ops)
    seen = set()
    versioned = {}
    shapes2 = {}
    track_blob_names2 = {}

    def ssa_name(name, versions):
        assert name in versions
        version = versions[name]
        if (name, version) in versioned:
            return versioned[(name, version)]
        # Always setting name2 = `{name}_{version}` would work, but we also try
        # to avoid a trailing `_0`, so we have to be careful not to introduce
        # name collisions, such as (foo_1, 0) = foo_1 = (foo, 1).
        # Note: operator names (if any) will be handled later.
        name2 = _make_unique_name(seen, name, min_version=version)
        versioned[(name, version)] = name2
        # Transfer shape.
        if name in shapes:
            shapes2[name2] = shapes[name]
        if track_blob_names and name in track_blob_names:
            track_blob_names2[name2] = track_blob_names[name]
        return name2

    for (op, ssa) in zip(ops, ir.ssa):
        assert op is ssa.op
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend(ssa_name(name, ssa.in_versions) for name in inputs)
        op.output.extend(ssa_name(name, ssa.out_versions) for name in outputs)

    shapes.clear()
    shapes.update(shapes2)
    if track_blob_names:
        track_blob_names.clear()
        track_blob_names.update(track_blob_names2)


def _get_blob_names(ops):
    names = set()
    for op in ops:
        names.update(op.input)
        names.update(op.output)
    return {name: name for name in names}


def _remap_keys(m, f):
    m2 = {f(key): value for key, value in six.iteritems(m)}
    m.clear()
    m.update(m2)


def _rename_all(shapes, track_blob_names, ops, f):
    seen = set()
    renamed = {}

    def g(name):
        """ Collision-free version of f.
        """
        if name is None:
            return None
        if name in renamed:
            return renamed[name]
        name2 = _make_unique_name(seen, f(name))
        renamed[name] = name2
        return name2

    for op in ops:
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend(g(name) for name in inputs)
        op.output.extend(g(name) for name in outputs)

    _remap_keys(shapes, g)
    if track_blob_names:
        _remap_keys(track_blob_names, g)
    # Rename all operator names (if any) independently so that the
    # unique-fication happens only once in _fill_missing_operator_names().
    seen.clear()
    renamed.clear()
    for op in ops:
        op.name = g(op.name)


def _add_gradient_scope(shapes, track_blob_names, ops):
    """
    For all operators or blobs with name containing "_grad", add a
    "GRADIENTS/" scope.

    Note: breaks graph execution since the blob -> gradient mapping is
    hardcoded.
    """
    def f(name):
        if '_grad' in name:
            return 'GRADIENTS/{}'.format(name)
        else:
            return name
    _rename_all(shapes, track_blob_names, ops, f)


def _replace_colons(shapes, track_blob_names, ops, repl):
    """
    `:i` has a special meaning in Tensorflow.
    """
    def f(name):
        return name.replace(':', repl)
    _rename_all(shapes, track_blob_names, ops, f)


def _fill_missing_operator_names(ops):
    ''' Give missing operators a name.

    We expect C2 operators to be generally unnamed. This gives them a scope
    (inferred from their outputs) and a name after their type. Duplicates will
    be postfixed by an index.
    '''
    seen = set()
    for op in ops:
        # Make sure operator names don't collide with blobs.
        seen.update(op.input)
        seen.update(op.output)
    for op in ops:
        if op.name:
            name = op.name
        elif op.output or op.input:
            l = [os.path.dirname(name) for name in op.output or op.input]
            scope = os.path.commonprefix(l)
            name = os.path.join(scope, op.type)
        else:
            name = op.type
        assert(name)
        op.name = _make_unique_name(seen, name)


def _tf_device(device_option):
    if not device_option.HasField("device_type"):
        return ""
    if device_option.device_type == caffe2_pb2.CPU:
        return "/cpu:*"
    if device_option.device_type == caffe2_pb2.CUDA:
        return "/gpu:{}".format(device_option.device_id)
    raise Exception("Unhandled device", device_option)


def _add_tf_shape(m, ints):
    sh = tensor_shape_pb2.TensorShapeProto()
    for i in ints:
        dim = tensor_shape_pb2.TensorShapeProto.Dim()
        dim.size = i
        sh.dim.extend([dim])
    m['_output_shapes'].list.shape.extend([sh])


def _set_tf_attr(m, arg):
    k = arg.name
    if k == 'shape' and arg.ints:
        _add_tf_shape(m, arg.ints)
        return
    if arg.HasField("f"):
        m[k].f = arg.f
        return
    if arg.HasField("i"):
        m[k].i = arg.i
        return
    if arg.HasField("s"):
        m[k].s = (
            arg.s if isinstance(arg.s, bytes) else str(arg.s).encode('utf-8')
        )
        return
    if arg.floats:
        m[k].list.f.extend(arg.floats)
        return
    if arg.ints:
        m[k].list.i.extend(arg.ints)
        return
    if arg.strings:
        m[k].list.s.extend(
            s if isinstance(s, bytes) else str(s).encode('utf-8')
            for s in arg.strings
        )
        return
    # The value is an empty list.
    m[k].list.s.extend([])


def _operator_to_node(shapes, op):
    assert op.name, op
    n = NodeDef()
    n.name = op.name
    n.input.extend(op.input)
    n.op = op.type
    n.device = _tf_device(op.device_option)
    if shapes:
        # Add shapes in order.
        for output in op.output:
            if output not in shapes:
                break
            _add_tf_shape(n.attr, shapes[output])
    for arg in op.arg:
        _set_tf_attr(n.attr, arg)
    return n


def _blob_to_node(producing_ops, shapes, name):
    assert name
    n = NodeDef()
    n.name = name
    inputs = producing_ops.get(name, [])
    if inputs:
        n.op = 'Blob'
    else:
        n.op = 'Placeholder'
    n.input.extend('%s:%d' % (op.name, i) for op, i in inputs)
    if inputs:
        device = inputs[0][0].device_option
        if (all(input[0].device_option == device for input in inputs)):
            n.device = _tf_device(device)
    if shapes and name in shapes:
        _add_tf_shape(n.attr, shapes[name])
    return n


def _operators_to_graph_def(
    shapes,
    ops,
    replace_colons='$',
    with_ssa=True,
    with_gradient_scope=True,
    track_blob_names=None,  # pass an empty array to track blob names
):
    if track_blob_names is not None:
        track_blob_names.clear()
        track_blob_names.update(_get_blob_names(ops))
    if replace_colons:
        _replace_colons(shapes, track_blob_names, ops, replace_colons)
    if with_ssa:
        _convert_to_ssa(shapes, track_blob_names, ops)
    if with_gradient_scope:
        _add_gradient_scope(shapes, track_blob_names, ops)
    _fill_missing_operator_names(ops)
    g = GraphDef()
    producing_ops = {}
    blobs = set()
    for op in ops:
        g.node.extend([_operator_to_node(shapes, op)])
        for input_blob in op.input:
            blobs.add(input_blob)
        for i, output_blob in enumerate(op.output):
            blobs.add(output_blob)
            producing_ops.setdefault(output_blob, []).append((op, i))
    for blob in blobs:
        g.node.extend([_blob_to_node(producing_ops, shapes, blob)])
    return g


def _propagate_device_option(net):
    if not net.HasField("device_option"):
        return
    for op in net.op:
        if not op.HasField("device_option"):
            op.device_option.CopyFrom(net.device_option)


def _try_get_shapes(nets):
    try:
        # Note: this will inspect the workspace for better or worse.
        shapes, _ = workspace.InferShapesAndTypes(nets)
        return shapes
    except Exception as e:
        logging.warning('Failed to compute shapes: %s', e)
        return {}


def nets_to_graph_def(nets, shapes=None, **kwargs):
    if shapes is None:
        shapes = _try_get_shapes(nets)
    nets = [copy.deepcopy(net.Proto()) for net in nets]
    shapes = copy.deepcopy(shapes)
    for net in nets:
        _propagate_device_option(net)
    return _operators_to_graph_def(
        shapes,
        [op for net in nets for op in net.op],
        **kwargs
    )


def cnn_to_graph_def(cnn, **kwargs):
    return nets_to_graph_def([cnn.param_init_net, cnn.net], **kwargs)


def ops_to_graph_def(ops, shapes=None, **kwargs):
    ops = copy.deepcopy(ops)
    shapes = copy.deepcopy(shapes or {})
    return _operators_to_graph_def(shapes, ops, **kwargs)
