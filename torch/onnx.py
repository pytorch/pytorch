"""
The torch.onnx module contains functions to export models into the ONNX
IR format.  These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
import collections
import string
import json
import math
from ._utils import _range


def export(model, args, f, export_params=True, kwargs=None, verbose=False):
    """
    Export a model into ONNX format.  This exporter runs your model
    once in order to get a trace of its execution to be exported; at the
    moment, it does not support dynamic models (e.g., RNNs.)

    See also: :ref:`onnx-export`

    Arguments:
        model (torch.nn.Module): the model to be exported.
        args (torch.autograd.Variable or tuple of variables): the inputs to
            the model, e.g., such that ``model(*args, **kwargs)`` is a valid
            invocation of the model (see kwargs below).
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name.  A binary Protobuf will be written
            to this file.
        export_params (bool, default True): if specified, all parameters will
            be exported.  Set this to False if you are exporting an
            untrained model.
        kwargs (dict, optional): keyword inputs to the model.
    """
    _export(model, args, f, export_params, kwargs, verbose)


def _export(model, args, f, export_params=True, kwargs=None, verbose=False):
    # Special case for common case of passing a single Variable
    if isinstance(args, torch.autograd.Variable):
        args = (args, )
    if not kwargs:
        kwargs = {}
    trace, torch_out = torch.jit.record_trace(model, *args, **kwargs)
    torch._C._jit_pass_onnx(trace)
    if verbose:
        print(trace)
    # TODO: Don't allocate a in-memory string for the protobuf
    if export_params:
        # NB: OrderedDict values is not actually a list, but trace.export is
        # not duck-typed and expects an actual list.
        proto = trace.export(list(model.state_dict().values()))
    else:
        proto = trace.export()
    torch.serialization._with_file_like(f, "wb", lambda f: f.write(proto))
    return torch_out


attr_pattern = re.compile("^(.+)_([ifstg])$")


def _add_attribute(node, key, value):
    """ initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if isinstance(value, collections.Iterable):
        kind += "s"
    return getattr(node, kind + '_')(name, value)


def _newNode(self, opname, *args, **kwargs):
    n = self.create(opname, args)
    for k, v in sorted(kwargs.items()):
        _add_attribute(n, k, v)
    return n


def _op(self, opname, *args, **kwargs):
    outputs = kwargs.pop('outputs', 1)
    n = self.appendNode(_newNode(self, opname, *args, **kwargs))
    if outputs == 1:
        return n
    return tuple(self.appendNode(self.createSelect(n, i)) for i in _range(outputs))


torch._C.Graph.op = _op


_vis_template = string.Template("""
<!doctype html>
<html>
<head>
  <title>Network | Basic usage</title>

  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.js"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css" rel="stylesheet" type="text/css" />

  <style type="text/css">
    #mynetwork {
      width: 1920px;
      height: 1080px;
      border: 1px solid lightgray;
    }
  </style>
</head>
<body>

<div id="mynetwork"></div>

<script type="text/javascript">
  // create an array with nodes
  var nodes = new vis.DataSet(
    $nodes
  );

  // create an array with edges
  var edges = new vis.DataSet(
    $edges
  );

  // create a network
  var container = document.getElementById('mynetwork');
  var data = {
    nodes: nodes,
    edges: edges
  };
  var options = $options;
  var network = new vis.Network(container, data, options);
</script>
</body>
</html>
""")


def _write_vis(self, filename):
    nodes = []
    edges = []
    options = {}
    for n, i in enumerate(self.inputs()):
        nodes.append({
            'id': i.unique(),
            'label': 'input {}'.format(n),
            'shape': 'square',
        })

    existing = set()

    def add_edge(i_, n):
        i = i_ if i_.kind() != 'Select' else i_.input()
        if (i, n) in existing:
            return
        existing.add((i, n))
        edges.append({
            'from': n.unique(),
            'to': i.unique(),
            'arrows': 'from',
        })

    counts = {}
    for n in self.nodes():
        if len(n.uses()) == 0 or n.kind() == 'Select':
            continue
        ident = counts.get(n.kind(),0)
        counts[n.kind()] = ident + 1
        d = {
            'id': n.unique(),
            'label': '{}_{}'.format(n.kind(), ident),
        }
        if n in self.outputs():
            d['shape'] = 'triangle'

        for i in n.inputs():
            add_edge(i, n)

        nodes.append(d)

    result = _vis_template.substitute(nodes=json.dumps(nodes),
                                      edges=json.dumps(edges),
                                      options=json.dumps(options))
    with open(filename, 'w') as f:
        f.write(result)


torch._C.Graph.write_vis = _write_vis
