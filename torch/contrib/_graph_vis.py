"""
Experimental. Tools for visualizing the torch.jit.Graph objects.
"""
import string
import json

_vis_template = string.Template("""
<!doctype html>
<html>
<head>
  <title>$name</title>

  <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.js"></script>
  <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css" rel="stylesheet" type="text/css" />

  <style type="text/css">
    #mynetwork {
      height: 100vh;
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


def write(self, filename):
    """
    Write an html file that visualizes a torch.jit.Graph using vis.js
    Arguments:
        self (torch.jit.Graph): the graph.
        filename (string): the output filename, an html-file.
    """

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
        e = {
            'from': n.unique(),
            'to': i.unique(),
            'arrows': 'from',
        }
        edges.append(e)

    counts = {}
    offset = 0
    for n in self.nodes():
        if len(n.uses()) == 0 or n.kind() == 'Undefined':
            continue
        ident = counts.get(n.kind(), 0)
        counts[n.kind()] = ident + 1
        d = {
            'id': n.unique(),
            'label': '{}_{}'.format(n.kind(), ident),
            'y': offset,
            'fixed': {'y': True},
        }
        if n in self.outputs():
            d['shape'] = 'triangle'

        for i in n.inputs():
            add_edge(i, n)

        nodes.append(d)
        offset += 30

    result = _vis_template.substitute(nodes=json.dumps(nodes),
                                      edges=json.dumps(edges),
                                      options=json.dumps(options),
                                      name=filename)
    with open(filename, 'w') as f:
        f.write(result)
