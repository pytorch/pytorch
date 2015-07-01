from collections import defaultdict
from pycaffe2 import utils
import sys
import subprocess

try:
  import pydot
except ImportError:
  print ('Cannot import pydot, which is required for drawing a network. This '
         'can usually be installed in python with "pip install pydot". Also, '
         'pydot requires graphviz to convert dot files to pdf: in ubuntu, this '
         'can usually be installed with "sudo apt-get install graphviz".')
  print ('net_drawer will not run correctly. Please install the correct '
         'dependencies.')
  pydot = None

from caffe2.proto import caffe2_pb2
from google.protobuf import text_format

OP_STYLE = {'shape': 'box', 'color': '#0F9D58', 'style': 'filled',
          'fontcolor': '#FFFFFF'}
BLOB_STYLE = {'shape': 'octagon'}

def GetPydotGraph(operators, name, rankdir='LR'):
  graph = pydot.Dot(name, rankdir=rankdir)
  pydot_nodes = {}
  pydot_node_counts = defaultdict(int)
  node_id = 0
  for op_id, op in enumerate(operators):
    if op.name:
      op_node = pydot.Node(
          '%s/%s (op#%d)' % (op.name, op.type, op_id), **OP_STYLE)
    else:
      op_node = pydot.Node(
          '%s (op#%d)' % (op.type, op_id), **OP_STYLE)
    graph.add_node(op_node)
    # print 'Op: %s' % op.name
    # print 'inputs: %s' % str(op.input)
    # print 'outputs: %s' % str(op.output)
    for input_name in op.input:
      if input_name not in pydot_nodes:
        input_node = pydot.Node(
            input_name + str(pydot_node_counts[input_name]),
            label=input_name, **BLOB_STYLE)
        pydot_nodes[input_name] = input_node
      else:
        input_node = pydot_nodes[input_name]
      graph.add_node(input_node)
      graph.add_edge(pydot.Edge(input_node, op_node))
    for output_name in op.output:
      if output_name in pydot_nodes:
        # we are overwriting an existing blob. need to updat the count.
        pydot_node_counts[output_name] += 1
      output_node = pydot.Node(
          output_name + str(pydot_node_counts[output_name]),
          label=output_name, **BLOB_STYLE)
      pydot_nodes[output_name] = output_node
      graph.add_node(output_node)
      graph.add_edge(pydot.Edge(op_node, output_node))
  return graph

def GetPydotGraphMinimal(operators, name, rankdir='LR'):
  """Different from GetPydotGraph, hide all blob nodes and only show op nodes.
  """
  graph = pydot.Dot(name, rankdir=rankdir)
  pydot_nodes = {}
  blob_parents = {}
  pydot_node_counts = defaultdict(int)
  node_id = 0
  for op_id, op in enumerate(operators):
    if op.name:
      op_node = pydot.Node(
          '%s/%s (op#%d)' % (op.name, op.type, op_id), **OP_STYLE)
    else:
      op_node = pydot.Node(
          '%s (op#%d)' % (op.type, op_id), **OP_STYLE)
    graph.add_node(op_node)
    for input_name in op.input:
      if input_name in blob_parents:
        graph.add_edge(pydot.Edge(blob_parents[input_name], op_node))
    for output_name in op.output:
      blob_parents[output_name] = op_node
  return graph

def GetOperatorMapForPlan(plan_def):
  graphs = {}
  for net_id, net in enumerate(plan_def.networks):
    if net.HasField('name'):
      graphs[plan_def.name + "_" + net.name] = net.operators
    else:
      graphs[plan_def.name + "_network_%d" % net_id] = net.operators
  return graphs

def main():
  with open(sys.argv[1], 'r') as fid:
    content = fid.read()
    graphs = utils.GetContentFromProtoString(
        content,{
            caffe2_pb2.PlanDef: lambda x: GetOperatorMapForPlan(x),
            caffe2_pb2.NetDef: lambda x: {x.name: x.operators},
        })
  for key, operators in graphs.iteritems():
    graph = GetPydotGraph(operators, key)
    filename = graph.get_name() + '.dot'
    graph.write(filename, format='raw')
    pdf_filename = filename[:-3] + 'pdf'
    with open(pdf_filename, 'w') as fid:
      try:
        subprocess.call(['dot', '-Tpdf', filename], stdout=fid)
      except OSError:
        print ('pydot requires graphviz to convert dot files to pdf: in ubuntu '
               'this can usually be installed with "sudo apt-get install '
               'graphviz". We have generated the .dot file but will not '
               'generate pdf file for now due to missing graphviz binaries.')

if __name__ == '__main__':
  main()
