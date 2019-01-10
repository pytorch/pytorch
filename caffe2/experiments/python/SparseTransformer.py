## @package SparseTransformer
# Module caffe2.experiments.python.SparseTransformer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace
import scipy.sparse


class NetDefNode():

    def __init__(self, name, optype, p=None, op=None):
        self.name = name
        self.optype = optype
        self.ops = {}
        self.prev = {}
        self.insertInput(p)
        self.visited = False
        self.op = op

    def insertInput(self, p):
        """
        Insert input of this op
        also maintain the output of previous op
        p: a node or a list of node
        """
        if isinstance(p, list):
            for i in p:
                self.prev[i.name] = i
                i.ops[self.name] = self
        elif isinstance(p, NetDefNode):
            self.prev[p.name] = p
            p.ops[self.name] = self

    def deleteInput(self, p):
        if isinstance(p, NetDefNode):
            del self.prev[p.name]
            del p.ops[self.name]


def maskNallocate(weight_name):
    """
    Combine mask and weights
    create wcsr, iw, jw, return their names
    """
    w = workspace.FetchBlob(weight_name)
    w_csr = scipy.sparse.csr_matrix(w)
    wcsr = w_csr.data
    iw = w_csr.indptr
    jw = w_csr.indices
    workspace.FeedBlob(weight_name + "wcsr", wcsr)
    workspace.FeedBlob(weight_name + "iw", iw)
    workspace.FeedBlob(weight_name + "jw", jw)
    return weight_name + "wcsr", weight_name + "iw", weight_name + "jw"


def transFCRelu(cur, id2node, name2id, ops, model):
    """
    Add trans before and after this FC_Prune->(Relu)->FC_Prune chain.
    """
    # 1. add trans before the start of this chain
    # assuming that cur is a FC_Prune, and it has only one input
    pre = cur.prev.itervalues().next()
    # Create a node /op and insert it.
    # TODO(wyiming): check whether it is correct here
    current_blob = model.Transpose(cur.op.input[0], cur.op.input[0] + "_trans")
#     print model.net.Proto()
    trans_op = model.net.Proto().op[-1]
    trans_node = NetDefNode(trans_op.output[0], "Transpose", pre, trans_op)
    trans_node.visited = True
    pre_new = trans_node

    # 2. use while loop to visit the chain
    while True:
        # breakup with the parent
        cur.deleteInput(pre)
        if not (cur.optype == "FC_Prune" or cur.optype == "Relu"):
            print("Reaching the end of the chain")
            break
        if len(cur.ops) > 1:
            print("A FC/Relu giving more than 1 useful outputs")
        if cur.optype == "FC_Prune":
            op = cur.op
            wcsr, iw, jw = maskNallocate(op.input[1])
            bias_name = op.input[3]
            # TODO(wyiming): create a new Op here
            current_blob = model.FC_Sparse(current_blob,
                                           cur.op.output[0] + "_Sparse",
                                           wcsr, iw, jw, bias_name)
            sps_op = model.net.Proto().op[-1]
            sps_node = NetDefNode(cur.op.output[0] + "_Sparse",
                                  "FC_Sparse",
                                  pre_new, sps_op)
            sps_node.visited = True
            pre_new = sps_node
        if cur.optype == "Relu":
            op = cur.op
            current_blob = model.Relu(current_blob, current_blob)
            rel_op = model.net.Proto().op[-1]
            rel_node = NetDefNode(str(current_blob), "Relu",
                                  pre_new, rel_op)
            rel_node.visited = True
            pre_new = rel_node

        cur.visited = True
        pre = cur
        flag = False
        for _, temp in cur.ops.iteritems():
            if temp.optype == "Relu" or temp.optype == "FC_Prune":
                flag = True
                cur = temp
        if not flag:
            # assume that there is only 1 output that is not PrintOP
            cur = cur.ops.itervalues().next()
            cur.deleteInput(pre)
            print("No FC/RElu children")
            print(cur.op.type)
            break
    # 3. add trans after this chain like 1.
    current_blob = model.Transpose(current_blob, pre.op.output[0])
    trans_op = model.net.Proto().op[-1]
    trans_node = NetDefNode(str(current_blob), "Transpose", pre_new, trans_op)
    trans_node.visited = True
    cur.insertInput(trans_node)
    print(cur.prev)
    print(trans_node.ops)


def Prune2Sparse(cur, id2node, name2id, ops, model):
    # Assume that FC and Relu takes in only 1 input;
    # If not raise warning
    if not cur.visited and cur.optype == "FC_Prune":
        transFCRelu(cur, id2node, name2id, ops, model)

    cur.visited = True
    for name, n in cur.ops.iteritems():
        Prune2Sparse(n, id2node, name2id, ops, model)


def net2list(net_root):
    """
    Use topological order(BFS) to print the op of a net in a list
    """
    bfs_queue = []
    op_list = []
    cur = net_root
    for _, n in cur.ops.iteritems():
        bfs_queue.append(n)
    while bfs_queue:
        node = bfs_queue[0]
        bfs_queue = bfs_queue[1:]
        op_list.append(node.op)
        for _, n in node.ops.iteritems():
            bfs_queue.append(n)

    return op_list


def netbuilder(model):
    print("Welcome to model checker")
    proto = model.net.Proto()
    net_name2id = {}
    net_id2node = {}
    net_root = NetDefNode("net_root", "root", None)

    for op_id, op in enumerate(proto.op):
        if op.type == "Print":
            continue
        op_name = '%s/%s (op#%d)' % (op.name, op.type, op_id) \
                  if op.name else '%s (op#%d)' % (op.type, op_id)
        # print(op_name)
        op_node = NetDefNode(op_name, op.type, op=op)
        net_id2node[op_id] = op_node

        if_has_layer_input = False
        for input_name in op.input:
            if input_name not in net_name2id:
                # assume that un_occured name are non_layers
                # TODO: write a non-layer checker and log it
                continue
            op_node.insertInput(net_id2node[net_name2id[input_name]])
            if_has_layer_input = True

        if not if_has_layer_input:
            op_node.insertInput(net_root)

        for output_name in op.output:
            net_name2id[output_name] = op_id

    return net_root, net_name2id, net_id2node
