

from collections import defaultdict

import caffe2.python.nomnigraph as ng
from caffe2.python import core, utils


def transpose_network(nn):
    """
    Convert all Convolutions operators which are in the NCHW order
    to NHWC order and also transform their inputs and outputs so that the
    rest of the graph is not affected.
    """
    # track the incoming tensors into NHWC2NCHW operators
    incoming = {}  # output tensor -> input tensor
    # track outgoing tensors from NCHW2NHWC operators
    outgoing = defaultdict(lambda: [])  # input tensor -> list of operators
    dfg = nn.dataFlow
    orig_nodes = [x for x in nn.nodes]
    for node in orig_nodes:
        if node.isOperator() and node.name == "Conv":
            arg_dict = utils.ArgsToDict(node.annotation.operator_def.arg)
            # a missing "order" argument implies default NCHW order
            if "order" in arg_dict and arg_dict["order"] != "NCHW":
                continue
            inputs = [x for x in node.inputs]
            assert len(inputs) >= 2, "Conv operator should have two inputs"
            outputs = [x for x in node.outputs]
            assert len(outputs) >= 1, "Conv operator should have an output"
            for inp in inputs:
                nn.deleteEdge(inp, node)
            for outp in outputs:
                nn.deleteEdge(node, outp)
            # only the first two inputs of the Convolution the data and the
            # weights need to be transformed
            for idx in range(2):
                new_inp = nn.createUniqueDataNode(inputs[idx].name)
                transp = dfg.createNode(ng.NeuralNetOperator("NCHW2NHWC"))
                nn.createEdge(inputs[idx], transp)
                nn.createEdge(transp, new_inp)
                outgoing[inputs[idx]].append(transp)
                inputs[idx] = new_inp
            for idx in range(len(outputs)):
                new_outp = nn.createUniqueDataNode(outputs[idx].name)
                transp = dfg.createNode(ng.NeuralNetOperator("NHWC2NCHW"))
                nn.createEdge(transp, outputs[idx])
                nn.createEdge(new_outp, transp)
                incoming[outputs[idx]] = new_outp
                outputs[idx] = new_outp
            # create a new Convolution with identical arguments as the original
            # one except for the order
            arg_dict["order"] = "NHWC"
            new_node = nn.createNode(core.CreateOperator("Conv", [], [],
                                                         **arg_dict))
            for inp in inputs:
                nn.createEdge(inp, new_node)
            for outp in outputs:
                nn.createEdge(new_node, outp)

            nn.deleteNode(node)

    # finally, we will compress
    # case 1:
    # X -> NHWC2NCHW -> Y -> NCHW2NHWC -> Z1 ; Y -> NCHW2NHWC -> Z2
    #  to:
    # X -> NHWC2NCHW -> Y   and replace Z1 with X and replace Z2 with X
    # And case 2:
    # Y -> NCHW2NHWC -> Z1 ; Y -> NCHW2NHWC -> Z2
    #  to:
    # Y -> NCHW2NHWC -> Z1     and   replace Z2 with Z1

    # orig_tensor is one of the tensors in the original graph in NCHW order
    for orig_tensor in outgoing:
        # new_tensor is identical to orig_tensor except the order is NHWC
        if orig_tensor in incoming:  # case 1 (see above)
            new_tensor = incoming[orig_tensor]
        else:  # case 2 (see above)
            out_ops = outgoing[orig_tensor]
            new_tensor = out_ops[0].outputs[0]
            outgoing[orig_tensor] = out_ops[1:]

        for opnode in outgoing[orig_tensor]:
            # there should only be one output, so this iteration is overkill
            for out in opnode.outputs:
                nn.replaceAllUsesWith(out, new_tensor)
                nn.deleteNode(out)
            nn.deleteNode(opnode)
