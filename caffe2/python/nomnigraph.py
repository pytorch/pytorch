from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import caffe2.python._import_c_extension as C
from caffe2.python import core
from caffe2.proto import caffe2_pb2
import os
from subprocess import Popen, PIPE
import errno


class NNModule(object):
    def __init__(self, net=None):
        if net is not None:
            if isinstance(net, core.Net):
                self._NNModule = C.NNModuleFromProtobuf(net.Proto().SerializeToString())
            elif isinstance(net, caffe2_pb2.NetDef):
                self._NNModule = C.NNModuleFromProtobuf(net.SerializeToString())
            else:
                raise Exception(
                    "NNModule can be constructed with core.Net or caffe2_pb2.NetDef types"
                )
        else:
            self._NNModule = C.NNModule()

    @property
    def dataFlow(self):
        return self._NNModule.dataFlow()

    def convertToCaffe2Proto(self, old_proto=None):
        if not old_proto:
            old_proto = caffe2_pb2.NetDef()
        output = self._NNModule.convertToCaffe2Proto(old_proto)
        new_proto = caffe2_pb2.NetDef()
        new_proto.ParseFromString(output)
        return new_proto

    def match(self, pattern):
        for n in self.dataFlow.getMutableNodes():
            m = C.matchSubgraph(n, pattern)
            if m:
                yield m


def render(s):
    s = str(s)
    cmd_exists = lambda x: any(
        os.access(os.path.join(path, x), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    )
    if cmd_exists("graph-easy"):
        p = Popen("graph-easy", stdin=PIPE)
        try:
            p.stdin.write(s.encode("utf-8"))
        except IOError as e:
            if e.errno == errno.EPIPE or e.errno == errno.EINVAL:
                pass
            else:
                # Raise any other error.
                raise

        p.stdin.close()
        p.wait()
    else:
        print(s)


NeuralNetOperator = C.NeuralNetOperator
Operator = C.NeuralNetOperator
NeuralNetData = C.NeuralNetData
Data = C.NeuralNetData
NNSubgraph = C.NNSubgraph
NNMatchGraph = C.NNMatchGraph
Graph = C.Graph
Annotation = C.Annotation
