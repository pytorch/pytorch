from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from caffe2.proto import caffe2_pb2


def _parseFile(filename):
    out_net = caffe2_pb2.NetDef()
    # TODO(bwasti): A more robust handler for pathnames.
    dir_path = os.path.dirname(__file__)
    with open('{dir_path}/{filename}'.format(dir_path=dir_path,
                                             filename=filename), 'rb') as f:
        out_net.ParseFromString(f.read())
    return out_net


init_net = _parseFile('init_net.pb')
predict_net = _parseFile('predict_net.pb')
