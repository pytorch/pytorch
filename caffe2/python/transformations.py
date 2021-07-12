# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################






import caffe2.python._import_c_extension as C


class Transformer(object):
    def __init__(self):
        pass

    @classmethod
    def runTransform(cls, transform_name, net):
        pb = net.Proto().SerializeToString()
        if C.transform_exists(transform_name):
            output = C.run_transform(transform_name, pb)
        elif C.workspace_transform_exists(transform_name):
            output = C.run_workspace_transform(transform_name, pb)
        else:
            raise AttributeError('Transformation {} not found.'.format(transform_name))
        net.Proto().ParseFromString(output)

    def __getattr__(self, transform_name):
        return lambda net : self.runTransform(transform_name, net)


def fuseNNPACKConvRelu(net):
    net.Proto().ParseFromString(
        C.transform_fuseNNPACKConvRelu(net.Proto().SerializeToString())
    )


def optimizeForMKLDNN(net, training_mode = False):
    net.Proto().ParseFromString(
        C.transform_optimizeForMKLDNN(net.Proto().SerializeToString(), training_mode)
    )


def fuseConvBN(net):
    net.Proto().ParseFromString(
        C.transform_fuseConvBN(net.Proto().SerializeToString())
    )
