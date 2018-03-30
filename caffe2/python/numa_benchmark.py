from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import time


def build_net(net_name, cross_socket):
    net = core.Net(net_name)
    net.Proto().type = "async_scheduling"
    numa_device_option = caffe2_pb2.DeviceOption()
    numa_device_option.device_type = caffe2_pb2.CPU
    numa_device_option.numa_node_id = 0
    net.XavierFill([], net_name + "/input_blob", shape=[1024, 1024],
                    device_option=numa_device_option)
    if cross_socket:
        numa_device_option.numa_node_id = 1
    net.Copy(net_name + "/input_blob", net_name + "/output_blob",
                device_option=numa_device_option)
    return net


def main():
    assert workspace.IsNUMAEnabled() and workspace.GetNumNUMANodes() >= 2

    single_net = build_net("single_net", False)
    cross_net = build_net("cross_net", True)
    workspace.CreateNet(single_net)
    workspace.CreateNet(cross_net)

    for _ in range(4):
        t = time.time()
        workspace.RunNet(single_net.Name(), 5000)
        print("Single socket time:", time.time() - t)

        t = time.time()
        workspace.RunNet(cross_net.Name(), 5000)
        print("Cross socket time:", time.time() - t)


if __name__ == '__main__':
    core.GlobalInit(["caffe2", "--caffe2_cpu_numa_enabled=1"])
    main()
