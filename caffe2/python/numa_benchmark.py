



from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import time

SHAPE_LEN = 4096
NUM_ITER = 1000
GB = 1024 * 1024 * 1024
NUM_REPLICAS = 48


def build_net(net_name, cross_socket):
    init_net = core.Net(net_name + "_init")
    init_net.Proto().type = "async_scheduling"
    numa_device_option = caffe2_pb2.DeviceOption()
    numa_device_option.device_type = caffe2_pb2.CPU
    numa_device_option.numa_node_id = 0
    for replica_id in range(NUM_REPLICAS):
        init_net.XavierFill([], net_name + "/input_blob_" + str(replica_id),
            shape=[SHAPE_LEN, SHAPE_LEN], device_option=numa_device_option)

    net = core.Net(net_name)
    net.Proto().type = "async_scheduling"
    if cross_socket:
        numa_device_option.numa_node_id = 1
    for replica_id in range(NUM_REPLICAS):
        net.Copy(net_name + "/input_blob_" + str(replica_id),
                net_name + "/output_blob_" + str(replica_id),
                device_option=numa_device_option)
    return init_net, net


def main():
    assert workspace.IsNUMAEnabled() and workspace.GetNumNUMANodes() >= 2

    single_init, single_net = build_net("single_net", False)
    cross_init, cross_net = build_net("cross_net", True)

    workspace.CreateNet(single_init)
    workspace.RunNet(single_init.Name())
    workspace.CreateNet(cross_init)
    workspace.RunNet(cross_init.Name())

    workspace.CreateNet(single_net)
    workspace.CreateNet(cross_net)

    for _ in range(4):
        t = time.time()
        workspace.RunNet(single_net.Name(), NUM_ITER)
        dt = time.time() - t
        print("Single socket time:", dt)
        single_bw = 4 * SHAPE_LEN * SHAPE_LEN * NUM_REPLICAS * NUM_ITER / dt / GB
        print("Single socket BW: {} GB/s".format(single_bw))

        t = time.time()
        workspace.RunNet(cross_net.Name(), NUM_ITER)
        dt = time.time() - t
        print("Cross socket time:", dt)
        cross_bw = 4 * SHAPE_LEN * SHAPE_LEN * NUM_REPLICAS * NUM_ITER / dt / GB
        print("Cross socket BW: {} GB/s".format(cross_bw))
        print("Single BW / Cross BW: {}".format(single_bw / cross_bw))


if __name__ == '__main__':
    core.GlobalInit(["caffe2", "--caffe2_cpu_numa_enabled=1"])
    main()
