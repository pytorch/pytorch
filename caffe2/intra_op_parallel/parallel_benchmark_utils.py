from __future__ import absolute_import, division, print_function, unicode_literals

import copy

from caffe2.proto import caffe2_pb2
from caffe2.python import core


def _get_replicated_blob_name(blob_name, numa_node_id):
    return (
        blob_name
        if numa_node_id == 0
        else "replica:{}/{}".format(numa_node_id, blob_name)
    )


def _get_sharded_blob_name(blob_name, numa_node_id):
    return "shard:{}/{}".format(numa_node_id, blob_name)


def _create_copies_for_replica(blob_name, num_numa_nodes):
    return [
        core.CreateOperator(
            "Copy",
            [blob_name],
            _get_replicated_blob_name(blob_name, i),
            device_option=caffe2_pb2.DeviceOption(numa_node_id=i),
        )
        for i in range(1, num_numa_nodes)
    ]


def _create_prepand_dim_split_merge_dim(blob_name, num_numa_nodes):
    default_device_option = caffe2_pb2.DeviceOption(numa_node_id=0)
    new_ops = [
        core.CreateOperator(
            "PrependDim",
            [blob_name],
            blob_name,
            dim_size=num_numa_nodes,
            device_option=default_device_option,
        ),
        core.CreateOperator(
            "Split",
            [blob_name],
            [
                _get_sharded_blob_name(blob_name, numa_node_id)
                for numa_node_id in range(num_numa_nodes)
            ],
            axis=0,
            device_option=default_device_option,
        ),
        core.CreateOperator(
            "MergeDim", [blob_name], blob_name, device_option=default_device_option
        ),
    ]
    new_ops += [
        core.CreateOperator(
            "MergeDim",
            [_get_sharded_blob_name(blob_name, numa_node_id)],
            _get_sharded_blob_name(blob_name, numa_node_id),
            device_option=default_device_option,
        )
        for numa_node_id in range(num_numa_nodes)
    ]
    return new_ops


def parallelize_model(model, engine, num_numa_nodes):
    # Parallelize across sockets
    # Modify init net
    new_ops = []
    for op in model.param_init_net.Proto().op:
        new_ops += [op]
        if op.type == "XavierFill":
            new_ops += _create_copies_for_replica(op.output[0], num_numa_nodes)
            model.net.Proto().external_input.extend(
                [
                    _get_replicated_blob_name(op.output[0], numa_node_id)
                    for numa_node_id in range(1, num_numa_nodes)
                ]
            )
        elif (
            op.type == "ConstantFill"
            and len(op.input) == 1
            and op.output[0] == op.input[0] + "_momentum"
        ):
            new_constant_fills = [copy.deepcopy(op) for i in range(1, num_numa_nodes)]
            for numa_node_id in range(1, num_numa_nodes):
                new_op = new_constant_fills[numa_node_id - 1]
                new_op.input[0] = _get_replicated_blob_name(op.input[0], numa_node_id)
                new_op.output[0] = _get_replicated_blob_name(op.output[0], numa_node_id)
                new_op.device_option.CopyFrom(
                    caffe2_pb2.DeviceOption(numa_node_id=numa_node_id)
                )
            new_ops += new_constant_fills
            model.net.Proto().external_input.extend(
                [
                    _get_replicated_blob_name(op.output[0], numa_node_id)
                    for numa_node_id in range(1, num_numa_nodes)
                ]
            )

    del model.param_init_net.Proto().op[:]
    model.param_init_net.Proto().op.extend(new_ops)

    # Modify predict net

    # Split the input
    input_blob_name = "activation"
    new_ops = _create_prepand_dim_split_merge_dim(input_blob_name, num_numa_nodes)

    for op in model.net.Proto().op:
        if op.type == "FC":
            # Parallelize FC
            new_ops += [
                core.CreateOperator(
                    "FC",
                    [_get_sharded_blob_name(op.input[0], numa_node_id)]
                    + [
                        _get_replicated_blob_name(blob, numa_node_id)
                        for blob in op.input[1:]
                    ],
                    _get_sharded_blob_name(op.output[0], numa_node_id),
                    engine=engine,
                    device_option=caffe2_pb2.DeviceOption(numa_node_id=numa_node_id),
                )
                for numa_node_id in range(num_numa_nodes)
            ]
            output_blob_name = op.output[0]
        elif op.type == "FCGradient":
            # Parallelize FCGradient
            new_ops += [
                core.CreateOperator(
                    "FCGradient",
                    [_get_sharded_blob_name(op.input[0], numa_node_id)]
                    + [_get_replicated_blob_name(op.input[1], numa_node_id)]
                    + [_get_sharded_blob_name(op.input[2], numa_node_id)],
                    [
                        _get_replicated_blob_name(op.output[i], numa_node_id)
                        for i in range(2)
                    ]
                    + [_get_sharded_blob_name(op.output[2], numa_node_id)],
                    device_option=caffe2_pb2.DeviceOption(numa_node_id=numa_node_id),
                    engine=engine,
                )
                for numa_node_id in range(num_numa_nodes)
            ]
        elif op.type == "Adagrad":
            # Parallelize Adagrad
            new_ops += [
                core.CreateOperator(
                    "NUMAAllReduce",
                    [
                        _get_replicated_blob_name(op.input[2], numa_node_id)
                        for numa_node_id in range(num_numa_nodes)
                    ],
                    [
                        _get_replicated_blob_name(op.input[2], numa_node_id)
                        for numa_node_id in range(num_numa_nodes)
                    ],
                    max_num_tasks=4,
                    engine=engine,
                )
            ]
            new_ops += [op]

            new_adagrads = [copy.deepcopy(op) for i in range(1, num_numa_nodes)]
            for numa_node_id in range(1, num_numa_nodes):
                new_op = new_adagrads[numa_node_id - 1]
                for i in range(len(op.input)):
                    new_op.input[i] = _get_replicated_blob_name(
                        op.input[i], numa_node_id
                    )
                for i in range(len(op.output)):
                    new_op.output[i] = _get_replicated_blob_name(
                        op.output[i], numa_node_id
                    )
                new_op.device_option.CopyFrom(
                    caffe2_pb2.DeviceOption(numa_node_id=numa_node_id)
                )
            new_ops += new_adagrads
        elif op.type == "LearningRate":
            new_ops += [op]
            new_ops += [
                core.CreateOperator(
                    "Copy",
                    [op.output[0]],
                    [_get_replicated_blob_name(op.output[0], numa_node_id)],
                    device_option=caffe2_pb2.DeviceOption(numa_node_id=numa_node_id),
                )
                for numa_node_id in range(1, num_numa_nodes)
            ]
        elif (
            op.type == "ConstantFill"
            and op.output[0] == output_blob_name + "_autogen_grad"
        ):
            # Concat the final FC output
            new_ops += [
                core.CreateOperator(
                    "Concat",
                    [
                        _get_sharded_blob_name(output_blob_name, numa_node_id)
                        for numa_node_id in range(num_numa_nodes)
                    ],
                    [output_blob_name, output_blob_name + "_concat_dims"],
                    axis=0,
                )
            ]
            new_ops += [op]
            new_ops += _create_prepand_dim_split_merge_dim(op.output[0], num_numa_nodes)
        else:
            new_ops += [op]

    del model.net.Proto().op[:]
    model.net.Proto().op.extend(new_ops)

    return model.param_init_net, model.net
