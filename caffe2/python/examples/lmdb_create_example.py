## @package lmdb_create_example
# Module caffe2.python.examples.lmdb_create_example





import argparse
import numpy as np

import lmdb
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper

'''
Simple example to create an lmdb database of random image data and labels.
This can be used a skeleton to write your own data import.

It also runs a dummy-model with Caffe2 that reads the data and
validates the checksum is same.
'''


def create_db(output_file):
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)

    checksum = 0
    with env.begin(write=True) as txn:
        for j in range(0, 128):
            # MODIFY: add your own data reader / creator
            label = j % 10
            width = 64
            height = 32

            img_data = np.random.rand(3, width, height)
            # ...

            # Create TensorProtos
            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(img_data.shape)
            img_tensor.data_type = 1

            flatten_img = img_data.reshape(np.prod(img_data.shape))
            img_tensor.float_data.extend(flatten_img)

            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2
            label_tensor.int32_data.append(label)
            txn.put(
                '{}'.format(j).encode('ascii'),
                tensor_protos.SerializeToString()
            )

            checksum += np.sum(img_data) * label
            if (j % 16 == 0):
                print("Inserted {} rows".format(j))

    print("Checksum/write: {}".format(int(checksum)))
    return checksum


def read_db_with_caffe2(db_file, expected_checksum):
    print(">>> Read database...")
    model = model_helper.ModelHelper(name="lmdbtest")
    batch_size = 32
    data, label = model.TensorProtosDBInput(
        [], ["data", "label"], batch_size=batch_size,
        db=db_file, db_type="lmdb")

    checksum = 0

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    for _ in range(0, 4):
        workspace.RunNet(model.net.Proto().name)

        img_datas = workspace.FetchBlob("data")
        labels = workspace.FetchBlob("label")
        for j in range(batch_size):
            checksum += np.sum(img_datas[j, :]) * labels[j]

    print("Checksum/read: {}".format(int(checksum)))
    assert np.abs(expected_checksum - checksum < 0.1), \
        "Read/write checksums dont match"


def main():
    parser = argparse.ArgumentParser(
        description="Example LMDB creation"
    )
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to write the database to",
                        required=True)

    args = parser.parse_args()
    checksum = create_db(args.output_file)

    # For testing reading:
    read_db_with_caffe2(args.output_file, checksum)


if __name__ == '__main__':
    main()
