#! /usr/bin/env python3

import onnx.backend

import argparse
import caffe2.python.workspace as c2_workspace
import glob
import json
import numpy as np
import onnx
import caffe2.python.onnx.frontend
import caffe2.python.onnx.backend
import os
import shutil
import tarfile
import tempfile

import boto3

from six.moves.urllib.request import urlretrieve

from caffe2.python.models.download import downloadFromURLToFile, getURLFromName, deleteDirectory
from caffe2.proto import caffe2_pb2
from onnx import numpy_helper


"""A script converting Caffe2 models to ONNX, and updating ONNX model zoos.

Arguments:
    -v, verbose
    --local-dir, where we store the ONNX and Caffe2 models
    --no-cache, ignore existing models in local-dir
    --clean-test-data, delete all the existing test data when updating ONNX model zoo
    --add-test-data, add add-test-data sets of test data for each ONNX model
    --only-local, run locally (for testing purpose)

Examples:
    # store the data in /home/username/zoo-dir, delete existing test data, ignore local cache,
    # and generate 3 sets of new test data
    python update-caffe2-models.py --local-dir /home/username/zoo-dir --clean-test-data --no-cache --add-test-data 3

"""

# TODO: Add GPU support


def upload_onnx_model(model_name, zoo_dir, backup=False, only_local=False):
    if only_local:
        print('No uploading in local only mode.')
        return
    model_dir = os.path.join(zoo_dir, model_name)
    suffix = '-backup' if backup else ''
    if backup:
        print('Backing up the previous version of ONNX model {}...'.format(model_name))
    rel_file_name = '{}{}.tar.gz'.format(model_name, suffix)
    abs_file_name = os.path.join(zoo_dir, rel_file_name)
    print('Compressing {} model to {}'.format(model_name, abs_file_name))
    with tarfile.open(abs_file_name, 'w:gz') as f:
        f.add(model_dir, arcname=model_name)
    file_size = os.stat(abs_file_name).st_size
    print('Uploading {} ({} MB) to s3 cloud...'.format(abs_file_name, float(file_size) / 1024 / 1024))
    client = boto3.client('s3', 'us-east-1')
    transfer = boto3.s3.transfer.S3Transfer(client)
    transfer.upload_file(abs_file_name, 'download.onnx', 'models/latest/{}'.format(rel_file_name),
                         extra_args={'ACL': 'public-read'})

    print('Successfully uploaded {} to s3!'.format(rel_file_name))


def download_onnx_model(model_name, zoo_dir, use_cache=True, only_local=False):
    model_dir = os.path.join(zoo_dir, model_name)
    if os.path.exists(model_dir):
        if use_cache:
            upload_onnx_model(model_name, zoo_dir, backup=True, only_local=only_local)
            return
        else:
            shutil.rmtree(model_dir)
    url = 'https://s3.amazonaws.com/download.onnx/models/latest/{}.tar.gz'.format(model_name)

    download_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        download_file.close()
        print('Downloading ONNX model {} from {} and save in {} ...\n'.format(
            model_name, url, download_file.name))
        urlretrieve(url, download_file.name)
        with tarfile.open(download_file.name) as t:
            print('Extracting ONNX model {} to {} ...\n'.format(model_name, zoo_dir))
            t.extractall(zoo_dir)
    except Exception as e:
        print('Failed to download/backup data for ONNX model {}: {}'.format(model_name, e))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    finally:
        os.remove(download_file.name)

    if not only_local:
        upload_onnx_model(model_name, zoo_dir, backup=True, only_local=only_local)


def download_caffe2_model(model_name, zoo_dir, use_cache=True):
    model_dir = os.path.join(zoo_dir, model_name)
    if os.path.exists(model_dir):
        if use_cache:
            return
        else:
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    for f in ['predict_net.pb', 'init_net.pb', 'value_info.json']:
        url = getURLFromName(model_name, f)
        dest = os.path.join(model_dir, f)
        try:
            try:
                downloadFromURLToFile(url, dest,
                                      show_progress=False)
            except TypeError:
                # show_progress not supported prior to
                # Caffe2 78c014e752a374d905ecfb465d44fa16e02a28f1
                # (Sep 17, 2017)
                downloadFromURLToFile(url, dest)
        except Exception as e:
            print("Abort: {reason}".format(reason=e))
            print("Cleaning up...")
            deleteDirectory(model_dir)
            raise


def caffe2_to_onnx(caffe2_model_name, caffe2_model_dir):
    caffe2_init_proto = caffe2_pb2.NetDef()
    caffe2_predict_proto = caffe2_pb2.NetDef()

    with open(os.path.join(caffe2_model_dir, 'init_net.pb'), 'rb') as f:
        caffe2_init_proto.ParseFromString(f.read())
        caffe2_init_proto.name = '{}_init'.format(caffe2_model_name)
    with open(os.path.join(caffe2_model_dir, 'predict_net.pb'), 'rb') as f:
        caffe2_predict_proto.ParseFromString(f.read())
        caffe2_predict_proto.name = caffe2_model_name
    with open(os.path.join(caffe2_model_dir, 'value_info.json'), 'rb') as f:
        value_info = json.loads(f.read())

    print('Converting Caffe2 model {} in {} to ONNX format'.format(caffe2_model_name, caffe2_model_dir))
    onnx_model = caffe2.python.onnx.frontend.caffe2_net_to_onnx_model(
        init_net=caffe2_init_proto,
        predict_net=caffe2_predict_proto,
        value_info=value_info
    )

    return onnx_model, caffe2_init_proto, caffe2_predict_proto


def tensortype_to_ndarray(tensor_type):
    shape = []
    for dim in tensor_type.shape.dim:
        shape.append(dim.dim_value)
    if tensor_type.elem_type == onnx.TensorProto.FLOAT:
        type = np.float32
    elif tensor_type.elem_type == onnx.TensorProto.INT:
        type = np.int32
    else:
        raise
    array = np.random.rand(*shape).astype(type)
    return array


def generate_test_input_data(onnx_model, scale):
    real_inputs_names = list(set([input.name for input in onnx_model.graph.input]) - set([init.name for init in onnx_model.graph.initializer]))
    real_inputs = []
    for name in real_inputs_names:
        for input in onnx_model.graph.input:
            if name == input.name:
                real_inputs.append(input)

    test_inputs = []
    for input in real_inputs:
        ndarray = tensortype_to_ndarray(input.type.tensor_type)
        test_inputs.append((input.name, ndarray * scale))

    return test_inputs


def generate_test_output_data(caffe2_init_net, caffe2_predict_net, inputs):
    p = c2_workspace.Predictor(caffe2_init_net, caffe2_predict_net)
    inputs_map = {input[0]:input[1] for input in inputs}

    output = p.run(inputs_map)
    c2_workspace.ResetWorkspace()
    return output


def onnx_verify(onnx_model, inputs, ref_outputs):
    prepared = caffe2.python.onnx.backend.prepare(onnx_model)
    onnx_inputs = []
    for input in inputs:
        if isinstance(input, tuple):
            onnx_inputs.append(input[1])
        else:
            onnx_inputs.append(input)
    onnx_outputs = prepared.run(inputs=onnx_inputs)
    np.testing.assert_almost_equal(onnx_outputs, ref_outputs, decimal=3)


model_mapping = {
    'bvlc_alexnet': 'bvlc_alexnet',
    'bvlc_googlenet': 'bvlc_googlenet',
    'bvlc_reference_caffenet': 'bvlc_reference_caffenet',
    'bvlc_reference_rcnn_ilsvrc13': 'bvlc_reference_rcnn_ilsvrc13',
    'densenet121': 'densenet121',
    #'finetune_flickr_style': 'finetune_flickr_style',
    'inception_v1': 'inception_v1',
    'inception_v2': 'inception_v2',
    'resnet50': 'resnet50',
    'shufflenet': 'shufflenet',
    'squeezenet': 'squeezenet_old',
    #'vgg16': 'vgg16',
    'vgg19': 'vgg19',
    'zfnet512': 'zfnet512',
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update the ONNX models.')
    parser.add_argument('-v', action="store_true", default=False, help="verbose")
    parser.add_argument("--local-dir", type=str, default=os.path.expanduser('~'),
                         help="local dir to store Caffe2 and ONNX models")
    parser.add_argument("--no-cache", action="store_true", default=False,
                         help="whether use local ONNX models")
    parser.add_argument('--clean-test-data', action="store_true", default=False,
                        help="remove the old test data")
    parser.add_argument('--add-test-data', type=int, default=0,
                        help="add new test data")
    parser.add_argument('--only-local', action="store_true", default=False,
                        help="no upload including backup")

    args = parser.parse_args()
    delete_test_data = args.clean_test_data
    add_test_data = args.add_test_data
    use_cache = not args.no_cache
    only_local = args.only_local

    root_dir = args.local_dir
    caffe2_zoo_dir = os.path.join(root_dir, ".caffe2", "models")
    onnx_zoo_dir = os.path.join(root_dir, ".onnx", "models")

    for onnx_model_name in model_mapping:
        c2_model_name = model_mapping[onnx_model_name]

        print('####### Processing ONNX model {} ({} in Caffe2) #######'.format(onnx_model_name, c2_model_name))
        download_caffe2_model(c2_model_name, caffe2_zoo_dir, use_cache=use_cache)
        download_onnx_model(onnx_model_name, onnx_zoo_dir, use_cache=use_cache, only_local=only_local)

        onnx_model_dir = os.path.join(onnx_zoo_dir, onnx_model_name)

        if delete_test_data:
            print('Deleting all the existing test data...')
            # NB: For now, we don't delete the npz files.
            #for f in glob.glob(os.path.join(onnx_model_dir, '*.npz')):
            #    os.remove(f)
            for f in glob.glob(os.path.join(onnx_model_dir, 'test_data_set*')):
                shutil.rmtree(f)

        onnx_model, c2_init_net, c2_predict_net = caffe2_to_onnx(c2_model_name, os.path.join(caffe2_zoo_dir, c2_model_name))

        print('Deleteing old ONNX {} model...'.format(onnx_model_name))
        for f in glob.glob(os.path.join(onnx_model_dir, 'model*'.format(onnx_model_name))):
            os.remove(f)

        print('Serializing generated ONNX {} model ...'.format(onnx_model_name))
        with open(os.path.join(onnx_model_dir, 'model.onnx'), 'wb') as file:
            file.write(onnx_model.SerializeToString())

        print('Verifying model {} with ONNX model checker...'.format(onnx_model_name))
        onnx.checker.check_model(onnx_model)

        total_existing_data_set = 0
        print('Verifying model {} with existing test data...'.format(onnx_model_name))
        for f in glob.glob(os.path.join(onnx_model_dir, '*.npz')):
            test_data = np.load(f, encoding='bytes')
            inputs = list(test_data['inputs'])
            ref_outputs = list(test_data['outputs'])
            onnx_verify(onnx_model, inputs, ref_outputs)
            total_existing_data_set += 1
        for f in glob.glob(os.path.join(onnx_model_dir, 'test_data_set*')):
            inputs = []
            inputs_num = len(glob.glob(os.path.join(f, 'input_*.pb')))
            for i in range(inputs_num):
                tensor = onnx.TensorProto()
                with open(os.path.join(f, 'input_{}.pb'.format(i)), 'rb') as pf:
                    tensor.ParseFromString(pf.read())
                inputs.append(numpy_helper.to_array(tensor))
            ref_outputs = []
            ref_outputs_num = len(glob.glob(os.path.join(f, 'output_*.pb')))
            for i in range(ref_outputs_num):
                tensor = onnx.TensorProto()
                with open(os.path.join(f, 'output_{}.pb'.format(i)), 'rb') as pf:
                    tensor.ParseFromString(pf.read())
                ref_outputs.append(numpy_helper.to_array(tensor))
            onnx_verify(onnx_model, inputs, ref_outputs)
            total_existing_data_set += 1

        starting_index = 0
        while os.path.exists(os.path.join(onnx_model_dir, 'test_data_set_{}'.format(starting_index))):
            starting_index += 1

        if total_existing_data_set == 0 and add_test_data == 0:
            add_test_data = 3
            total_existing_data_set = 3

        print('Generating {} sets of new test data...'.format(add_test_data))
        for i in range(starting_index, add_test_data + starting_index):
            data_dir = os.path.join(onnx_model_dir, 'test_data_set_{}'.format(i))
            os.makedirs(data_dir)
            inputs = generate_test_input_data(onnx_model, 255)
            ref_outputs = generate_test_output_data(c2_init_net, c2_predict_net, inputs)
            onnx_verify(onnx_model, inputs, ref_outputs)
            for index, input in enumerate(inputs):
                tensor = numpy_helper.from_array(input[1])
                with open(os.path.join(data_dir, 'input_{}.pb'.format(index)), 'wb') as file:
                    file.write(tensor.SerializeToString())
            for index, output in enumerate(ref_outputs):
                tensor = numpy_helper.from_array(output)
                with open(os.path.join(data_dir, 'output_{}.pb'.format(index)), 'wb') as file:
                    file.write(tensor.SerializeToString())

        del onnx_model
        del c2_init_net
        del c2_predict_net

        upload_onnx_model(onnx_model_name, onnx_zoo_dir, backup=False, only_local=only_local)

        print('\n\n')
