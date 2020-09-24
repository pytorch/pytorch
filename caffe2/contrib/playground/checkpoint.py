




import numpy as np
import pickle
from collections import OrderedDict

from caffe2.proto import caffe2_pb2

from caffe2.python import workspace, core, scope

import logging
logging.basicConfig()
log = logging.getLogger("AnyExpOnTerm")
log.setLevel(logging.DEBUG)


def initialize_params_from_file(
        model, weights_file, num_xpus, opts,
        broadcast_computed_param=False, reset_epoch=False):
    start_epoch, lr, best_metric = initialize_master_xpu_model_params(
        model, weights_file, opts, reset_epoch)
    broadcast_parameters(opts, model, num_xpus, broadcast_computed_param)
    return start_epoch, lr, best_metric


def initialize_master_xpu_model_params(model, weights_file, opts, reset_epoch):
    log.info("Initializing model params from file: {}".format(weights_file))
    with open(weights_file, 'r') as fopen:
        blobs = pickle.load(fopen)
    if 'blobs' in blobs:
        blobs = blobs['blobs']

    start_epoch = 0
    best_metric = float('-inf')
    if 'epoch' in blobs:
        log.info('epoch {} is found in model file'.format(blobs['epoch']))
        if not reset_epoch:
            start_epoch = blobs['epoch']
        else:
            log.info('Reset epoch')
    else:
        log.info('no epoch is found in model file')
    lr = opts['model_param']['base_learning_rate']
    if 'lr' in blobs:
        lr = blobs['lr']
    if 'best_metric' in blobs and not reset_epoch:
        best_metric = blobs['best_metric']

    if model is not None:
        log.info('initialize model parameters using weights file: {}'.format(
            weights_file
        ))
        ws_blobs = workspace.Blobs()
        unscoped_blob_names = OrderedDict()
        for blob in model.GetAllParams():
            unscoped_blob_names[unscope_name(str(blob))] = True
        root_xpu_id = opts['distributed']['first_xpu_id']
        device = opts['distributed']['device']
        caffe2_pb2_DEVICE =\
            caffe2_pb2.CUDA if opts['distributed']['device'] == 'gpu'\
            else caffe2_pb2.CPU
        with core.NameScope('{}_{}'.format(device, root_xpu_id)):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2_DEVICE, 0)):
                for unscoped_blob_name in unscoped_blob_names.keys():
                    scoped_blob_name = scoped_name(unscoped_blob_name)
                    if unscoped_blob_name not in blobs:
                        log.info('{:s} not found'.format(unscoped_blob_name))
                        continue
                    log.info(
                        '{:s} loaded from weights file into: {:s}'.format(
                            unscoped_blob_name, scoped_blob_name
                        )
                    )
                    if scoped_blob_name in ws_blobs:
                        ws_blob = workspace.FetchBlob(scoped_blob_name)
                        if not ws_blob.shape == blobs[unscoped_blob_name].shape:
                            log.info(
                                ('Workspace blob {} with shape {} does '
                                    'not match weights file shape {}').format(
                                            unscoped_blob_name, ws_blob.shape,
                                            blobs[unscoped_blob_name].shape)
                            )
                        else:
                            workspace.FeedBlob(
                                scoped_blob_name,
                                blobs[unscoped_blob_name].astype(
                                    np.float32, copy=False))
    else:
        log.info('Skip initializing model parameters from file: {}'.format(
            weights_file
        ))
    log.info('Complete initialize_master_xpu_model_params')
    return start_epoch, lr, best_metric


def broadcast_parameters(opts, model, num_xpus, broadcast_computed_param=False):
    if num_xpus == 1:
        log.info("only 1 device. Skip parameter broadcast")
        return
    all_params = [model.GetParams()]
    if broadcast_computed_param:
        all_params.append(model.GetComputedParams())
    caffe2_pb2_DEVICE =\
        caffe2_pb2.CUDA if opts['distributed']['device'] == 'gpu'\
        else caffe2_pb2.CPU
    for params in all_params:
        assert len(params) % num_xpus == 0, \
            "Current model doesn't match device number when loading checkpoint"
        params_per_xpu = int(len(params) / num_xpus)
        for idx in range(params_per_xpu):
            blobs = [param for param in params[idx::params_per_xpu]]
            data = workspace.FetchBlob(blobs[0])
            log.info('Broadcasting {} to'.format(str(blobs[0])))
            for i, p in enumerate(blobs[1:]):
                log.info(' |-> {}'.format(str(p)))
                with core.DeviceScope(core.DeviceOption(caffe2_pb2_DEVICE, i+1)):
                    workspace.FeedBlob(p, data)
    log.info("Complete parameter broadcast")


def save_model_params(is_checkpoint, model, checkpoint_path, epoch, opts, best_metric):
    # best_metric=float('-inf')
    if checkpoint_path is None:
        return None

    try:
        save_model_params_blob(
            model, checkpoint_path, epoch, opts, best_metric
        )
    except Exception as e:
        log.warning('Exception from save_model_params {}'.format(str(e)))
    return checkpoint_path


def save_model_params_blob(model, params_file, epoch, opts, best_metric):
    # best_metric=float('-inf')
    log.info("Saving model params...")
    root_xpu_id = opts['distributed']['first_xpu_id']
    device = opts['distributed']['device']
    save_params = [str(param) for param in
                   model.GetParams('{}_{}'.format(device, root_xpu_id))]
    save_computed_params = [str(param) for param in
                            model.GetComputedParams('{}_{}'
                            .format(device, root_xpu_id))]
    save_blobs = {}
    save_blobs['epoch'] = epoch
    save_blobs['best_metric'] = best_metric
    save_blobs['lr'] = \
        workspace.FetchBlob('{}_{}/lr'.format(device, root_xpu_id))
    for param in save_params + save_computed_params:
        scoped_blob_name = str(param)
        unscoped_blob_name = unscope_name(scoped_blob_name)
        if unscoped_blob_name not in save_blobs:
            save_blobs[unscoped_blob_name] = workspace.FetchBlob(
                scoped_blob_name)
            log.debug(
                '{:s} -> {:s}'.format(scoped_blob_name, unscoped_blob_name))
    log.info('to weights file {}'.format(params_file))
    try:
        with open(params_file, 'w') as fwrite:
            pickle.dump(dict(blobs=save_blobs), fwrite, pickle.HIGHEST_PROTOCOL)
    except IOError as e:
        log.error('I/O error({0}): {1}'.format(e.errno, e.strerror))


def unscope_name(blob_name):
    return blob_name[blob_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


def scoped_name(blob_name):
    return scope.CurrentNameScope() + blob_name
