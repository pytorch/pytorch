## @package resnet50_trainer
# Module caffe2.python.examples.resnet50_trainer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import time
import os

from caffe2.python import core, workspace, experiment_util, data_parallel_model
from caffe2.python import dyndep, optimizer
from caffe2.python import timeout_guard, model_helper, brew

import caffe2.python.models.resnet as resnet
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants as predictor_constants

'''
Parallelized multi-GPU distributed trainer for Resnet 50. Can be used to train
on imagenet data, for example.

To run the trainer in single-machine multi-gpu mode by setting num_shards = 1.

To run the trainer in multi-machine multi-gpu mode with M machines,
run the same program on all machines, specifying num_shards = M, and
shard_id = a unique integer in the set [0, M-1].

For rendezvous (the trainer processes have to know about each other),
you can either use a directory path that is visible to all processes
(e.g. NFS directory), or use a Redis instance. Use the former by
passing the `file_store_path` argument. Use the latter by passing the
`redis_host` and `redis_port` arguments.
'''

logging.basicConfig()
log = logging.getLogger("resnet50_trainer")
log.setLevel(logging.DEBUG)

dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:file_store_handler_ops')
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:redis_store_handler_ops')


def AddImageInput(model, reader, batch_size, img_size):
    '''
    Image input operator that loads data from reader and
    applies certain transformations to the images.
    '''
    data, label = brew.image_input(
        model,
        reader, ["data", "label"],
        batch_size=batch_size,
        use_caffe_datum=True,
        mean=128.,
        std=128.,
        scale=256,
        crop=img_size,
        mirror=1
    )

    data = model.StopGradient(data, data)


def SaveModel(args, train_model, epoch):
    prefix = "[]_{}".format(train_model._device_prefix, train_model._devices[0])
    predictor_export_meta = pred_exp.PredictorExportMeta(
        predict_net=train_model.net.Proto(),
        parameters=data_parallel_model.GetCheckpointParams(train_model),
        inputs=[prefix + "/data"],
        outputs=[prefix + "/softmax"],
        shapes={
            prefix + "/softmax": (1, args.num_labels),
            prefix + "/data": (args.num_channels, args.image_size, args.image_size)
        }
    )

    # save the train_model for the current epoch
    model_path = "%s/%s_%d.mdl" % (
        args.file_store_path,
        args.save_model_name,
        epoch,
    )

    # set db_type to be "minidb" instead of "log_file_db", which breaks
    # the serialization in save_to_db. Need to switch back to log_file_db
    # after migration
    pred_exp.save_to_db(
        db_type="minidb",
        db_destination=model_path,
        predictor_export_meta=predictor_export_meta,
    )


def LoadModel(path, model):
    '''
    Load pretrained model from file
    '''
    log.info("Loading path: {}".format(path))
    meta_net_def = pred_exp.load_from_db(path, 'minidb')
    init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.GLOBAL_INIT_NET_TYPE))
    predict_init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.PREDICT_INIT_NET_TYPE))

    predict_init_net.RunAllOnGPU()
    init_net.RunAllOnGPU()
    assert workspace.RunNetOnce(predict_init_net)
    assert workspace.RunNetOnce(init_net)


def RunEpoch(
    args,
    epoch,
    train_model,
    test_model,
    total_batch_size,
    num_shards,
    expname,
    explog,
):
    '''
    Run one epoch of the trainer.
    TODO: add checkpointing here.
    '''
    # TODO: add loading from checkpoint
    log.info("Starting epoch {}/{}".format(epoch, args.num_epochs))
    epoch_iters = int(args.epoch_size / total_batch_size / num_shards)
    for i in range(epoch_iters):
        # This timeout is required (temporarily) since CUDA-NCCL
        # operators might deadlock when synchronizing between GPUs.
        timeout = 600.0 if i == 0 else 60.0
        with timeout_guard.CompleteInTimeOrDie(timeout):
            t1 = time.time()
            workspace.RunNet(train_model.net.Proto().name)
            t2 = time.time()
            dt = t2 - t1

        fmt = "Finished iteration {}/{} of epoch {} ({:.2f} images/sec)"
        log.info(fmt.format(i + 1, epoch_iters, epoch, total_batch_size / dt))
        prefix = "{}_{}".format(
            train_model._device_prefix,
            train_model._devices[0])
        accuracy = workspace.FetchBlob(prefix + '/accuracy')
        loss = workspace.FetchBlob(prefix + '/loss')
        train_fmt = "Training loss: {}, accuracy: {}"
        log.info(train_fmt.format(loss, accuracy))

    num_images = epoch * epoch_iters * total_batch_size
    prefix = "{}_{}".format(train_model._device_prefix, train_model._devices[0])
    accuracy = workspace.FetchBlob(prefix + '/accuracy')
    loss = workspace.FetchBlob(prefix + '/loss')
    learning_rate = workspace.FetchBlob(prefix + '/conv1_w_lr')
    test_accuracy = 0
    if (test_model is not None):
        # Run 100 iters of testing
        ntests = 0
        for _ in range(0, 100):
            workspace.RunNet(test_model.net.Proto().name)
            for g in test_model._devices:
                test_accuracy += np.asscalar(workspace.FetchBlob(
                    "{}_{}".format(test_model._device_prefix, g) + '/accuracy'
                ))
                ntests += 1
        test_accuracy /= ntests
    else:
        test_accuracy = (-1)

    explog.log(
        input_count=num_images,
        batch_count=(i + epoch * epoch_iters),
        additional_values={
            'accuracy': accuracy,
            'loss': loss,
            'learning_rate': learning_rate,
            'epoch': epoch,
            'test_accuracy': test_accuracy,
        }
    )
    assert loss < 40, "Exploded gradients :("

    # TODO: add checkpointing
    return epoch + 1


def Train(args):
    # Either use specified device list or generate one
    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(',')]
        num_gpus = len(gpus)
    else:
        gpus = list(range(args.num_gpus))
        num_gpus = args.num_gpus

    log.info("Running on GPUs: {}".format(gpus))

    # Verify valid batch size
    total_batch_size = args.batch_size
    batch_per_device = total_batch_size // num_gpus
    assert \
        total_batch_size % num_gpus == 0, \
        "Number of GPUs must divide batch size"

    # Round down epoch size to closest multiple of batch size across machines
    global_batch_size = total_batch_size * args.num_shards
    epoch_iters = int(args.epoch_size / global_batch_size)
    args.epoch_size = epoch_iters * global_batch_size
    log.info("Using epoch size: {}".format(args.epoch_size))

    # Create ModelHelper object
    train_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustice_search': True,
        'ws_nbytes_limit': (args.cudnn_workspace_limit_mb * 1024 * 1024),
    }
    train_model = model_helper.ModelHelper(
        name="resnet50", arg_scope=train_arg_scope
    )

    num_shards = args.num_shards
    shard_id = args.shard_id
    if num_shards > 1:
        # Create rendezvous for distributed computation
        store_handler = "store_handler"
        if args.redis_host is not None:
            # Use Redis for rendezvous if Redis host is specified
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "RedisStoreHandlerCreate", [], [store_handler],
                    host=args.redis_host,
                    port=args.redis_port,
                    prefix=args.run_id,
                )
            )
        else:
            # Use filesystem for rendezvous otherwise
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "FileStoreHandlerCreate", [], [store_handler],
                    path=args.file_store_path,
                )
            )
        rendezvous = dict(
            kv_handler=store_handler,
            shard_id=shard_id,
            num_shards=num_shards,
            engine="GLOO",
            exit_nets=None)
    else:
        rendezvous = None

    # Model building functions
    def create_resnet50_model_ops(model, loss_scale):
        [softmax, loss] = resnet.create_resnet50(
            model,
            "data",
            num_input_channels=args.num_channels,
            num_labels=args.num_labels,
            label="label",
            no_bias=True,
        )
        loss = model.Scale(loss, scale=loss_scale)
        brew.accuracy(model, [softmax, "label"], "accuracy")
        return [loss]

    def add_optimizer(model):
        stepsz = int(30 * args.epoch_size / total_batch_size / num_shards)
        optimizer.add_weight_decay(model, args.weight_decay)
        optimizer.build_sgd(
            model,
            args.base_learning_rate,
            momentum=0.9,
            nesterov=1,
            policy="step",
            stepsize=stepsz,
            gamma=0.1
        )

    # Input. Note that the reader must be shared with all GPUS.
    reader = train_model.CreateDB(
        "reader",
        db=args.train_data,
        db_type=args.db_type,
        num_shards=num_shards,
        shard_id=shard_id,
    )

    def add_image_input(model):
        AddImageInput(
            model,
            reader,
            batch_size=batch_per_device,
            img_size=args.image_size,
        )

    # Create parallelized model
    data_parallel_model.Parallelize(
        train_model,
        input_builder_fun=add_image_input,
        forward_pass_builder_fun=create_resnet50_model_ops,
        optimizer_builder_fun=add_optimizer,
        devices=gpus,
        rendezvous=rendezvous,
        optimize_gradient_memory=True,
        cpu_device=args.use_cpu,
    )

    # Add test model, if specified
    test_model = None
    if (args.test_data is not None):
        log.info("----- Create test net ----")
        test_arg_scope = {
            'order': "NCHW",
            'use_cudnn': True,
            'cudnn_exhaustive_search': True,
        }
        test_model = model_helper.ModelHelper(
            name="resnet50_test", arg_scope=test_arg_scope
        )

        test_reader = test_model.CreateDB(
            "test_reader",
            db=args.test_data,
            db_type=args.db_type,
        )

        def test_input_fn(model):
            AddImageInput(
                model,
                test_reader,
                batch_size=batch_per_device,
                img_size=args.image_size,
            )

        data_parallel_model.Parallelize(
            test_model,
            input_builder_fun=test_input_fn,
            forward_pass_builder_fun=create_resnet50_model_ops,
            param_update_builder_fun=None,
            devices=gpus,
            cpu_device=args.use_cpu,
        )
        workspace.RunNetOnce(test_model.param_init_net)
        workspace.CreateNet(test_model.net)

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    epoch = 0
    # load the pre-trained model and reset epoch
    if args.load_model_path is not None:
        LoadModel(args.load_model_path, train_model)

        # Sync the model params
        data_parallel_model.FinalizeAfterCheckpoint(train_model)

        # reset epoch. load_model_path should end with *_X.mdl,
        # where X is the epoch number
        last_str = args.load_model_path.split('_')[-1]
        if last_str.endswith('.mdl'):
            epoch = int(last_str[:-4])
            log.info("Reset epoch to {}".format(epoch))
        else:
            log.warning("The format of load_model_path doesn't match!")

    expname = "resnet50_gpu%d_b%d_L%d_lr%.2f_v2" % (
        args.num_gpus,
        total_batch_size,
        args.num_labels,
        args.base_learning_rate,
    )
    explog = experiment_util.ModelTrainerLog(expname, args)

    # Run the training one epoch a time
    while epoch < args.num_epochs:
        epoch = RunEpoch(
            args,
            epoch,
            train_model,
            test_model,
            total_batch_size,
            num_shards,
            expname,
            explog
        )

        # Save the model for each epoch
        SaveModel(args, train_model, epoch)

        model_path = "%s/%s_" % (
            args.file_store_path,
            args.save_model_name
        )
        # remove the saved model from the previous epoch if it exists
        if os.path.isfile(model_path + str(epoch - 1) + ".mdl"):
            os.remove(model_path + str(epoch - 1) + ".mdl")


def main():
    # TODO: use argv
    parser = argparse.ArgumentParser(
        description="Caffe2: Resnet-50 training"
    )
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data or 'everstore_sampler'",
                        required=True)
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test data")
    parser.add_argument("--db_type", type=str, default="lmdb",
                        help="Database type (such as lmdb or leveldb)")
    parser.add_argument("--gpus", type=str,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPU devices (instead of --gpus)")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of color channels")
    parser.add_argument("--image_size", type=int, default=227,
                        help="Input image size (to crop to)")
    parser.add_argument("--num_labels", type=int, default=1000,
                        help="Number of labels")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, total over all GPUs")
    parser.add_argument("--epoch_size", type=int, default=1500000,
                        help="Number of images/epoch, total over all machines")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Num epochs.")
    parser.add_argument("--base_learning_rate", type=float, default=0.1,
                        help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--cudnn_workspace_limit_mb", type=int, default=64,
                        help="CuDNN workspace limit in MBs")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of machines in distributed run")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard id.")
    parser.add_argument("--run_id", type=str,
                        help="Unique run identifier (e.g. uuid)")
    parser.add_argument("--redis_host", type=str,
                        help="Host of Redis server (for rendezvous)")
    parser.add_argument("--redis_port", type=int, default=6379,
                        help="Port of Redis server (for rendezvous)")
    parser.add_argument("--file_store_path", type=str, default="/tmp",
                        help="Path to directory to use for rendezvous")
    parser.add_argument("--save_model_name", type=str, default="resnet50_model",
                        help="Save the trained model to a given name")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Load previously saved model to continue training")
    parser.add_argument("--use_cpu", type=bool, default=False,
                        help="Use CPU instead of GPU")

    args = parser.parse_args()

    Train(args)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
