from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np

from caffe2.python import workspace, experiment_util, data_parallel_model
from caffe2.python import timeout_guard, cnn

import caffe2.python.models.resnet as resnet

'''
Parallelized multi-GPU trainer for Resnet 50. Can be used to train on imagenet
data, for example.
'''


logging.basicConfig()
log = logging.getLogger("resnet50_trainer")
log.setLevel(logging.DEBUG)


def AddImageInput(model, reader, batch_size, img_size):
    '''
    Image input operator that loads data from reader and
    applies certain transformations to the images.
    '''
    data, label = model.ImageInput(
        reader,
        ["data", "label"],
        batch_size=batch_size,
        use_caffe_datum=True,
        mean=128.,
        std=128.,
        scale=256,
        crop=img_size,
        mirror=1
    )

    data = model.StopGradient(data, data)


def AddMomentumParameterUpdate(train_model, LR):
    '''
    Add the momentum-SGD update.
    '''
    params = train_model.GetParams()
    assert(len(params) > 0)
    ONE = train_model.param_init_net.ConstantFill(
        [], "ONE", shape=[1], value=1.0,
    )
    NEGONE = train_model.param_init_net.ConstantFill(
        [], 'NEGONE', shape=[1], value=-1.0,
    )

    for param in params:
        param_grad = train_model.param_to_grad[param]
        param_momentum = train_model.param_init_net.ConstantFill(
            [param], param + '_momentum', value=0.0
        )

        # Update param_grad and param_momentum in place
        train_model.net.MomentumSGD(
            [param_grad, param_momentum, LR],
            [param_grad, param_momentum],
            momentum=0.9,
            nesterov=1
        )

        # Update parameters by applying the moment-adjusted gradient
        train_model.WeightedSum(
            [param, ONE, param_grad, NEGONE],
            param
        )


def RunEpoch(
    args,
    epoch,
    train_model,
    test_model,
    total_batch_size,
    expname,
    explog,
):
    '''
    Run one epoch of the trainer.
    TODO: add checkpointing here.
    '''
    # TODO: add loading from checkpoint
    epoch_iters = int(args.epoch_size / total_batch_size)
    for i in range(epoch_iters):
        log.info("Start iteration {}/{} of epoch {}".format(
            i, epoch_iters, epoch,
        ))

        # This timeout is required (temporarily) since CUDA-NCCL
        # operators might deadlock when synchronizing between GPUs.
        timeout = 600.0 if i == 0 else 60.0
        with timeout_guard.CompleteInTimeOrDie(timeout):
            workspace.RunNet(train_model.net.Proto().name)

        num_images = (i + epoch * epoch_iters) * total_batch_size
        record_freq = total_batch_size * 20

        # Report progress, compute train and test accuracies.
        if num_images % record_freq == 0 and i > 0:
            prefix = "gpu_{}".format(train_model._devices[0])
            accuracy = workspace.FetchBlob(prefix + '/accuracy')
            loss = workspace.FetchBlob(prefix + '/loss')
            learning_rate = workspace.FetchBlob(prefix + '/LR')

            test_accuracy = 0
            ntests = 0

            if (test_model is not None):
                # Run 5 iters of testing
                for t in range(0, 5):
                    workspace.RunNet(test_model.net.Proto().name)
                    for g in test_model._devices:
                        test_accuracy += np.asscalar(workspace.FetchBlob(
                            "gpu_{}".format(g) + '/accuracy'
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
    total_batch_size = args.batch_size
    num_gpus = args.num_gpus
    batch_per_device = total_batch_size // num_gpus

    assert \
        total_batch_size % num_gpus == 0, \
        "Number of GPUs must divide batch size"

    gpus = range(num_gpus)
    log.info("Running on gpus: {}".format(gpus))

    # Create CNNModeLhelper object
    train_model = cnn.CNNModelHelper(
        order="NCHW",
        name="resnet50",
        use_cudnn=True,
        cudnn_exhaustive_search=False
    )

    # Model building functions
    def create_resnet50_model_ops(model):
        [softmax, loss] = resnet.create_resnet50(
            model,
            "data",
            num_input_channels=args.num_channels,
            num_labels=args.num_labels,
            label="label",
        )
        model.Accuracy([softmax, "label"], "accuracy")
        return [loss]

    # SGD
    def add_parameter_update_ops(model):
        model.AddWeightDecay(args.weight_decay)
        ITER = model.Iter("ITER")
        stepsz = int(30 * args.epoch_size / total_batch_size)
        LR = model.net.LearningRate(
            [ITER],
            "LR",
            base_lr=args.base_learning_rate,
            policy="step",
            stepsize=stepsz,
            gamma=0.1,
        )
        AddMomentumParameterUpdate(model, LR)

    # Input. Note that the reader must be shared with all GPUS.
    reader = train_model.CreateDB(
        "reader",
        db=args.train_data,
        db_type=args.db_type,
    )

    def add_image_input(model):
        AddImageInput(
            model,
            reader,
            batch_size=batch_per_device,
            img_size=args.image_size,
        )

    # Create parallelized model
    data_parallel_model.Parallelize_GPU(
        train_model,
        input_builder_fun=add_image_input,
        forward_pass_builder_fun=create_resnet50_model_ops,
        param_update_builder_fun=add_parameter_update_ops,
        devices=gpus,
    )

    # Add test model, if specified
    test_model = None
    if (args.test_data is not None):
        log.info("----- Create test net ----")
        test_model = cnn.CNNModelHelper(
            order="NCHW",
            name="resnet50_test",
            use_cudnn=True,
            cudnn_exhaustive_search=False
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

        data_parallel_model.Parallelize_GPU(
            test_model,
            input_builder_fun=test_input_fn,
            forward_pass_builder_fun=create_resnet50_model_ops,
            param_update_builder_fun=None,
            devices=gpus,
        )
        workspace.RunNetOnce(test_model.param_init_net)
        workspace.CreateNet(test_model.net)

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    expname = "resnet50_gpu%d_b%d_L%d_lr%.2f_v2" % (
        args.num_gpus,
        total_batch_size,
        args.num_labels,
        args.base_learning_rate,
    )
    explog = experiment_util.ModelTrainerLog(expname, args)

    # Run the training one epoch a time
    epoch = 0
    while epoch < args.num_epochs:
        epoch = RunEpoch(
            args,
            epoch,
            train_model,
            test_model,
            total_batch_size,
            expname,
            explog
        )

    # TODO: save final model.


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
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs.")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of color channels")
    parser.add_argument("--image_size", type=int, default=227,
                        help="Input image size (to crop to)")
    parser.add_argument("--num_labels", type=int, default=1000,
                        help="Number of labels")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, total over all GPUs.")
    parser.add_argument("--epoch_size", type=int, default=1500000,
                        help="Number of images/epoch")
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Num epochs.")
    parser.add_argument("--base_learning_rate", type=float, default=0.1,
                        help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization)")
    args = parser.parse_args()

    Train(args)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
