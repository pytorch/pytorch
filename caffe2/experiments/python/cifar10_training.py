## @package cifar10_training
# Module caffe2.experiments.python.cifar10_training
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
from libfb import pyinit

from caffe2.python import core, cnn, workspace
from caffe2.python import SparseTransformer
import caffe2.python.models.resnet as resnet


def AddInput(model, batch_size, db, db_type):
    """Adds the data input part."""
    # Load the data from a DB.
    data_uint8, label_orig = model.TensorProtosDBInput(
        [], ["data_uint8", "label_orig"], batch_size=batch_size,
        db=db, db_type=db_type)
    # Since we are going to do float computations, what we will do is to cast
    # the data to float.
    data = model.Cast(data_uint8, "data_nhwc", to=core.DataType.FLOAT)
    data = model.NHWC2NCHW(data, "data")
    data = model.Scale(data, data, scale=float(1. / 256))
    data = model.StopGradient(data, data)

    # Flatten the label
    label = model.net.FlattenToVec(label_orig, "label")
    return data, label


def AddAccuracy(model, softmax, label):
    """Adds an accuracy op to the model"""
    accuracy = model.Accuracy([softmax, label], "accuracy")
    return accuracy


def AddTrainingOperators(model, softmax, label, nn_model):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    loss = model.AveragedLoss(xent, "loss")
    # For bookkeeping purposes, we will also compute the accuracy of the model.
    AddAccuracy(model, softmax, label)
    # Now, this is the key part of the training model: we add all the gradient
    # operators to the model. The gradient is computed with respect to the loss
    # that we computed above.
    model.AddGradientOperators([loss])
    # Now, here what we will do is a very simple stochastic gradient descent.
    ITER = model.Iter("iter")
    # We do a simple learning rate schedule where lr = base_lr * (t ^ gamma)
    # Note that we are doing minimization, so the base_lr is negative so we are
    # going the DOWNHILL direction.

    LR = model.LearningRate(
        ITER, "LR", base_lr=-0.01, policy="step", stepsize=15000, gamma=0.5)
    # ONE is a constant value that is used in the gradient update. We only need
    # to create it once, so it is explicitly placed in param_init_net.
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    # Now, for each parameter, we do the gradient updates.
    for param in model.params:
        # Note how we get the gradient of each parameter - CNNModelHelper keeps
        # track of that.
        param_grad = model.param_to_grad[param]
        # The update is a simple weighted sum: param = param + param_grad * LR
        model.WeightedSum([param, ONE, param_grad, LR], param)


def AddBookkeepingOperators(model):
    """This adds a few bookkeeping operators that we can inspect later.

    These operators do not affect the training procedure: they only collect
    statistics and prints them to file or to logs.
    """
    # Print basically prints out the content of the blob. to_file=1 routes the
    # printed output to a file. The file is going to be stored under
    #     root_folder/[blob name]
    model.Print('accuracy', [], to_file=1)
    model.Print('loss', [], to_file=1)
    # Summarizes the parameters. Different from Print, Summarize gives some
    # statistics of the parameter, such as mean, std, min and max.
    for param in model.params:
        model.Summarize(param, [], to_file=1)
        model.Summarize(model.param_to_grad[param], [], to_file=1)
    # Now, if we really want to be very verbose, we can summarize EVERY blob
    # that the model produces; it is probably not a good idea, because that
    # is going to take time - summarization do not come for free. For this
    # demo, we will only show how to summarize the parameters and their
    # gradients.


def AlexNet(model, data, args):
    conv1 = model.Conv(
        data,
        "conv1",
        3,
        64,
        5,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=2
    )
    relu1 = model.Relu(conv1, "conv1")
    pool1 = model.MaxPool(relu1, "pool1", kernel=3, stride=2)
    conv2 = model.Conv(
        pool1,
        "conv2",
        64,
        192,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu2 = model.Relu(conv2, "conv2")
    pool2 = model.MaxPool(relu2, "pool2", kernel=3, stride=2)
    conv3 = model.Conv(
        pool2,
        "conv3",
        192,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu3 = model.Relu(conv3, "conv3")
    conv4 = model.Conv(
        relu3,
        "conv4",
        384,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu4 = model.Relu(conv4, "conv4")
    conv5 = model.Conv(
        relu4,
        "conv5",
        256,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu5 = model.Relu(conv5, "conv5")
    pool5 = model.MaxPool(relu5, "pool5", kernel=3, stride=2)
    fc6 = model.FC(
        pool5, "fc6", 256 * 3 * 3, 4096, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    relu6 = model.Relu(fc6, "fc6")
    fc7 = model.FC(
        relu6, "fc7", 4096, 4096, ('XavierFill', {}), ('ConstantFill', {})
    )
    relu7 = model.Relu(fc7, "fc7")
    fc8 = model.FC(
        relu7, "fc8", 4096, 10, ('XavierFill', {}), ('ConstantFill', {})
    )
    softmax = model.Softmax(fc8, "pred")
    return softmax


def AlexNet_Prune(model, data, args):
    conv1 = model.Conv(
        data,
        "conv1",
        3,
        64,
        5,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=2
    )
    relu1 = model.Relu(conv1, "conv1")
    pool1 = model.MaxPool(relu1, "pool1", kernel=3, stride=2)
    conv2 = model.Conv(
        pool1,
        "conv2",
        64,
        192,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu2 = model.Relu(conv2, "conv2")
    pool2 = model.MaxPool(relu2, "pool2", kernel=3, stride=2)
    conv3 = model.Conv(
        pool2,
        "conv3",
        192,
        384,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu3 = model.Relu(conv3, "conv3")
    conv4 = model.Conv(
        relu3,
        "conv4",
        384,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu4 = model.Relu(conv4, "conv4")
    conv5 = model.Conv(
        relu4,
        "conv5",
        256,
        256,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        pad=1
    )
    relu5 = model.Relu(conv5, "conv5")
    pool5 = model.MaxPool(relu5, "pool5", kernel=3, stride=2)
    fc6 = model.FC_Prune(
        pool5, "fc6", 256 * 3 * 3, 4096, ('XavierFill', {}),
        ('ConstantFill', {}),
        mask_init=None,
        threshold=args.prune_thres * 2,
        need_compress_rate=True,
        comp_lb=args.comp_lb
    )
    compress_fc6 = fc6[1]
    model.Print(compress_fc6, [], to_file=0)
    fc6 = fc6[0]
    relu6 = model.Relu(fc6, "fc6")
    fc7 = model.FC_Prune(
        relu6, "fc7", 4096, 4096, ('XavierFill', {}), ('ConstantFill', {}),
        mask_init=None,
        threshold=args.prune_thres,
        need_compress_rate=True,
        comp_lb=args.comp_lb
    )
    compress_fc7 = fc7[1]
    model.Print(compress_fc7, [], to_file=0)
    fc7 = fc7[0]
    relu7 = model.Relu(fc7, "fc7")
    fc8 = model.FC(
        relu7, "fc8", 4096, 10, ('XavierFill', {}), ('ConstantFill', {})
    )
    softmax = model.Softmax(fc8, "pred")
    return softmax


def ConvBNReLUDrop(model, currentblob, outputblob,
                   input_dim, output_dim, drop_ratio=None):
    currentblob = model.Conv(
        currentblob,
        outputblob,
        input_dim,
        output_dim,
        3,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        stride=1,
        pad=1
    )
    currentblob = model.SpatialBN(currentblob,
                                  str(currentblob) + '_bn',
                                  output_dim, epsilon=1e-3)
    currentblob = model.Relu(currentblob, currentblob)
    if drop_ratio:
        currentblob = model.Dropout(currentblob,
                                    str(currentblob) + '_dropout',
                                    ratio=drop_ratio)
    return currentblob


def VGG(model, data, args):
    """Adds the VGG-Like kaggle winner Model on Cifar-10
      The original blog about the model can be found on:
          http://torch.ch/blog/2015/07/30/cifar.html
      """
    conv1 = ConvBNReLUDrop(model, data, 'conv1', 3, 64, drop_ratio=0.3)
    conv2 = ConvBNReLUDrop(model, conv1, 'conv2', 64, 64)
    pool2 = model.MaxPool(conv2, 'pool2', kernel=2, stride=1)
    conv3 = ConvBNReLUDrop(model, pool2, 'conv3', 64, 128, drop_ratio=0.4)
    conv4 = ConvBNReLUDrop(model, conv3, 'conv4', 128, 128)
    pool4 = model.MaxPool(conv4, 'pool4', kernel=2, stride=2)

    conv5 = ConvBNReLUDrop(model, pool4, 'conv5', 128, 256, drop_ratio=0.4)
    conv6 = ConvBNReLUDrop(model, conv5, 'conv6', 256, 256, drop_ratio=0.4)
    conv7 = ConvBNReLUDrop(model, conv6, 'conv7', 256, 256)
    pool7 = model.MaxPool(conv7, 'pool7', kernel=2, stride=2)

    conv8 = ConvBNReLUDrop(model, pool7, 'conv8', 256, 512, drop_ratio=0.4)
    conv9 = ConvBNReLUDrop(model, conv8, 'conv9', 512, 512, drop_ratio=0.4)
    conv10 = ConvBNReLUDrop(model, conv9, 'conv10', 512, 512)
    pool10 = model.MaxPool(conv10, 'pool10', kernel=2, stride=2)

    conv11 = ConvBNReLUDrop(model, pool10, 'conv11',
                            512, 512, drop_ratio=0.4)
    conv12 = ConvBNReLUDrop(model, conv11, 'conv12',
                            512, 512, drop_ratio=0.4)
    conv13 = ConvBNReLUDrop(model, conv12, 'conv13', 512, 512)
    pool13 = model.MaxPool(conv13, 'pool13', kernel=2, stride=2)

    fc14 = model.FC(
        pool13, "fc14", 512, 512, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    relu14 = model.Relu(fc14, "fc14")
    pred = model.FC(
        relu14, "pred", 512, 10, ('XavierFill', {}),
        ('ConstantFill', {})
    )
    softmax = model.Softmax(pred, 'softmax')
    return softmax


def ResNet110(model, data, args):
    """
    Residual net as described in section 4.2 of He at. al. (2015)
    """
    return resnet.create_resnet_32x32(
        model,
        data,
        num_input_channels=3,
        num_groups=18,
        num_labels=10,
    )


def ResNet20(model, data, args):
    """
    Residual net as described in section 4.2 of He at. al. (2015)
    """
    return resnet.create_resnet_32x32(
        model,
        data,
        num_input_channels=3,
        num_groups=3,
        num_labels=10,
    )


def sparse_transform(model):
    print("====================================================")
    print("                 Sparse Transformer                ")
    print("====================================================")
    net_root, net_name2id, net_id2node = SparseTransformer.netbuilder(model)
    SparseTransformer.Prune2Sparse(
        net_root,
        net_id2node,
        net_name2id,
        model.net.Proto().op,
        model)
    op_list = SparseTransformer.net2list(net_root)
    del model.net.Proto().op[:]
    model.net.Proto().op.extend(op_list)


def test_sparse(test_model):
    # Sparse Implementation
    sparse_transform(test_model)
    sparse_test_accuracy = np.zeros(100)
    for i in range(100):
        workspace.RunNet(test_model.net.Proto().name)
        sparse_test_accuracy[i] = workspace.FetchBlob('accuracy')
    # After the execution is done, let's plot the values.
    print('Sparse Test Accuracy:')
    print(sparse_test_accuracy)
    print('sparse_test_accuracy: %f' % sparse_test_accuracy.mean())


def trainNtest(model_gen, args):
    print("Print running on GPU: %s" % args.gpu)
    train_model = cnn.CNNModelHelper(
        "NCHW",
        name="Cifar_%s" % (args.model),
        use_cudnn=True,
        cudnn_exhaustive_search=True)
    data, label = AddInput(
        train_model, batch_size=64,
        db=args.train_input_path,
        db_type=args.db_type)
    softmax = model_gen(train_model, data, args)
    AddTrainingOperators(train_model, softmax, label, args.model)
    AddBookkeepingOperators(train_model)

    if args.gpu:
        train_model.param_init_net.RunAllOnGPU()
        train_model.net.RunAllOnGPU()

    # The parameter initialization network only needs to be run once.
    workspace.RunNetOnce(train_model.param_init_net)

    # Now, since we are going to run the main network multiple times,
    # we first create the network - which puts the actual network generated
    # from the protobuf into the workspace - and then call RunNet by
    # its name.
    workspace.CreateNet(train_model.net)

    # On the Python side, we will create two numpy arrays to record the accuracy
    # and loss for each iteration.
    epoch_num = 200
    epoch_iters = 1000
    record = 1000

    accuracy = np.zeros(int(epoch_num * epoch_iters / record))
    loss = np.zeros(int(epoch_num * epoch_iters / record))
    # Now, we will manually run the network for 200 iterations.
    for e in range(epoch_num):
        for i in range(epoch_iters):
            workspace.RunNet(train_model.net.Proto().name)
            if i % record is 0:
                count = int(i / record)
                accuracy[count] = workspace.FetchBlob('accuracy')
                loss[count] = workspace.FetchBlob('loss')
                print('Train Loss: {}'.format(loss[count]))
                print('Train Accuracy: {}'.format(accuracy[count]))

    # Testing model. We will set the batch size to 100, so that the testing
    # pass is 100 iterations (10,000 images in total).
    # For the testing model, we need the data input part, the main LeNetModel
    # part, and an accuracy part. Note that init_params is set False because
    # we will be using the parameters obtained from the test model.
    test_model = cnn.CNNModelHelper(
        order="NCHW", name="cifar10_test", init_params=False)
    data, label = AddInput(
        test_model, batch_size=100,
        db=args.test_input_path,
        db_type=args.db_type)
    softmax = model_gen(test_model, data, args)
    AddAccuracy(test_model, softmax, label)

    # In[11]:
    if args.gpu:
        test_model.param_init_net.RunAllOnGPU()
        test_model.net.RunAllOnGPU()
    # Now, remember that we created the test net? We will run the test
    # pass and report the test accuracy here.
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)
    # On the Python side, we will create two numpy arrays to record the accuracy
    # and loss for each iteration.
    test_accuracy = np.zeros(100)
    for i in range(100):
        workspace.RunNet(test_model.net.Proto().name)
        test_accuracy[i] = workspace.FetchBlob('accuracy')

    print('Train Loss:')
    print(loss)
    print('Train Accuracy:')
    print(accuracy)
    print('Test Accuracy:')
    print(test_accuracy)
    print('test_accuracy: %f' % test_accuracy.mean())

    if args.model == 'AlexNet_Prune':
        test_sparse(test_model)


MODEL_TYPE_FUNCTIONS = {
    'AlexNet': AlexNet,
    'AlexNet_Prune': AlexNet_Prune,
    'VGG': VGG,
    'ResNet-110': ResNet110,
    'ResNet-20': ResNet20
}

if __name__ == '__main__':
    # it's hard to init flags correctly... so here it is
    sys.argv.append('--caffe2_keep_on_shrink')

    # FbcodeArgumentParser calls initFacebook which is necessary for NNLoader
    # initialization
    parser = pyinit.FbcodeArgumentParser(description='cifar-10 Tutorial')

    # arguments starting with single '-' are compatible with Lua
    parser.add_argument("--model", type=str, default='AlexNet',
                        choices=MODEL_TYPE_FUNCTIONS.keys(),
                        help="The batch size of benchmark data.")
    parser.add_argument("--prune_thres", type=float, default=0.0001,
                        help="Pruning threshold for FC layers.")
    parser.add_argument("--comp_lb", type=float, default=0.02,
                        help="Compression Lower Bound for FC layers.")
    parser.add_argument("--gpu", default=False,
                        help="Whether to run on gpu", type=bool)
    parser.add_argument("--train_input_path", type=str,
                        default=None,
                        required=True,
                        help="Path to the database for training data")
    parser.add_argument("--test_input_path", type=str,
                        default=None,
                        required=True,
                        help="Path to the database for test data")
    parser.add_argument("--db_type", type=str,
                        default="lmbd", help="Database type")
    args = parser.parse_args()

    # If you would like to see some really detailed initializations,
    # you can change --caffe2_log_level=0 to --caffe2_log_level=-1
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])

    trainNtest(MODEL_TYPE_FUNCTIONS[args.model], args)
