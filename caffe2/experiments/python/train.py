## @package train
# Module caffe2.experiments.python.train
"""
Benchmark for ads based model.

To run a benchmark with full forward-backward-update, do e.g.

OMP_NUM_THREADS=8 _build/opt/caffe2/caffe2/fb/ads/train_cpu.lpar \
  --batchSize 100 \
  --hidden 128-64-32 \
  --loaderConfig /mnt/vol/gfsdataswarm-global/namespaces/ads/fblearner/users/ \
    dzhulgakov/caffe2/tests/test_direct_loader.config

For more details, run with --help.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# TODO(jiayq): breaks Caffe2, need to investigate
# from __future__ import unicode_literals

from caffe2.python import workspace, cnn, core
from caffe2.python.fb.models.mlp import (
    mlp,
    mlp_decomp,
    mlp_prune,
    sparse_mlp,
    debug_sparse_mlp,
    debug_sparse_mlp_decomposition,
    debug_sparse_mlp_prune,
)
from caffe2.python.fb.models.loss import BatchLRLoss
from caffe2.python.fb.metrics.metrics import LogScoreReweightedMeasurements
from caffe2.python.fb.executor.executor import Trainer
from caffe2.python.sgd import build_sgd
from caffe2.python import net_drawer
from caffe2.python import SparseTransformer

from collections import namedtuple
import os
import sys
import json
import subprocess
import logging

import numpy as np
from libfb import pyinit
import hiveio.par_init
import fblearner.nn.gen_conf as conf_utils

dyndep.InitOpsLibrary('@/caffe2/caffe2/fb/data:reading_ops')

hiveio.par_init.install_class_path()

for h in logging.root.handlers:
    h.setFormatter(logging.Formatter(
        '%(levelname)s %(asctime)s : %(message)s'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

InputData = namedtuple(
    'InputData', ['data', 'label', 'weight', 'prod_pred', 'sparse_segments'])


def FakeData(args, model):
    logger.info('Input dimensions is %d', args.input_dim)

    workspace.FeedBlob('data', np.random.normal(
        size=(args.batchSize, args.input_dim)).astype(np.float32))
    workspace.FeedBlob('label', np.random.randint(
        2, size=args.batchSize).astype(np.int32))
    workspace.FeedBlob('weight', np.ones(args.batchSize).astype(np.float32))

    sparseBin = 100
    workspace.FeedBlob('eid', np.arange(args.batchSize).astype(np.int32))
    workspace.FeedBlob('key', np.random.randint(
        0, sparseBin, args.batchSize).astype(np.int64))
    workspace.FeedBlob('val', np.ones(args.batchSize).astype(np.float32))

    sparseSegments = [
        {
            'size': sparseBin,
            'eid': 'eid',
            'key': 'key',
            'val': 'val',
        },
    ]

    return InputData(data='data', label='label', weight='weight',
                     prod_pred=None, sparse_segments=sparseSegments)


def NNLoaderData(args, model):
    cfg = conf_utils.loadNNLoaderConfig(args.loaderConfig)
    loaderConfig = conf_utils.getLoaderConfigFromNNLoaderConfig(cfg)
    preperConfig = loaderConfig.preperConfig
    metaFile = preperConfig.metaFile
    assert metaFile, 'meta data not found'

    if type(loaderConfig).__name__ == 'LocalDirectLoader':
        loaderConfig.batchConfig.batchSize = args.batchSize
        logger.info('Batch size = %d', loaderConfig.batchConfig.batchSize)
    else:
        logger.info('Batch size unknown here. will be determined by the reader')

    logger.info('Parsing meta data %s', metaFile)
    cmd = 'cat "{}" | {}'.format(metaFile, args.meta2json)
    meta = json.loads(subprocess.check_output(cmd, shell=True))
    args.input_dim = len(meta['denseFeatureNames'])
    logger.info('Input dimensions is %d', args.input_dim)

    fields = ['data', 'label', 'weight', 'prod_pred']

    sparseSegments = []
    if preperConfig.skipSparse or not preperConfig.sparseSegments.segments:
        logger.info('No sparse features found')
    else:
        segments = loaderConfig.preperConfig.sparseSegments.segments
        logger.info('Found %d sparse segments', len(segments))

        sparseFieldNames = ('eid', 'key', 'val', 'size')
        for i, segment in enumerate(segments):
            sparseData = ['{}_{}'.format(fn, i) for fn in sparseFieldNames[:3]]
            fields.extend(sparseData)

            size = max(sf.mod + sf.offset for sf in segment.inputs)
            sparseSegments.append(
                dict(zip(sparseFieldNames, sparseData + [size])))
            logger.info('Sparse segment %d: %s', i, sparseSegments[-1])

    loader = model.param_init_net.NNLoaderCreate(
        [], json_config=conf_utils.structToString(cfg))

    model.net.NNLoaderRead([loader], fields, add_sparse_bias=True)

    return InputData(*(fields[:4] + [sparseSegments]))


def sparse_transform(model):
    print("====================================================")
    print("                 Sparse Transformer                ")
    print("====================================================")
    net_root, net_name2id, net_id2node = SparseTransformer.netbuilder(model)
    SparseTransformer.Prune2Sparse(
        net_root, net_id2node, net_name2id, model.net.Proto().op, model)
    op_list = SparseTransformer.net2list(net_root)
    del model.net.Proto().op[:]
    model.net.Proto().op.extend(op_list)


def train(model_gen, data_gen, args):
    model = cnn.CNNModelHelper("NCHW", name="mlp")
    input_data = data_gen(args, model)
    logger.info(input_data)
    batch_loss = model_gen(args, model, input_data)

    try:
        print(model.net.Proto())
        graph = net_drawer.GetPydotGraph(model.net.Proto().op, 'net', 'TB')
        netGraphFile = os.path.join(
            os.path.expanduser('~'), 'public_html/net.png')
        logger.info('Drawing network to %s', netGraphFile)
        graph.write(netGraphFile, format='png')
    except Exception as err:
        logger.error('Failed to draw net: %s', err)

    # Add gradients
    model.AddGradientOperators([batch_loss.loss])

    if model.net.Proto().op[-1].output[-1] == 'data_grad':
        logger.info('Skipping grad for data')
        del model.net.Proto().op[-1].output[-1]

    build_sgd(model,
              base_learning_rate=args.rateOfLearning,
              policy="inv",
              gamma=args.learnRateDecay,
              power=args.learnRatePower)

    if args.seed:
        logger.info('Setting random seed to %d', args.seed)
        model.param_init_net._net.device_option.CopyFrom(
            core.DeviceOption(0, 0, random_seed=args.seed))

    if args.gpu:
        model.param_init_net.RunAllOnGPU()
        model.net.RunAllOnGPU()

    if args.net_type:
        model.net.Proto().type = args.net_type
        model.net.Proto().num_workers = args.num_workers

    trainer = Trainer(
        model,
        epoch_size=args.epochSize // args.batchSize,
        num_threads=args.numThreads,
        num_epochs=args.maxEpoch,
        reporter=LogScoreReweightedMeasurements(
            batch_loss, input_data.weight, args.negDownsampleRate,
            args.batchSize, args.last_n_stats))
    trainer.run(args.maxEpoch)

    print(model.net.Proto())
    sparse_transform(model)
    print(model.net.Proto())
    workspace.RunNetOnce(model.net)


def mlp_model(args, model, input_data):
    hiddens = [int(s) for s in args.hidden.split('-')] + [2]
    sums = mlp(model, input_data.data, args.input_dim, hiddens)
    return BatchLRLoss(model, sums, input_data.label)


def mlp_decomp_model(args, model, input_data):
    hiddens = [int(s) for s in args.hidden.split('-')] + [2]
    sums = mlp_decomp(model, input_data.data, args.input_dim, hiddens)
    return BatchLRLoss(model, sums, input_data.label)


def mlp_prune_model(args, model, input_data):
    hiddens = [int(s) for s in args.hidden.split('-')] + [2]
    sums = mlp_prune(model, input_data.data, args.input_dim,
                     hiddens, prune_thres=args.prune_thres,
                     comp_lb=args.compress_lb)
    return BatchLRLoss(model, sums, input_data.label)


def sparse_mlp_model(args, model, input_data):
    hiddens = [int(s) for s in args.hidden.split('-')] + [2]
    sums = sparse_mlp(model, input_data.data, args.input_dim, hiddens,
                      input_data.sparse_segments)
    return BatchLRLoss(model, sums, input_data.label)


def debug_sparse_mlp_model(args, model, input_data):
    hiddens = [int(s) for s in args.hidden.split('-')] + [2]
    sums = debug_sparse_mlp(model, input_data.data, args.input_dim, hiddens,
                            input_data.sparse_segments)
    return BatchLRLoss(model, sums, input_data.label)


def debug_sparse_mlp_decomposition_model(args, model, input_data):
    hiddens = [int(s) for s in args.hidden.split('-')] + [2]
    sums = debug_sparse_mlp_decomposition(model, input_data.data,
                                          args.input_dim, hiddens,
                                          input_data.sparse_segments)
    return BatchLRLoss(model, sums, input_data.label)


def debug_sparse_mlp_prune_model(args, model, input_data):
    hiddens = [int(s) for s in args.hidden.split('-')] + [2]
    sums = debug_sparse_mlp_prune(model, input_data.data, args.input_dim,
                                  hiddens,
                                  input_data.sparse_segments)
    return BatchLRLoss(model, sums, input_data.label)


MODEL_TYPE_FUNCTIONS = {
    'mlp': mlp_model,
    'mlp_decomp': mlp_decomp_model,
    'mlp_prune': mlp_prune_model,
    'sparse_mlp': sparse_mlp_model,
    'debug_sparse_mlp': debug_sparse_mlp_model,
    'debug_sparse_mlp_decomposition': debug_sparse_mlp_decomposition_model,
    'debug_sparse_mlp_prune': debug_sparse_mlp_prune_model,
    # Add more model_type functions here.
}


if __name__ == '__main__':
    # it's hard to init flags correctly... so here it is
    sys.argv.append('--caffe2_keep_on_shrink')

    # FbcodeArgumentParser calls initFacebook which is necessary for NNLoader
    # initialization
    parser = pyinit.FbcodeArgumentParser(description='Ads NN trainer')

    # arguments starting with single '-' are compatible with Lua
    parser.add_argument("-batchSize", type=int, default=100,
                        help="The batch size of benchmark data.")
    parser.add_argument("-loaderConfig", type=str,
                        help="Json file with NNLoader's config. If empty some "
                             "fake data is used")
    parser.add_argument("-meta", type=str, help="Meta file (deprecated)")
    parser.add_argument("-hidden", type=str,
                        help="A dash-separated string specifying the "
                             "model dimensions without the output layer.")
    parser.add_argument("-epochSize", type=int, default=1000000,
                        help="Examples to process in one take")
    parser.add_argument("-maxEpoch", type=int,
                        help="Limit number of epochs, if empty reads all data")
    parser.add_argument("-negDownsampleRate", type=float, default=0.1,
                        help="Used to compute the bias term")
    parser.add_argument("-rateOfLearning", type=float, default=0.02,
                        help="Learning rate, `lr/(1+t*d)^p`")
    parser.add_argument("-learnRateDecay", type=float, default=1e-06,
                        help="d in `lr/(1+t*d)^p`")
    parser.add_argument("-learnRatePower", type=float, default=0.5,
                        help="p in `lr/(1+t*d)^p`")
    parser.add_argument("-numThreads", type=int, help="If set runs hogwild")
    parser.add_argument("-model_type", type=str, default='mlp',
                        choices=MODEL_TYPE_FUNCTIONS.keys(),
                        help="The model to benchmark.")
    parser.add_argument("-seed", type=int, help="random seed.")

    # arguments for Lua compatibility which are not implemented yet
    parser.add_argument("-output", help="not implemented")

    # arguments starting with double '--' are additions of this script
    parser.add_argument("--input_dim", type=int, default=1500,
                        help="The input dimension of benchmark data.")
    parser.add_argument("--gpu", action="store_true",
                        help="If set, run testing on GPU.")
    parser.add_argument("--net_type", type=str,
                        help="Set the type of the network to run with.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers, if the net type has "
                             "multiple workers.")
    parser.add_argument("--last_n_stats", type=int, default=0,
                        help="LastN reporting, big values can slow things down")
    parser.add_argument("--meta2json",
                        default='_bin/fblearner/nn/ads/meta2json.llar',
                        help='Path to meta2json.lar')
    parser.add_argument("--prune_thres", type=float, default=0.00001,
                        help="The threshold to prune the weights")
    parser.add_argument("--compress_lb", type=float, default=0.05,
                        help="The lower bound of layer compression")
    args = parser.parse_args()

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=-1'])
    data_gen = NNLoaderData if args.loaderConfig else FakeData
    train(MODEL_TYPE_FUNCTIONS[args.model_type], data_gen, args)
