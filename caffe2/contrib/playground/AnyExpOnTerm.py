from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import caffe2.contrib.playground.AnyExp as AnyExp

import logging
logging.basicConfig()
log = logging.getLogger("AnyExpOnTerm")
log.setLevel(logging.DEBUG)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Any Experiment training.')
    parser.add_argument("--parameters-json", type=json.loads,
                        help='model options in json format', dest="params")

    args = parser.parse_args()
    opts = args.params['opts']
    opts = AnyExp.initOpts(opts)
    log.info('opts is: {}'.format(str(opts)))

    AnyExp.initDefaultModuleMap()

    opts['input']['datasets'] = AnyExp.aquireDatasets(opts)

    # defined this way so that AnyExp.trainFun(opts) can be replaced with
    # some other custermized training function.
    ret = AnyExp.runShardedTrainLoop(opts, AnyExp.trainFun())

    log.info('ret is: {}'.format(str(ret)))
