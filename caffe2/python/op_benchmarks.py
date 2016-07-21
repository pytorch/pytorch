from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace

init_net = core.Net("init")
net = core.Net("bench")


def benchScatterWeightedSum():
    for itype in [np.int32, np.int64]:
        for isize in [100, 10000]:
            for dsize in [100000]:
                for extra_size in [1, 8]:
                    name = "ScatterWeightedSum_{}_{}x{}_pick_{}".format(
                        itype.__name__, dsize, extra_size, isize)
                    x0 = init_net.UniformFill([], shape=[dsize, extra_size])
                    w0 = init_net.UniformFill([], shape=[1])
                    i = init_net.UniformIntFill([],
                                                shape=[isize],
                                                max=dsize - 1)
                    x1 = init_net.UniformFill([], shape=[isize, extra_size])
                    w1 = init_net.UniformFill([], shape=[1])
                    net.ScatterWeightedSum([x0, w0, i, x1, w1], [x0], name=name)


def benchGather():
    for itype in [np.int32, np.int64]:
        for isize in [100, 10000]:
            for dsize in [100000]:
                for extra_size in [1, 8]:
                    name = "Gather_{}_{}x{}_pick_{}".format(
                        itype.__name__, dsize, extra_size, isize)
                    d = init_net.UniformFill([], shape=[dsize, extra_size])
                    i = init_net.UniformIntFill([],
                                                shape=[isize],
                                                max=dsize - 1)
                    net.Gather([d, i], name=name)


def benchDenseFtrl():
    for size in [100, 10000]:
        for engine in [None, "SIMD"]:
            w = init_net.UniformFill([], shape=[size])
            nz = init_net.UniformFill([], shape=[size, 2])
            g = init_net.UniformFill([], shape=[size])
            net.Ftrl(
                [w, nz, g], [w, nz],
                name="{}Ftrl_{}".format('' if engine is None else engine, size),
                engine=engine)


def benchSparseFtrl():
    for itype in [np.int32, np.int64]:
        for isize in [100, 10000]:
            for dsize in [100000]:
                for extra_size in [1, 8]:
                    for engine in [None, "SIMD"]:
                        name = "Sparse{}Ftrl_{}_{}x{}_pick_{}".format(
                            '' if engine is None else engine,
                            itype.__name__, dsize, extra_size, isize)
                        d = init_net.UniformFill([], shape=[dsize, extra_size])
                        nz = init_net.UniformFill([],
                                                  shape=[dsize, extra_size, 2])
                        i = init_net.UniformIntFill([],
                                                    shape=[isize],
                                                    max=dsize - 1)
                        g = init_net.UniformFill([], shape=[isize, extra_size])
                        net.SparseFtrl(
                            [d, nz, i, g], [d, nz], name=name, engine=engine)


if __name__ == '__main__':
    for k, v in globals().items():
        if k.startswith('bench'):
            v()
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    workspace.RunNetOnce(init_net)
    workspace.CreateNet(net)
    workspace.BenchmarkNet(net.Proto().name, 1, 100, True)
