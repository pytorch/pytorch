from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase, rand_array


class TestPartitionOps(TestCase):
    def test_configs(self):
        # (main dims, partitions,  main type, [list of (extra dims, type)])
        configs = [
            ((10, ), 3),
            ((4, ), 10),
            ((10, 10), 4),
            ((100, ), 2),
            ((5, ), 1),
            ((1, ), 1),
            ((2, 10), 2),
        ]
        suffixes = [
            [],
            [((2, 2), np.float32)],
            [((3, ), np.int64), ((2, ), np.float32)],
        ]
        return [
            (main_dims, parts, main_type, extra, pack)
            for main_dims, parts in configs
            for main_type in [np.int32, np.int64] for extra in suffixes
            for pack in [False, True]
        ]

    def testSharding(self):
        for main_dims, parts, main_type, extra_ins, pack in self.test_configs():
            ins = ['in' + str(i) for i in range(1 + len(extra_ins))]
            outs = [
                'in{}_p{}'.format(i, j)
                for i in range(1 + len(extra_ins)) for j in range(parts)
            ]
            op = core.CreateOperator(
                'Sharding', ins, outs, pack_first_input=(1 if pack else 0))
            x = []
            for i, (dims, t) in enumerate([((), main_type)] + extra_ins):
                if t in [np.float32, np.float64]:
                    d = rand_array(*(main_dims + dims))
                else:
                    d = np.random.randint(-100, 100, (main_dims + dims))
                d = d.astype(t)
                workspace.FeedBlob(ins[i], d)
                x.append(d)

            def sharding(x):
                # numpy has proper modulo op that yields non-negative results
                shards = (x[0] % parts).reshape([-1])
                out = []
                for ind, v in enumerate(x):
                    suffix_shape = v.shape[len(x[0].shape):]
                    accum = [[] for i in range(parts)]
                    a = v.reshape((-1, ) + suffix_shape)
                    if pack and ind == 0:
                        a //= parts
                    for i, s in enumerate(shards):
                        accum[s].append(a[i])

                    def join(a):
                        if not a:
                            return np.empty(shape=(0, ) + suffix_shape)
                        return np.stack(a)

                    out.extend(join(a) for a in accum)
                return out

            workspace.RunOperatorOnce(op)
            ref = sharding(x)
            for name, expected in zip(outs, ref):
                np.testing.assert_array_equal(
                    expected, workspace.FetchBlob(name)
                )
