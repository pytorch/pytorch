



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

    def testPartition(self):
        for main_dims, parts, main_type, extra_ins, pack in self.test_configs():
            ins = ['in' + str(i) for i in range(1 + len(extra_ins))]
            outs = [
                'in{}_p{}'.format(j, i)
                for i in range(parts) for j in range(1 + len(extra_ins))
            ]
            op = core.CreateOperator(
                'Partition', ins, outs, pack_first_input=(1 if pack else 0))
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
                for i in range(parts):
                    for ind, v in enumerate(x):
                        suffix_shape = v.shape[len(x[0].shape):]
                        accum = []
                        data = v.reshape((-1, ) + suffix_shape)

                        if pack and ind == 0:
                            data = data // parts

                        for j, s in enumerate(shards):
                            if s == i:
                                accum.append(data[j])

                        def join(a):
                            if not a:
                                return np.empty(shape=(0, ) + suffix_shape)
                            return np.stack(a)

                        out.append(join(accum))
                return out

            workspace.RunOperatorOnce(op)
            ref = sharding(x)
            print(x)
            print(ref)
            for name, expected in zip(outs, ref):
                np.testing.assert_array_equal(
                    expected, workspace.FetchBlob(name)
                )

            # test inverse operation (GatherByKey)
            if len(main_dims) == 1:
                # currently only 1D key tensor supported
                for i in range(len(extra_ins)):
                    expected_out = ins[i + 1]
                    gather_ins = [ins[0]] + [
                        outs[len(ins) * p + i + 1] for p in range(parts)]
                    actual_out = expected_out + '_actual'
                    op = core.CreateOperator(
                        'GatherByKey', gather_ins, actual_out)
                    workspace.RunOperatorOnce(op)
                    expected = workspace.FetchBlob(expected_out)
                    actual = workspace.FetchBlob(actual_out)
                    np.testing.assert_array_equal(expected, actual)


    def testLengthsPartition(self):
        for main_dims, parts, main_type, extra_ins, pack in self.test_configs():
            # For LengthsSharding only 1-D tensors supported as a first input
            if len(main_dims) > 1:
                continue
            ins = ['in' + str(i) for i in range(2 + len(extra_ins))]
            outs = [
                'in{}_p{}'.format(j, i)
                for i in range(parts) for j in range(2 + len(extra_ins))
            ]
            op = core.CreateOperator(
                'LengthsPartition', ins, outs,
                pack_first_input=(1 if pack else 0)
            )
            x = []
            for i, (dims, t) in enumerate([((), main_type)] + extra_ins):
                if t in [np.float32, np.float64]:
                    d = rand_array(*(main_dims + dims))
                else:
                    d = np.random.randint(-100, 100, (main_dims + dims))
                d = d.astype(t)
                workspace.FeedBlob(ins[i + 1], d)
                x.append(d)

            # Randomly generate length tensor as well
            elements = np.random.randint(2, 10)
            lengths = []
            total_length = 0
            for _ in range(elements - 1):
                lengths.append(np.random.randint(main_dims[0] - total_length))
                total_length += lengths[-1]
            lengths.append(main_dims[0] - total_length)
            workspace.FeedBlob(ins[0], np.array(lengths, dtype=np.int32))

            def sharding(x):
                # numpy has proper modulo op that yields non-negative results
                shards = (x[0] % parts).reshape([-1])
                out = []
                for i in range(parts):
                    idx = 0
                    sharded_lengths = np.zeros(elements)
                    for ind, length in enumerate(lengths):
                        for _ in range(length):
                            if shards[idx] == i:
                                sharded_lengths[ind] += 1
                            idx += 1
                    out.append(sharded_lengths)

                    for ind, v in enumerate(x):
                        suffix_shape = v.shape[len(x[0].shape):]
                        accum = []
                        data = v.reshape((-1, ) + suffix_shape)

                        if pack and ind == 0:
                            data = data // parts

                        for j, s in enumerate(shards):
                            if s == i:
                                accum.append(data[j])

                        def join(a):
                            if not a:
                                return np.empty(shape=(0, ) + suffix_shape)
                            return np.stack(a)

                        out.append(join(accum))
                return out

            workspace.RunOperatorOnce(op)
            ref = sharding(x)
            for name, expected in zip(outs, ref):
                np.testing.assert_array_equal(
                    expected, workspace.FetchBlob(name)
                )

if __name__ == "__main__":
    import unittest
    unittest.main()
