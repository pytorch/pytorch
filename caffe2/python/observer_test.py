




import numpy as np
import unittest
from hypothesis import given, settings
import hypothesis.strategies as st

from caffe2.python import brew, core, model_helper, rnn_cell
import caffe2.python.workspace as ws


class TestObservers(unittest.TestCase):
    def setUp(self):
        core.GlobalInit(["python", "caffe2"])
        ws.ResetWorkspace()
        self.model = model_helper.ModelHelper()
        brew.fc(self.model, "data", "y",
                    dim_in=4, dim_out=2,
                    weight_init=('ConstantFill', dict(value=1.0)),
                    bias_init=('ConstantFill', dict(value=0.0)),
                    axis=0)
        ws.FeedBlob("data", np.zeros([4], dtype='float32'))

        ws.RunNetOnce(self.model.param_init_net)
        ws.CreateNet(self.model.net)

    def testObserver(self):
        ob = self.model.net.AddObserver("TimeObserver")
        ws.RunNet(self.model.net)
        print(ob.average_time())
        num = self.model.net.NumObservers()
        self.model.net.RemoveObserver(ob)
        assert(self.model.net.NumObservers() + 1 == num)

    @given(
        num_layers=st.integers(1, 4),
        forward_only=st.booleans()
    )
    @settings(deadline=1000)
    def test_observer_rnn_executor(self, num_layers, forward_only):
        '''
        Test that the RNN executor produces same results as
        the non-executor (i.e running step nets as sequence of simple nets).
        '''

        Tseq = [2, 3, 4]
        batch_size = 10
        input_dim = 3
        hidden_dim = 3

        run_cnt = [0] * len(Tseq)
        avg_time = [0] * len(Tseq)
        for j in range(len(Tseq)):
            T = Tseq[j]

            ws.ResetWorkspace()
            ws.FeedBlob(
                "seq_lengths",
                np.array([T] * batch_size, dtype=np.int32)
            )
            ws.FeedBlob("target", np.random.rand(
                T, batch_size, hidden_dim).astype(np.float32))
            ws.FeedBlob("hidden_init", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))
            ws.FeedBlob("cell_init", np.zeros(
                [1, batch_size, hidden_dim], dtype=np.float32
            ))

            model = model_helper.ModelHelper(name="lstm")
            model.net.AddExternalInputs(["input"])

            init_blobs = []
            for i in range(num_layers):
                hidden_init, cell_init = model.net.AddExternalInputs(
                    "hidden_init_{}".format(i),
                    "cell_init_{}".format(i)
                )
                init_blobs.extend([hidden_init, cell_init])

            output, last_hidden, _, last_state = rnn_cell.LSTM(
                model=model,
                input_blob="input",
                seq_lengths="seq_lengths",
                initial_states=init_blobs,
                dim_in=input_dim,
                dim_out=[hidden_dim] * num_layers,
                drop_states=True,
                forward_only=forward_only,
                return_last_layer_only=True,
            )

            loss = model.AveragedLoss(
                model.SquaredL2Distance([output, "target"], "dist"),
                "loss"
            )
            # Add gradient ops
            if not forward_only:
                model.AddGradientOperators([loss])

            # init
            for init_blob in init_blobs:
                ws.FeedBlob(init_blob, np.zeros(
                    [1, batch_size, hidden_dim], dtype=np.float32
                ))
            ws.RunNetOnce(model.param_init_net)

            # Run with executor
            self.enable_rnn_executor(model.net, 1, forward_only)

            np.random.seed(10022015)
            input_shape = [T, batch_size, input_dim]
            ws.FeedBlob(
                "input",
                np.random.rand(*input_shape).astype(np.float32)
            )
            ws.FeedBlob(
                "target",
                np.random.rand(
                    T,
                    batch_size,
                    hidden_dim
                ).astype(np.float32)
            )
            ws.CreateNet(model.net, overwrite=True)

            time_ob = model.net.AddObserver("TimeObserver")
            run_cnt_ob = model.net.AddObserver("RunCountObserver")
            ws.RunNet(model.net)
            avg_time[j] = time_ob.average_time()
            run_cnt[j] = int(''.join(x for x in run_cnt_ob.debug_info() if x.isdigit()))
            model.net.RemoveObserver(time_ob)
            model.net.RemoveObserver(run_cnt_ob)

        print(avg_time)
        print(run_cnt)
        self.assertTrue(run_cnt[1] > run_cnt[0] and run_cnt[2] > run_cnt[1])
        self.assertEqual(run_cnt[1] - run_cnt[0], run_cnt[2] - run_cnt[1])

    def enable_rnn_executor(self, net, value, forward_only):
        num_found = 0
        for op in net.Proto().op:
            if op.type.startswith("RecurrentNetwork"):
                for arg in op.arg:
                    if arg.name == 'enable_rnn_executor':
                        arg.i = value
                        num_found += 1
        # This sanity check is so that if someone changes the
        # enable_rnn_executor parameter name, the test will
        # start failing as this function will become defective.
        self.assertEqual(1 if forward_only else 2, num_found)
