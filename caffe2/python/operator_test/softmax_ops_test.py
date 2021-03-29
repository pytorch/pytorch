




from caffe2.python import core, workspace
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

import unittest


class TestSoftmaxOps(serial.SerializedTestCase):

    @serial.given(n=st.sampled_from([0, 2, 4, 71, 103]),
                  D=st.sampled_from([0, 4, 8, 64, 79, 256, 333]),
                  engine=st.sampled_from([None, 'CUDNN']),
                  **hu.gcs)
    def test_softmax(self, n, D, engine, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        X = np.random.rand(n, D).astype(np.float32)
        X = X + 1e-2

        # Reference implementation of cross entropy with soft labels
        def label_softmax(X):
            probs = np.zeros((n, D))
            rowmax = np.zeros(n)

            if D == 0:
                return [probs]

            for i in range(n):
                rowmax[i] = max(X[i, ])
                # We need to subtract the max to avoid numerical issues
                probs[i] = X[i] - rowmax[i]
                exps = np.exp(probs[i, ])
                norm = sum(exps)
                probs[i, ] = exps / norm

            return [probs]

        op = core.CreateOperator(
            "Softmax",
            ["X"],
            ["probs"],
            engine=engine
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=label_softmax,
        )

    @given(n=st.sampled_from([0, 2, 4, 71, 103, 555, 751, 1201]),
                  D=st.sampled_from([0, 4, 8, 64, 79, 256, 333, 1000]),
                  engine=st.sampled_from([None, 'CUDNN']),
                  **hu.gcs)
    @settings(deadline=10000)
    def test_softmax_grad(self, n, D, engine, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        Y = np.random.rand(n, D).astype(np.float32)
        dY = np.random.rand(n, D).astype(np.float32)
        Y = Y + 1e-2

        # Reference implementation of cross entropy with soft labels
        def label_softmax_grad(X, dY):
            dX = Y * 0.0
            for i in range(n):
                d = np.dot(Y[i, :], dY[i, :])
                dX[i, :] = Y[i, :] * (dY[i, :] - d)
            return [dX]

        op = core.CreateOperator(
            "SoftmaxGradient",
            ["Y", "dY"],
            ["dX"],
            engine=engine
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[Y, dY],
            reference=label_softmax_grad,
        )

    @given(axis=st.integers(min_value=1, max_value=4),
           engine=st.sampled_from([None, 'CUDNN']),
           **hu.gcs)
    def test_softmax_axis(self, axis, engine, gc, dc):
        np.random.seed(1)
        X = np.random.randn(1, 2, 3, 2, 1).astype(np.float32)
        X = X + 1e-2

        def prod(xs):
            p = 1
            for x in xs:
                p *= x
            return p

        N = prod(list(X.shape)[:axis])
        D = prod(list(X.shape)[axis:])

        # Reference implementation of cross entropy with soft labels
        def label_softmax(X):
            X_ = X.reshape(N, D)
            probs = np.zeros((N, D))
            rowmax = np.zeros(N)
            for i in range(N):
                rowmax[i] = max(X_[i, ])
                # We need to subtract the max to avoid numerical issues
                probs[i] = X_[i] - rowmax[i]
                exps = np.exp(probs[i, ])
                norm = sum(exps)
                probs[i, ] = exps / norm

            return [probs.reshape(*X.shape)]

        op = core.CreateOperator(
            "Softmax",
            ["X"],
            ["probs"],
            axis=axis,
            engine=engine,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X],
            reference=label_softmax,
        )

        self.assertGradientChecks(
            gc, op, [X], 0, [0], stepsize=1e-4, threshold=1e-2)

    @given(n=st.integers(2, 10), D=st.integers(4, 16),
           only_loss=st.booleans(), **hu.gcs)
    @settings(deadline=1000)
    def test_softmax_with_loss(self, n, D, gc, only_loss, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        np.random.seed(2603)
        X = np.random.rand(n, D).astype(np.float32)
        X = X + 1e-2

        # Initialize label
        label = (np.random.rand(n) * D).astype(np.int32)

        # Reference implementation of cross entropy with soft labels
        def label_softmax_crossent(X, label):
            probs = np.zeros((n, D))
            rowmax = np.zeros(n)
            for i in range(n):
                rowmax[i] = max(X[i, ])
                # We need to subtract the max to avoid numerical issues
                probs[i] = X[i] - rowmax[i]
                exps = np.exp(probs[i, ])
                norm = sum(exps)
                probs[i, ] = exps / norm

            label_xent = [-np.log(max(probs[i][label[i]], 1e-20))
                          for i in range(n)]
            avgloss = np.sum(label_xent) / float(n)
            return (probs, avgloss)

        op = core.CreateOperator(
            "SoftmaxWithLoss",
            ["X", "label"],
            ["probs", "avgloss"],
            only_loss=only_loss,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, label],
            reference=label_softmax_crossent,
        )

        self.assertGradientChecks(
            gc, op, [X, label], 0, [1], stepsize=1e-4, threshold=1e-2)

    @given(
        n=st.integers(2, 5),
        D=st.integers(4, 16),
        only_loss=st.booleans(),
        label_prob=st.booleans(),
        **hu.gcs
    )
    @settings(deadline=10000)
    def test_softmax_with_loss_axis_2(
        self, n, D, only_loss, label_prob,
        gc, dc
    ):
        np.random.seed(2603)
        X = np.random.rand(n, n, D).astype(np.float32)
        X = X + 1e-2

        if label_prob:
            label = np.random.rand(n, n, D).astype(np.float32)
            label /= label.sum(axis=2, keepdims=True)
        else:
            label = (np.random.rand(n, n) * D).astype(np.int32)

        # Reference implementation of cross entropy with soft labels
        def label_softmax_crossent(X, label):
            probs = np.zeros((n, n, D))
            rowmax = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    rowmax[i, j] = max(X[i, j, ])
                    # We need to subtract the max to avoid numerical issues
                    probs[i, j] = X[i, j] - rowmax[i, j]
                    exps = np.exp(probs[i, j, ])
                    norm = sum(exps)
                    probs[i, j, ] = exps / norm
            label_xent = 0
            for i in range(n):
                for j in range(n):
                    if label_prob:
                        for k in range(D):
                            label_xent += (
                                -np.log(max(probs[i, j, k], 1e-20)) *
                                label[i, j, k]
                            )
                    else:
                        label_xent += -np.log(max(probs[i, j, label[i, j]], 1e-20))

            avgloss = label_xent / float(n * n)
            return (probs, avgloss)

        op = core.CreateOperator(
            "SoftmaxWithLoss",
            ["X", "label"],
            ["probs", "avgloss"],
            only_loss=only_loss,
            label_prob=label_prob,
            axis=2,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, label],
            reference=label_softmax_crossent,
        )

        self.assertGradientChecks(
            gc, op, [X, label], 0, [1], stepsize=1e-4, threshold=1e-2)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    @given(**hu.gcs_gpu_only)
    def test_softmax_with_loss_large(self, gc, dc):
        np.random.seed(2603)
        for n in [32]:
            for D in [1000, 2000, 20000]:
                # n = number of examples, D = |labels|
                # Initialize X and add 1e-2 for numerical stability
                X = np.random.rand(n, D).astype(np.float32)
                X = X + 1e-2

                # Initialize label
                label = (np.random.rand(n) * D).astype(np.int32)

                # Reference implementation of cross entropy with soft labels
                def label_softmax_crossent(X, label):
                    probs = np.zeros((n, D))
                    rowmax = np.zeros(n)
                    for i in range(n):
                        rowmax[i] = max(X[i, ])
                        # We need to subtract the max to avoid numerical issues
                        probs[i] = X[i] - rowmax[i]
                        exps = np.exp(probs[i, ])
                        norm = sum(exps)
                        probs[i, ] = exps / norm

                    label_xent = [-np.log(max(probs[i][label[i]], 1e-20))
                                  for i in range(n)]
                    avgloss = np.sum(label_xent) / float(n)
                    return (probs, avgloss)

                op = core.CreateOperator(
                    "SoftmaxWithLoss",
                    ["X", "label"],
                    ["probs", "avgloss"]
                )

                self.assertReferenceChecks(
                    device_option=gc,
                    op=op,
                    inputs=[X, label],
                    reference=label_softmax_crossent,
                )

    @given(n=st.integers(2, 10), D=st.integers(4, 16), **hu.gcs)
    @settings(deadline=1000)
    def test_softmax_with_loss_label_prob(self, n, D, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        np.random.seed(2603)
        X = np.random.rand(n, D).astype(np.float32)
        X = X + 1e-2

        # Initialize label
        label = np.random.rand(D, n).astype(np.float32)

        # normalize labels to sum to 1
        label /= np.sum(label, axis=0)
        label = label.transpose()

        # Reference implementation of cross entropy with soft labels
        def label_softmax_crossent(X, label):
            probs = np.zeros((n, D))
            rowmax = np.zeros(n)
            for i in range(n):
                rowmax[i] = max(X[i, ])
                # We need to subtract the max to avoid numerical issues
                probs[i] = X[i] - rowmax[i]
                exps = np.exp(probs[i, ])
                norm = sum(exps)
                probs[i, ] = exps / norm

            label_xent = np.zeros(X.shape)
            for i in range(n):
                for j in range(D):
                    label_xent[i][j] = -np.log(
                        max(probs[i, j], 1e-20)) * label[i, j]
            avgloss = np.sum(label_xent) / float(n)
            return (probs, avgloss)

        op = core.CreateOperator(
            "SoftmaxWithLoss",
            ["X", "label"],
            ["probs", "avgloss"],
            label_prob=1
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, label],
            reference=label_softmax_crossent,
        )

        self.assertGradientChecks(
            gc, op, [X, label], 0, [1], stepsize=1e-4, threshold=1e-2)

    @given(
        n=st.integers(2, 10),
        D=st.integers(4, 16),
        only_loss=st.booleans(),
        **hu.gcs)
    @settings(deadline=1000)
    def test_softmax_with_loss_weighted(self, n, D, only_loss, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        np.random.seed(2603)
        X = np.random.rand(n, D).astype(np.float32)
        X = X + 1e-2

        # Initialize label
        label = (np.random.rand(n) * D).astype(np.int32)

        # Init weights (weight by sample)
        weights = np.random.rand(n).astype(np.float32)

        # Reference implementation of cross entropy with soft labels
        def label_softmax_crossent_weighted(X, label, weights):
            probs = np.zeros((n, D))
            rowmax = np.zeros(n)
            for i in range(n):
                rowmax[i] = max(X[i, ])
                # We need to subtract the max to avoid numerical issues
                probs[i] = X[i] - rowmax[i]
                exps = np.exp(probs[i, ])
                norm = sum(exps)
                probs[i, ] = exps / norm

            label_xent = [-weights[i] * np.log(max(probs[i][label[i]], 1e-20))
                          for i in range(n)]
            avgloss = np.sum(label_xent) / sum(weights)
            return (probs, avgloss)

        op = core.CreateOperator(
            "SoftmaxWithLoss",
            ["X", "label", "weights"],
            ["probs", "avgloss"],
            only_loss=only_loss,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, label, weights],
            reference=label_softmax_crossent_weighted,
        )

        self.assertGradientChecks(
            gc, op, [X, label, weights], 0, [1], stepsize=1e-4, threshold=1e-2)

    @given(n=st.integers(2, 10), D=st.integers(4, 16), **hu.gcs)
    @settings(deadline=1000)
    def test_softmax_with_loss_label_prob_weighted(self, n, D, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        X = np.random.rand(n, D).astype(np.float32)
        X = X + 1e-2

        # Initialize label
        label = np.random.rand(D, n).astype(np.float32)

        # normalize labels to sum to 1
        label /= np.sum(label, axis=0)
        label = label.transpose()

        # Init weights (weight by sample)
        weights = np.random.rand(n).astype(np.float32)

        # Reference implementation of cross entropy with soft labels
        def label_softmax_crossent_weighted(X, label, weights):
            probs = np.zeros((n, D))
            rowmax = np.zeros(n)
            for i in range(n):
                rowmax[i] = max(X[i, ])
                # We need to subtract the max to avoid numerical issues
                probs[i] = X[i] - rowmax[i]
                exps = np.exp(probs[i, ])
                norm = sum(exps)
                probs[i, ] = exps / norm

            label_xent = np.zeros(X.shape)
            for i in range(n):
                for j in range(D):
                    label_xent[i][j] = -np.log(
                        max(probs[i, j], 1e-20)) * label[i, j] * weights[i]
            avgloss = np.sum(label_xent) / sum(weights)
            return (probs, avgloss)

        op = core.CreateOperator(
            "SoftmaxWithLoss",
            ["X", "label", "weights"],
            ["probs", "avgloss"],
            label_prob=1,
        )

        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=[X, label, weights],
            reference=label_softmax_crossent_weighted,
        )

        self.assertGradientChecks(
            gc, op, [X, label, weights], 0, [1], stepsize=1e-4, threshold=1e-2)

    @given(n=st.integers(2, 5), D=st.integers(2, 4),
           weighted=st.booleans(), **hu.gcs)
    @settings(deadline=None, max_examples=50)
    def test_spatial_softmax_with_loss(self, n, D, weighted, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        W = 18
        H = 12
        np.random.seed(2603)
        X = np.random.rand(n, D, H, W).astype(np.float32)
        X = X + 1e-2

        weighted = True
        weights = None
        if weighted:
            weights = np.random.rand(n, H, W).astype(np.float32)

        # Initialize label. Some of the labels are (-1), i.e "DONT CARE"
        label = (np.random.rand(n, H, W) * (D + 1)).astype(np.int32) - 1

        def label_softmax_crossent_spatial(X, label, weights=None):
            probs = np.zeros((n, D, H, W))
            rowmax = np.zeros((n, H, W))
            label_xent = np.zeros((n, H, W))
            for i in range(n):
                for x in range(W):
                    for y in range(H):
                        rowmax[i, y, x] = max(X[i, :, y, x])
                        # We need to subtract the max to avoid numerical issues
                        probs[i, :, y, x] = X[i, :, y, x] - rowmax[i, y, x]
                        exps = np.exp(probs[i, :, y, x])
                        probs[i, :, y, x] = exps / sum(exps)

                        label_xent[:, y, x] = \
                            [-np.log(max(probs[j, label[i, y, x], y, x], 1e-20))
                             for j in range(n)]

            total_xent = 0.0
            total_weight = 0.0
            for y in range(H):
                for x in range(W):
                    for i in range(n):
                        l = label[i, y, x]
                        if (l != (-1)):
                            w = 1.0 if weights is None else weights[i, y, x]
                            total_xent += \
                                -np.log(max(probs[i, l, y, x], 1e-20)) * w
                            total_weight += w
            print("Total weight {}".format(total_weight))

            return (probs, total_xent / total_weight)

        op = core.CreateOperator(
            "SpatialSoftmaxWithLoss",
            ["X", "label"] + ([] if weights is None else ["weights"]),
            ["probs", "avgloss"],
        )

        inputs = [X, label] + ([] if weights is None else [weights])
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=label_softmax_crossent_spatial,
        )

        self.assertGradientChecks(
            gc, op, inputs, 0, [1], stepsize=1e-4, threshold=1e-2)

    @given(n=st.integers(4, 5), D=st.integers(3, 4),
           weighted=st.booleans(), **hu.gcs)
    def test_spatial_softmax_with_loss_allignore(self, n, D, weighted, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        W = 18
        H = 12
        np.random.seed(2603)
        X = np.random.rand(n, D, H, W).astype(np.float32)
        X = X + 1e-2

        weighted = True
        weights = None
        if weighted:
            weights = np.random.rand(n, H, W).astype(np.float32)

        # Initialize label. All labels as "DONT CARE"
        label = np.zeros((n, H, W)).astype(np.int32) - 1
        print(label)

        def label_softmax_crossent_spatial(X, label, weights=None):
            probs = np.zeros((n, D, H, W))
            rowmax = np.zeros((n, H, W))
            label_xent = np.zeros((n, H, W))
            for i in range(n):
                for x in range(W):
                    for y in range(H):
                        rowmax[i, y, x] = max(X[i, :, y, x])
                        # We need to subtract the max to avoid numerical issues
                        probs[i, :, y, x] = X[i, :, y, x] - rowmax[i, y, x]
                        exps = np.exp(probs[i, :, y, x])
                        probs[i, :, y, x] = exps / sum(exps)

                        label_xent[:, y, x] = \
                            [-np.log(max(probs[j, label[i, y, x], y, x], 1e-20))
                            for j in range(n)]

            return (probs, 0.0)

        op = core.CreateOperator(
            "SpatialSoftmaxWithLoss",
            ["X", "label"] + ([] if weights is None else ["weights"]),
            ["probs", "avgloss"],
        )

        inputs = [X, label] + ([] if weights is None else [weights])
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=label_softmax_crossent_spatial,
        )

    @given(n=st.integers(4, 5), D=st.integers(3, 4),
           weighted=st.booleans(), **hu.gcs)
    def test_softmax_with_loss_zero_weight(self, n, D, weighted, gc, dc):
        # n = number of examples, D = |labels|
        # Initialize X and add 1e-2 for numerical stability
        np.random.seed(2603)
        X = np.random.rand(n, D).astype(np.float32)
        X = X + 1e-2

        weights = np.zeros(n).astype(np.float32)

        # Initialize label
        label = (np.random.rand(n) * D).astype(np.int32)

        def label_softmax_crossent(X, label, weights=None):
            probs = np.zeros((n, D))
            rowmax = np.zeros((n))
            for i in range(n):
                rowmax[i] = max(X[i, ])
                # We need to subtract the max to avoid numerical issues
                probs[i] = X[i] - rowmax[i]
                exps = np.exp(probs[i, ])
                norm = sum(exps)
                probs[i, ] = exps / norm
            return (probs, 0.0)

        op = core.CreateOperator(
            "SoftmaxWithLoss",
            ["X", "label", "weights"],
            ["probs", "avgloss"]
        )

        inputs = [X, label] + ([] if weights is None else [weights])
        self.assertReferenceChecks(
            device_option=gc,
            op=op,
            inputs=inputs,
            reference=label_softmax_crossent,
        )

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    def test_compare_cpugpu(self):
        '''
        Additional test that checks CPU and GPU returns same values
        with larger examples. This is mainly to test the more complex
        GPU implementation is correct.
        '''
        from caffe2.proto import caffe2_pb2

        for _j in range(3):
            gpuop = core.CreateOperator(
                "SpatialSoftmaxWithLoss",
                ["X_gpu", "label_gpu"],
                ["probs_gpu", "avgloss_gpu"],
                device_option=core.DeviceOption(workspace.GpuDeviceType, 0)
            )

            cpuop = core.CreateOperator(
                "SpatialSoftmaxWithLoss",
                ["X_cpu", "label_cpu"],
                ["probs_cpu", "avgloss_cpu"],
                device_option=core.DeviceOption(caffe2_pb2.CPU)
            )

            n = 8
            D = 4
            W = 64 + int(np.random.rand(1) * 1024)
            H = 64 + int(np.random.rand(1) * 1024)

            print("W: {} H: {}".format(W, H))

            X = np.random.rand(n, D, H, W).astype(np.float32)
            X = X + 1e-2

            # Initialize label. Some of the labels are (-1), i.e "DONT CARE"
            label = (np.random.rand(n, H, W) * (D + 1)).astype(np.int32) - 1

            gpu0 = core.DeviceOption(workspace.GpuDeviceType, 0)
            workspace.FeedBlob("X_cpu", X)
            workspace.FeedBlob("label_cpu", label)
            workspace.FeedBlob("X_gpu", X, device_option=gpu0)
            workspace.FeedBlob("label_gpu", label, device_option=gpu0)

            workspace.RunOperatorOnce(gpuop)
            workspace.RunOperatorOnce(cpuop)

            probs_gpu = workspace.FetchBlob("probs_gpu")
            probs_cpu = workspace.FetchBlob("probs_cpu")
            loss_gpu = workspace.FetchBlob("avgloss_gpu")
            loss_cpu = workspace.FetchBlob("avgloss_cpu")

            np.testing.assert_allclose(probs_gpu, probs_cpu, rtol=1e-4)
            np.testing.assert_allclose(loss_gpu, loss_cpu, rtol=1e-1)

if __name__ == "__main__":
    import unittest
    import random
    random.seed(2603)
    unittest.main()
