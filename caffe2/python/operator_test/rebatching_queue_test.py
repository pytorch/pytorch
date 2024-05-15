



from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase

import numpy as np
import numpy.testing as npt

from hypothesis import given, settings
import hypothesis.strategies as st

import functools


def primefac(n):
    ret = []
    divisor = 2
    while divisor * divisor <= n:
        while (n % divisor) == 0:
            ret.append(divisor)
            n = n // divisor
        divisor = divisor + 1
    if n > 1:
        ret.append(n)
    return ret


class TestReBatchingQueue(TestCase):
    def test_rebatching_queue_single_enqueue_dequeue(self):
        net = core.Net('net')

        tensors = [
            net.ConstantFill([], 1, value=1.0, run_once=False)
            for times in range(3)
        ]

        queue = net.CreateRebatchingQueue([], 1, capacity=10, num_blobs=1)

        net.EnqueueRebatchingQueue([queue, tensors[0]], [])
        net.EnqueueRebatchingQueue([queue, tensors[1]], [])
        net.EnqueueRebatchingQueue([queue, tensors[2]], [])

        results = [
            net.DequeueRebatchingQueue([queue], 1),
            net.DequeueRebatchingQueue([queue], 1),
            net.DequeueRebatchingQueue([queue], 1),
        ]

        workspace.RunNetOnce(net)

        for idx in range(3):
            self.assertEqual(workspace.FetchBlob(results[idx]), [1.0])

    def test_rebatching_queue_multi_enqueue_dequeue(self):
        net = core.Net('net')
        workspace.FeedBlob(
            "tensors", np.array([x for x in range(10)], np.int32)
        )

        queue = net.CreateRebatchingQueue([], 1, capacity=10, num_blobs=1)

        net.EnqueueRebatchingQueue([queue, "tensors"], [], enqueue_batch=True)

        results = [
            net.DequeueRebatchingQueue([queue], 1, num_elements=5),
            net.DequeueRebatchingQueue([queue], 1, num_elements=5),
        ]

        workspace.RunNetOnce(net)

        npt.assert_array_equal(
            workspace.FetchBlob(results[0]), workspace.FetchBlob("tensors")[:5]
        )
        npt.assert_array_equal(
            workspace.FetchBlob(results[1]), workspace.FetchBlob("tensors")[5:]
        )

    def test_rebatching_queue_closes_properly(self):
        net = core.Net('net')
        workspace.FeedBlob(
            "tensors", np.array([x for x in range(10)], np.int32)
        )

        queue = net.CreateRebatchingQueue([], 1, capacity=10, num_blobs=1)

        net.EnqueueRebatchingQueue([queue, "tensors"], 0, enqueue_batch=True)

        net.CloseRebatchingQueue([queue], 0)

        results = [
            net.DequeueRebatchingQueue([queue], 1, num_elements=5),
            net.DequeueRebatchingQueue([queue], 1, num_elements=5),
        ]

        workspace.RunNetOnce(net)

        npt.assert_array_equal(
            workspace.FetchBlob(results[0]), workspace.FetchBlob("tensors")[:5]
        )
        npt.assert_array_equal(
            workspace.FetchBlob(results[1]), workspace.FetchBlob("tensors")[5:]
        )

        # Enqueuing more should fail now since the queue is closed
        net.EnqueueRebatchingQueue([queue, "tensors"], [], enqueue_batch=True)

        with self.assertRaises(RuntimeError):
            workspace.RunNetOnce(net)

        # Dequeuing more should fail now since the queue is closed
        results = [
            net.DequeueRebatchingQueue([queue], 1, num_elements=5),
        ]

        with self.assertRaises(RuntimeError):
            workspace.RunNetOnce(net)

    def test_rebatching_queue_multiple_components(self):
        NUM_BLOBS = 4
        NUM_ELEMENTS = 10

        net = core.Net('net')

        workspace.blobs['complex_tensor'] = np.array(
            [[x, x + 1] for x in range(NUM_ELEMENTS)], dtype=np.int32
        )

        tensors = [
            net.GivenTensorIntFill(
                [],
                1,
                shape=[NUM_ELEMENTS],
                values=[x for x in range(NUM_ELEMENTS)]
            ),
            net.GivenTensorFill(
                [],
                1,
                shape=[NUM_ELEMENTS],
                values=[x * 1.0 for x in range(NUM_ELEMENTS)]
            ),
            net.GivenTensorBoolFill(
                [],
                1,
                shape=[NUM_ELEMENTS],
                values=[(x % 2 == 0) for x in range(NUM_ELEMENTS)]
            ),
            'complex_tensor',
        ]

        queue = net.CreateRebatchingQueue(
            [], 1, capacity=10, num_blobs=NUM_BLOBS
        )

        net.EnqueueRebatchingQueue([queue] + tensors, [], enqueue_batch=True)

        results = net.DequeueRebatchingQueue([queue], NUM_BLOBS, num_elements=5)

        workspace.RunNetOnce(net)

        for idx in range(NUM_BLOBS):
            npt.assert_array_equal(
                workspace.FetchBlob(results[idx]),
                workspace.FetchBlob(tensors[idx])[:5]
            )

    @given(
        num_producers=st.integers(1, 5),
        num_consumers=st.integers(1, 5),
        producer_input_size=st.integers(1, 10),
        producer_num_iterations=st.integers(1, 10),
        capacity=st.integers(1, 10)
    )
    @settings(deadline=10000)
    def test_rebatching_parallel_producer_consumer(
        self, num_producers, num_consumers, producer_input_size,
        producer_num_iterations, capacity
    ):
        ### Init ###
        total_inputs = producer_num_iterations * producer_input_size * num_producers
        inputs = []
        init_net = core.Net('init_net')
        queue = init_net.CreateRebatchingQueue(
            [], 1, capacity=capacity, num_blobs=1
        )

        ### Producers ###
        producer_steps = []
        for i in range(num_producers):
            name = 'producer_%d' % i
            net = core.Net(name)
            values = [
                producer_input_size * i + x for x in range(producer_input_size)
            ]
            for _ in range(producer_num_iterations):
                inputs.extend(values)
            tensors = net.GivenTensorIntFill(
                [], 1, shape=[producer_input_size], values=values
            )

            net.EnqueueRebatchingQueue([queue, tensors], [], enqueue_batch=True)

            step = core.execution_step(
                name, net, num_iter=producer_num_iterations
            )
            producer_steps.append(step)

        producer_step = core.execution_step(
            'producer', [
                core.execution_step(
                    'producers', producer_steps, concurrent_substeps=True
                )
            ]
        )

        ### Consumers ###
        outputs = []

        def append(ins, outs):
            # Extend is atomic
            outputs.extend(ins[0].data.tolist())

        consumer_steps = []
        for i in range(num_consumers):
            # This is just a way of deterministally read all the elements.
            # We make `num_consumers` almost equal splits
            # (the reminder goes to the last consumer).
            num_elements_to_read = total_inputs // num_consumers
            if i == num_consumers - 1:
                num_elements_to_read = num_elements_to_read \
                    + total_inputs % num_consumers

            # If we have nothing to read this consumer will be idle
            if (num_elements_to_read == 0):
                continue

            # Now we have to make a split on number of iterations and the read
            # size for each iteration. This is again just one of many
            # deterministic  ways of doing it. We factorize the total number of
            # elements we have to read and assign half of the factors to the
            # iterations half to the read size.
            factors = list(primefac(num_elements_to_read))

            num_elements_per_iteration = functools.reduce(
                lambda x, y: x * y, factors[len(factors) // 2:], 1
            )

            num_iterations = functools.reduce(
                lambda x, y: x * y, factors[:len(factors) // 2], 1
            )

            name = 'consumer_%d' % i
            net = core.Net(name)
            blobs = net.DequeueRebatchingQueue(
                [queue], 1, num_elements=num_elements_per_iteration
            )
            net.Python(append)([blobs], 0)
            consumer_steps.append(
                core.execution_step(name, net, num_iter=num_iterations)
            )

        consumer_step = core.execution_step(
            'consumer', consumer_steps, concurrent_substeps=True
        )

        init_step = core.execution_step('init', init_net)
        worker_step = core.execution_step(
            'worker', [consumer_step, producer_step], concurrent_substeps=True
        )

        ### Execute Plan ###
        plan = core.Plan('test')
        plan.AddStep(init_step)
        plan.AddStep(worker_step)

        self.ws.run(plan)

        ### Check Results ###
        # We check that the outputs are a permutation of inputs
        inputs.sort()
        outputs.sort()
        self.assertEqual(inputs, outputs)


if __name__ == "__main__":
    import unittest
    unittest.main()
