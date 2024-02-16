




import unittest

from caffe2.python import workspace, core
import caffe2.python.parallel_workers as parallel_workers


def create_queue():
    queue = 'queue'

    workspace.RunOperatorOnce(
        core.CreateOperator(
            "CreateBlobsQueue", [], [queue], num_blobs=1, capacity=1000
        )
    )
    # Technically, blob creations aren't thread safe. Since the unittest below
    # does RunOperatorOnce instead of CreateNet+RunNet, we have to precreate
    # all blobs beforehand
    for i in range(100):
        workspace.C.Workspace.current.create_blob("blob_" + str(i))
        workspace.C.Workspace.current.create_blob("status_blob_" + str(i))
    workspace.C.Workspace.current.create_blob("dequeue_blob")
    workspace.C.Workspace.current.create_blob("status_blob")

    return queue


def create_worker(queue, get_blob_data):
    def dummy_worker(worker_id):
        blob = 'blob_' + str(worker_id)

        workspace.FeedBlob(blob, get_blob_data(worker_id))

        workspace.RunOperatorOnce(
            core.CreateOperator(
                'SafeEnqueueBlobs', [queue, blob], [blob, 'status_blob_' + str(worker_id)]
            )
        )

    return dummy_worker


def dequeue_value(queue):
    dequeue_blob = 'dequeue_blob'
    workspace.RunOperatorOnce(
        core.CreateOperator(
            "SafeDequeueBlobs", [queue], [dequeue_blob, 'status_blob']
        )
    )

    return workspace.FetchBlob(dequeue_blob)


class ParallelWorkersTest(unittest.TestCase):
    def testParallelWorkers(self):
        workspace.ResetWorkspace()

        queue = create_queue()
        dummy_worker = create_worker(queue, str)
        worker_coordinator = parallel_workers.init_workers(dummy_worker)
        worker_coordinator.start()

        for _ in range(10):
            value = dequeue_value(queue)
            self.assertTrue(
                value in [b'0', b'1'], 'Got unexpected value ' + str(value)
            )

        self.assertTrue(worker_coordinator.stop())

    def testParallelWorkersInitFun(self):
        workspace.ResetWorkspace()

        queue = create_queue()
        dummy_worker = create_worker(
            queue, lambda worker_id: workspace.FetchBlob('data')
        )
        workspace.FeedBlob('data', 'not initialized')

        def init_fun(worker_coordinator, global_coordinator):
            workspace.FeedBlob('data', 'initialized')

        worker_coordinator = parallel_workers.init_workers(
            dummy_worker, init_fun=init_fun
        )
        worker_coordinator.start()

        for _ in range(10):
            value = dequeue_value(queue)
            self.assertEqual(
                value, b'initialized', 'Got unexpected value ' + str(value)
            )

        # A best effort attempt at a clean shutdown
        worker_coordinator.stop()

    def testParallelWorkersShutdownFun(self):
        workspace.ResetWorkspace()

        queue = create_queue()
        dummy_worker = create_worker(queue, str)
        workspace.FeedBlob('data', 'not shutdown')

        def shutdown_fun():
            workspace.FeedBlob('data', 'shutdown')

        worker_coordinator = parallel_workers.init_workers(
            dummy_worker, shutdown_fun=shutdown_fun
        )
        worker_coordinator.start()

        self.assertTrue(worker_coordinator.stop())

        data = workspace.FetchBlob('data')
        self.assertEqual(data, b'shutdown', 'Got unexpected value ' + str(data))
