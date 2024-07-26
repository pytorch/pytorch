import torch

import multiprocessing
import logging

log = logging.getLogger(__name__)

# Constant properties of our device
NUM_DEVICES = 7

# Global state of our driver
CURR_DEVICE_IDX = 0

# Our allocator
class Allocator():
    def __init__(self):
        self.allocated = {}

    def malloc(self, size):
        new_data = torch.empty(size, dtype=torch.uint8)
        ptr = new_data.data_ptr()
        self.allocated[ptr] = new_data
        return ptr

    def free(self, ptr):
        if not ptr in self.allocated:
            return False
        else:
            del self.allocated[ptr]
            return True

class _Daemon():
    def __init__(self):
        super().__init__()
        self.req_queue = multiprocessing.Queue()
        self.ans_queue = multiprocessing.Queue()

        self.runner = multiprocessing.Process(target=self.run_forever, args=(self.req_queue, self.ans_queue), daemon=True)
        self.runner.start()

    def exec(self, cmd, *args):
        log.info(f"Main process launched: {cmd}{args}")
        self.req_queue.put((cmd,) + args)
        res = self.ans_queue.get()
        log.info(f"Main process result: {res}")
        if res == "ERROR":
            raise RuntimeError(f"Error in daemon while executing {cmd}, see logs")
        else:
            return res

    @staticmethod
    def run_forever(req_queue, ans_queue):
        # Initialize our device
        global CURR_DEVICE_IDX
        empty_res = object()
        allocator = Allocator()

        # Serve all requests
        while True:
            cmd, *args = req_queue.get()
            log.info(f"Worker executing: {cmd}")
            res = empty_res
            if cmd == "deviceCount":
                assert len(args) == 0
                res = NUM_DEVICES
            elif cmd == "getDevice":
                res = CURR_DEVICE_IDX
            elif cmd == "uncheckedSetDevice":
                assert len(args) == 1
                CURR_DEVICE_IDX = int(args[0])
                res = None
            elif cmd == "malloc":
                res = allocator.malloc(*args)
            elif cmd == "free":
                res = allocator.free(*args)
            else:
                log.warning("Bad command in worker")
                res = "ERROR"

            if res == empty_res:
                raise RuntimeError("Bad impl did return anything")
            ans_queue.put(res)

daemon = _Daemon()

