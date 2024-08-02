import logging

import torch

from ._meta_parser import (
    OpenRegTensorData,
    receive_after_sending,
    safe_str,
    validate_send_queue_args,
)


log = logging.getLogger(__name__)
mp_context = torch.multiprocessing.get_context("spawn")

# Constant properties of our device
NUM_DEVICES = 7

# Global state of our driver
CURR_DEVICE_IDX = 0
CURR_STREAM = 0


# Our allocator
class Allocator:
    def __init__(self):
        self.allocated = {}

    def malloc(self, size):
        new_data = torch.empty(size, dtype=torch.uint8)
        ptr = new_data.data_ptr()
        self.allocated[ptr] = new_data
        return ptr

    def free(self, ptr):
        if ptr not in self.allocated:
            return False
        else:
            del self.allocated[ptr]
            return True

    def tensor_from_meta(self, meta):
        # Usual case, we're receiving a known Tensor
        found_base = self.allocated.get(meta.data_ptr, None)

        # Might be a rewrap of another storage at a different offset
        # Slow path to try and find the corresponding storage
        if found_base is None:
            for tag, t in self.allocated.items():
                # t is always a 1D uint8 storage!
                if meta.data_ptr > tag and meta.data_ptr < tag + t.nelement():
                    # Blame @ngimel for this
                    slice_size = t.nelement() - (meta.data_ptr - tag)
                    found_base = torch.tensor((), dtype=torch.uint8).set_(
                        t.untyped_storage()[meta.data_ptr - tag :],
                        size=(slice_size,),
                        stride=(1,),
                        storage_offset=0,
                    )

        # This pointer is not allocated here, segfault !
        if found_base is None:
            log.info("Currently allocated blocks:\n %s", safe_str(self.allocated))
            log.info("Trying to access %s", meta)
            raise RuntimeError("SEGFAULT!")

        # Raw 1d uint8 data
        raw = found_base
        # Slice the right storage part
        raw_slice = raw.narrow(0, 0, meta.nelem_in_bytes)
        # Reinterpret cast in the right dtype
        as_dtype = raw_slice.view(dtype=meta.dtype)
        # View to the right shape/stride/offset
        view = as_dtype.as_strided(meta.size, meta.stride, meta.storage_offset)
        return view


def run_op(allocator, op_name, args, kwargs):
    op, _ = torch._C._jit_get_operation(op_name)
    args, kwargs = receive_after_sending(allocator, args, kwargs)
    return op(*args, **kwargs)


class _Daemon:
    def __init__(self):
        super().__init__()
        self.is_initialized = False

    def _lazy_init(self):
        if self.is_initialized:
            return
        self.req_queue = mp_context.Queue()
        self.ans_queue = mp_context.Queue()

        self.runner = mp_context.Process(
            target=self.run_forever, args=(self.req_queue, self.ans_queue), daemon=True
        )
        self.runner.start()
        self.is_initialized = True

    def exec(self, cmd, *args):
        self._lazy_init()
        log.info("Main process launched: %s(*%s)", cmd, safe_str(args))
        validate_send_queue_args(cmd, args)
        self.req_queue.put((cmd,) + args)
        res = self.ans_queue.get()
        log.info("Main process result for %s received: %s", cmd, safe_str(res))
        if res == "ERROR":
            raise RuntimeError(f"Error in daemon while executing {cmd}, see logs")
        else:
            return res

    def __del__(self):
        print("DEL")

    @staticmethod
    def run_forever(req_queue, ans_queue):
        # Initialize our device
        global CURR_DEVICE_IDX
        empty_res = object()
        allocator = Allocator()

        # Serve all requests
        while True:
            cmd, *args = req_queue.get()
            log.info("Worker executing: %s", cmd)
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
            elif cmd == "exchangeDevice":
                assert len(args) == 1
                res = CURR_DEVICE_IDX
                CURR_DEVICE_IDX = int(args[0])
            elif cmd == "malloc":
                res = allocator.malloc(*args)
            elif cmd == "free":
                res = allocator.free(*args)
            elif cmd == "run_op":
                op_name, args, kwargs = args
                run_op(allocator, op_name, args, kwargs)
                res = None
            elif cmd == "send_data":
                assert len(args) == 1
                res = OpenRegTensorData.from_meta(allocator, args[0])
            elif cmd == "recv_data":
                assert len(args) == 2
                host_tensor, dev_mem = args
                dev_tensor = OpenRegTensorData.from_meta(allocator, dev_mem)
                dev_tensor.copy_(host_tensor)
                res = None
            elif cmd == "get_op_output_shape":
                op_name, args, kwargs = args
                res = run_op(allocator, op_name, args, kwargs).size()
            else:
                log.warning("Bad command in worker")
                res = "ERROR"

            if res == empty_res:
                raise RuntimeError("Bad impl didn't return anything")
            log.info("Worker answering to: %s", cmd)
            ans_queue.put(res)


daemon = _Daemon()
