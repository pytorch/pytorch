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


def register(registry):
    def func(fn):
        registry[fn.__name__] = fn
        return fn

    return func


class Driver:
    def __init__(self):
        super().__init__()
        self.is_initialized = False

    def _lazy_init(self):
        if self.is_initialized:
            return

        # State of our driver
        self.curr_device_idx = 0
        self.curr_stream = 0
        # Constant properties of our device
        self.num_devices = 7

        self.req_queue = mp_context.Queue()
        self.ans_queue = mp_context.Queue()

        self.runner = mp_context.Process(
            target=_Executor().run_forever,
            args=(self.req_queue, self.ans_queue),
            daemon=True,
        )
        self.runner.start()
        self.is_initialized = True

    def exec(self, cmd, *args):
        self._lazy_init()
        log.info("Main process launched: %s(*%s)", cmd, safe_str(args))

        if cmd in Driver.registry:
            res = Driver.registry[cmd](self, *args)
        else:
            validate_send_queue_args(cmd, args)
            self.req_queue.put((cmd,) + args)
            res = self.ans_queue.get()

        log.info("Main process result for %s received: %s", cmd, safe_str(res))
        if res == "ERROR":
            raise RuntimeError(f"Error in daemon while executing {cmd}, see logs")
        else:
            return res

    registry = {}

    @register(registry)
    def deviceCount(self, *args):
        assert len(args) == 0
        return self.num_devices

    @register(registry)
    def getDevice(self):
        return self.curr_device_idx

    @register(registry)
    def uncheckedSetDevice(self, *args):
        assert len(args) == 1
        self.curr_device_idx = int(args[0])

    @register(registry)
    def exchangeDevice(self, *args):
        assert len(args) == 1
        res = self.curr_device_idx
        self.curr_device_idx = int(args[0])
        return res


class _Executor:
    def __init__(self):
        self.allocator = Allocator()

    def run_forever(self, req_queue, ans_queue):
        # Serve all requests
        while True:
            cmd, *args = req_queue.get()
            log.info("Worker executing: %s", cmd)
            if cmd in _Executor.registry:
                res = _Executor.registry[cmd](self, *args)
            else:
                log.warning("Bad command in worker")
                res = "ERROR"

            log.info("Worker answering to: %s", cmd)
            ans_queue.put(res)

    registry = {}

    @register(registry)
    def malloc(self, size):
        return self.allocator.malloc(size)

    @register(registry)
    def free(self, ptr):
        return self.allocator.free(ptr)

    def _run_op(self, op_name, args, kwargs):
        op, _ = torch._C._jit_get_operation(op_name)
        args, kwargs = receive_after_sending(self.allocator, args, kwargs)
        return op(*args, **kwargs)

    @register(registry)
    def run_op(self, op_name, args, kwargs):
        self._run_op(op_name, args, kwargs)

    @register(registry)
    def get_op_output_shape(self, op_name, args, kwargs):
        return self._run_op(op_name, args, kwargs).size()

    @register(registry)
    def send_data(self, *args):
        assert len(args) == 1
        return OpenRegTensorData.from_meta(self.allocator, args[0])

    @register(registry)
    def recv_data(self, host_tensor, dev_mem):
        dev_tensor = OpenRegTensorData.from_meta(self.allocator, dev_mem)
        dev_tensor.copy_(host_tensor)


driver = Driver()
