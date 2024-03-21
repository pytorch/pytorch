import os
import time
import torch
import torch.multiprocessing as mp
from torch.distributed._state_dict_utils import _create_cpu_state_dict, _offload_state_dict_to_cpu

def rand_tensor():
    # return torch.rand(5000, 5000, device="cuda")
    return torch.rand(50, 50, device="cuda")

def test_shared_pinned_ipc(
    use_shared=True,
    use_pinned=True,
):

    print(f"{os.getpid()=}")

    state_dict = {"a": rand_tensor()}
    rand_event = torch.cuda.Event()
    rand_event.record()

    cache = _create_cpu_state_dict(
        state_dict,
        share_memory=use_shared,
        pin_memory=use_pinned
    )
    torch.cuda.synchronize()

    spawn_context = mp.get_context("spawn")
    send_queue = spawn_context.Queue()
    recv_queue = spawn_context.Queue()
    mp.spawn(fn=sub_process, args=(send_queue, recv_queue), nprocs=1, join=False)

    t0 = time.monotonic()
    stream = torch.cuda.Stream()
    offload_event = torch.cuda.Event(interprocess=True, blocking=True)
    for idx in range(100):

        with torch.cuda.stream(stream):
            rand_event.wait()
            _offload_state_dict_to_cpu(
                state_dict,
                cpu_offload_state_dict=cache,
                cpu_offload_sync=False
            )
            offload_event.record()

        send_queue.put((cache, offload_event))

        # we will synchronize before here, bc of child process
        t_recv = recv_queue.get()
        t_recv["a"] -= 1
        torch.testing.assert_close(state_dict["a"].cpu(), t_recv["a"])

        state_dict = {"a": rand_tensor()}
        rand_event.record()

    send_queue.put((None, None))
    print(f"{time.monotonic() - t0}")

def sub_process(_, recv_queue, send_queue):
    try:
        t, event = recv_queue.get()
        while t is not None:
            event.wait()
            t["a"] += 1
            send_queue.put(t)
            t, event = recv_queue.get()
    except Exception as e:
        print(e)
        raise


if __name__ == "__main__":
    test_shared_pinned_ipc()
