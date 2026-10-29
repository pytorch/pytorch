# One-sided tensor transports (experimental)

`torch.distributed._transport` moves registered tensor byte ranges between
independently managed workers. It does not require a process group, global rank
assignment, or matching receives.

An endpoint connects to one peer. Exchange `bind()` bytes and remote memory
descriptors through a trusted application control plane. Exchange descriptors only with authorized peers. A write copies from a local view to the remote base; a read
copies from the remote base to a writable local view. Offsets and lengths are bytes;
shape and dtype agreement is the application's responsibility.

Synchronous reads/writes return zero; `async_op=True` returns
{class}`torch.distributed.Work`. Successful completion means the transfer completed,
not that the remote application consumed or acknowledged the data. Asyncio callers
can use `read_async`, `write_async`, or `wait_all`. Registration remains valid
until unregistration or close, and tensors must not be resized or have their storage replaced.

Ordinary reads and writes are not ordered on CUDA streams. The API does not
provide tracing, batching, or remote slicing. Descriptor classes define explicit
`serialize()`/`deserialize()` methods.
The built-in backends declare their fields in a versioned JSON envelope; binary
metadata is base64-encoded. Unknown fields, versions, backends, and invalid field
types are rejected. Tensor contents and native handles are never serialized.

```{eval-rst}
.. autofunction:: torch.distributed._transport.new_transport
.. autofunction:: torch.distributed._transport.new_transport_rank
.. autoclass:: torch.distributed._transport.Transport
   :members:
.. autoclass:: torch.distributed._transport.Memory
   :members:
.. autoclass:: torch.distributed._transport.MemoryView
   :members:
.. autoclass:: torch.distributed._transport.MutableMemoryView
.. autoclass:: torch.distributed._transport.RemoteBuffer
.. autofunction:: torch.distributed._transport.wait_all
.. autofunction:: torch.distributed._transport.available_transports
.. autofunction:: torch.distributed._transport.register_transport
```

## CUDA streams

`read_stream` and `write_stream` order a transfer on `stream`, or the
current CUDA stream if `None`. The transfer starts after prior work on the
stream; later work waits for it to complete. Calls return after enqueueing and
no kernel runs while the transfer is in flight.

```python
with torch.cuda.stream(stream):
    source.copy_(producer)
    transport.write_stream(source_view, remote_destination)
    transport.read_stream(destination_view, remote_source)
    consume(destination)
```

Stream transfers can be captured with `torch.cuda.graph`; each replay submits
a new transfer. Register buffers and warm up transfers before capture. Captured
buffers stay registered until `close`; replaying after `close` terminates
the process.

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    source.copy_(producer)
    transport.write_stream(source_view, remote_destination)
graph.replay()
```

The default implementation awaits `write_async`/`read_async` on a shared
asyncio loop thread and gates the stream with a CUDA stream memory wait. It requires Linux
and GPUDirect RDMA write ordering, and allows 64 outstanding transfers per
stream; captured transfers hold their slot until `close`. Since consumers may
already be enqueued, a failed transfer terminates the process. Backends with
native stream support may override these methods.

## NIXL backend

The initial backend is NIXL, installed separately with `pip install nixl`;
`supported()` requires a build providing `nixlRegDList`, `DRAM_SEG`,
`nixlNotAllowedError` and `nixlNotFoundError`.
Its default UCX plugin supports CPU and CUDA memory, subject to the installed
NIXL/UCX build and hardware.

`unregister_memory(memory)` deregisters a local allocation without closing the
transport. Wait for local transfers first and coordinate with peers to stop remote
access; the backend cannot detect incoming DMA. All handles sharing a registration,
including previously created views and exported descriptors, become invalid.
Register again and exchange fresh descriptors before resuming transfers.

NIXL transfers return Work objects that retain their request handles until
completion. `wait_all`, `read_async`, and `write_async` await Work futures.
The NIXL adapter resolves those futures by checking native transfer status.
Each live transfer owns a distinct request handle.
Independent requests may overlap; explicitly wait before issuing dependent or
overlapping reads/writes. Completion ordering is not implicit.
`bind()` returns connection info only; each descriptor's metadata is loaded on
its first transfer. NIXL cannot unload single remote descriptors, so a transfer
resets the peer when its remote buffer overlaps a different one already loaded
from that peer, reuses a loaded range with new metadata, or exceeds 4096 loaded
buffers. A failed transfer, or a load refused with `nixlNotAllowedError` after
the peer restarts under the same `agent_name`, marks the peer so its next
transfer resets it. A reset first waits for the peer's pending transfers, so a
stuck transfer blocks the peer until `close`. The wait counts against `timeout`
and raises `TimeoutError`. With `async_op=True` it happens before the Work is
returned, and the Work's default wait is the time remaining, possibly zero.
After a peer restart, a retry reloads the peer once its old transfers finish.
Any other failed metadata load may leave the peer unusable and its pending
transfers unfinished, so it closes the transport: `connected()` returns False,
and `close` times out while those transfers remain, leaving the transport's
memory registered until process exit. Peers must run the same PyTorch version:
`connect` loads an older peer's descriptors untracked, which can make reloads
fail.

NIXL's default wait timeout is 30 seconds. `timeout` is in seconds; `None`
selects the backend default and zero polls without waiting. `Work.wait` instead
takes a `datetime.timedelta`; its zero default waits for the timeout remaining
when the transfer was submitted.
Timeout and asyncio cancellation stop waiting, not DMA. The transport retains
pending requests and buffers, even if the caller drops its Work. Wait again or
successfully close before reusing buffers. Coordinate with peers before closing
exposed memory; close only drains locally submitted operations.

`NIXLTransport.close_async` awaits pending transfers before native cleanup.
A timed-out or cancelled close rejects new work and retains resources; retry
close to finish cleanup. Registrations keep the transport alive even after its last outgoing transfer,
because peers may still access exposed memory. Forgotten registrations may retain
resources indefinitely:
call `unregister_memory` or `close` after coordinating with peers.
Checking `is_completed` also releases completed requests.
For pending NIXL work, `get_future` requires a running asyncio loop; the loop
must remain running to drive that future. Completed work needs no event loop.

Native metadata, registration, request submission, status checks, and cleanup
calls execute synchronously on the calling thread. Their timeouts bound lock
acquisition and transfer completion waits, not execution inside NIXL. Python
cannot interrupt a blocked native call, even when it releases the GIL.

```{eval-rst}
.. autoclass:: torch.distributed._transport.nixl.NIXLTransport
   :members: close_async
```
