# One-sided tensor transports (experimental)

`torch.distributed._transport` moves registered tensor byte ranges between
independently managed workers. It does not require a process group, global rank
assignment, or matching receives.

An endpoint may connect to multiple peers; remote descriptors identify their
peer. Exchange `bind()` bytes and remote memory descriptors through a trusted
application control plane. Exchange descriptors only with authorized peers. A write copies from a local view to the remote base; a read
copies from the remote base to a writable local view. Offsets and lengths are bytes;
shape and dtype agreement is the application's responsibility.

Synchronous reads/writes return zero; `async_op=True` returns
{class}`torch.distributed.Work`. Successful completion means the transfer completed,
not that the remote application consumed or acknowledged the data. Asyncio callers
can use `read_async`, `write_async`, or `wait_all`. Registration remains valid
until unregistration or close, and tensors must not be resized or have their storage replaced.

Ordinary reads and writes are not ordered on CUDA streams. The API does not
provide tracing or batching. Views and remote buffers support `slice(offset, length)`
and step-1 `[a:b]` byte ranges. Like Python sequences, `[a:b]` counts negative
bounds from the end and clamps out-of-range ones, while `slice` raises; a
transfer of a clamped view copies fewer bytes. A remote slice does not
restrict the peer's access to the rest of the registration. Descriptor classes define explicit
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
   :members:
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

The initial backend is NIXL, installed separately with `pip install nixl`.
Its default UCX plugin supports CPU and CUDA memory, subject to the installed
NIXL/UCX build and hardware.

Peers are identified by NIXL agent name. Connecting again with the same
`bind()` bytes is a no-op. While NIXL holds a peer, connecting a restarted peer
under the same agent name raises the backend error. A descriptor whose
well-formed metadata names a different connected peer is rejected without
unloading that peer. Malformed metadata naming a connected peer, even in a
descriptor carrying the sender's own `agent_name`, drops that peer's
descriptors: its pending transfers never complete, `connected()` stays True,
and `close` times out with memory still registered. An endpoint serving
several peers must therefore trust every peer's descriptor metadata; checking
`agent_name` does not prevent this. Peer state grows with each connected peer
and is released only by `close`, which waits for pending transfers to all
peers. The transport does not tie a descriptor to the peer that sent it, so
such an endpoint must also authorize each request, for example by assigning
each client an `agent_name` and checking that the request's
`remote_buffer.agent_name` matches.

`unregister_memory(memory)` deregisters a local allocation without closing the
transport. Wait for local transfers first and coordinate with peers to stop remote
access; the backend cannot detect incoming DMA. All handles sharing a registration,
including previously created views, exported descriptors, and their slices, become invalid.
Register again and exchange fresh descriptors before resuming transfers.

NIXL transfers return Work objects that retain their request handles until
completion. `wait_all`, `read_async`, and `write_async` await Work futures.
The NIXL adapter resolves those futures by checking native transfer status.
Each live transfer owns a distinct request handle.
Independent requests may overlap; explicitly wait before issuing dependent or
overlapping reads/writes. Completion ordering is not implicit.

NIXL's default wait timeout is 30 seconds. `timeout` is in seconds; `None`
selects the backend default and zero polls without waiting. `Work.wait` instead
takes a `datetime.timedelta`; its zero default selects the transfer's timeout.
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
