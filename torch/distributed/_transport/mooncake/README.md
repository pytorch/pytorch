# Mooncake transport

`MooncakeTransport` implements the experimental one-sided transport interface
using Mooncake Transfer Engine directly. It is registered as `"mooncake"` alongside
`"nixl"`; importing it does not import NIXL or initialize a process group.

## Native prerequisite

This implementation requires a Mooncake build with managed
`TransferEngine.submit_transfer`, `get_endpoint`, and `refresh_remote` bindings.
Stock Mooncake wheels are not sufficient unless they expose these methods.
`MooncakeTransport.supported()` checks for the managed submission binding, and
construction reports a missing binding explicitly.

The companion native patch also fixes explicit TCP protocol selection and uses
nonblocking CUDA streams for TCP staging copies. The latter is required to avoid
a default-stream dependency cycle during CUDA graph replay. The draft PR contains
the exact companion patch and its Mooncake base revision for reproduction; it is
an external prerequisite, not a change to PyTorch's native libraries.

## Usage

```python
from torch.distributed._transport import new_transport
from torch.distributed._transport.mooncake import MooncakeRemoteBuffer

transport = new_transport(
    "mooncake", "cuda:0", hostname="192.0.2.10", protocol="rdma"
)
# Exchange these bytes through an application-owned, trusted control plane.
connection_bytes = transport.bind()
transport.connect(peer_connection_bytes)
memory = transport.register_memory(tensor)
descriptor_bytes = memory.to_remote_buffer().serialize()
remote = MooncakeRemoteBuffer.deserialize(peer_descriptor_bytes)
work = transport.write(memory.to_view(), remote, async_op=True)
work.wait()
# After coordinating with every peer to stop remote accesses:
transport.close()
```

Set `hostname` to an address reachable by peers. The default is loopback for local
use. `protocol="tcp"` selects TCP; `protocol="rdma"` selects RDMA. The native engine
uses `P2PHANDSHAKE` metadata. Multiple peers can connect to one endpoint, and
reconnecting with identical connection bytes is a no-op.

Contiguous, nonempty CPU and CUDA tensors can be registered. Local views and
remote descriptors support byte ranges through `slice(offset, length)` and
`[start:stop]`. Exact aliases share a registration lifetime; overlapping,
nonidentical registrations are rejected. Registration changes are refreshed per
peer generation before submission.

Async work holds its buffers until native completion. A wait timeout or coroutine
cancellation does not cancel DMA or permit early buffer reuse. Drain the work or
retry `close()` successfully before releasing memory. Failed status queries retain
resources conservatively. Native calls themselves are synchronous and cannot be
interrupted by the Python timeout.

`read_stream` and `write_stream` use the shared PyTorch CUDA stream ordering
implementation, including graph capture and replay. Warm native connections
before capture. Captured registrations remain retained until close; coordinate
with peers and finish all replays before closing. Stream callback failures follow
the base transport's fatal-error policy.

## Tests

With a built PyTorch environment and the patched Mooncake native extension:

```bash
python test/distributed/test_transport_mooncake.py -v
MOONCAKE_TEST_TORCH_TRANSPORT=1 MOONCAKE_TEST_PROTOCOL=tcp \
  python test/distributed/test_transport_mooncake_native.py -v
MOONCAKE_TEST_TORCH_TRANSPORT=1 MOONCAKE_TEST_PROTOCOL=rdma \
  python test/distributed/test_transport_mooncake_native.py -v
```

The contract suite uses real CPU tensors and a controlled native boundary. The
opt-in native suite requires two CUDA devices for its CUDA cases. It checks CPU
and CUDA reads/writes, registration refresh, multiple peers, CUDA stream ordering,
and three graph replays in each direction. Its 131,076-byte payload crosses the
TCP staging chunk boundary. No cross-host or performance claim is made by these
same-host correctness tests.
