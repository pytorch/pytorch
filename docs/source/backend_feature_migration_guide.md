# BackendFeature — Backend Capability Registration

## Background

PyTorch has many "does this backend support X" checks that, lacking a first-class capability interface, were written as hardcoded device-type lists:

```python
# torch/_inductor/utils.py
GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]

# torch/_inductor/codegen/common.py
device_type in ["cuda", "xpu"] and ... == "triton"  # online-softmax
```

This binds capability to backend identity: an out-of-tree (OOT) or PrivateUse1 backend that implements a capability cannot advertise it without monkey-patching upstream symbols. The same capability ends up duplicated across multiple hand-maintained device-name lists.

`BackendFeature` (``torch._dynamo.device_interface``) replaces these scattered lists with a unified capability enum. A backend overrides ``DeviceInterface.backend_features(device)`` to return the set of members it supports, and framework checks derive their answers from that set.

## Member Tiers

Members are split into two tiers, marked by comments in the enum definition:

| Tier | Members | Consumers |
|---|---|---|
| **Inductor codegen** | ``FOREACH``, ``BUCKETIZE``, ``SCAN``, ``SORT``, ``TRITON_TEMPLATES``, … | ``has_backend_feature`` / ``V.graph.has_feature`` |
| **Framework-level** | ``GPU``, ``ONLINE_SOFTMAX``, … | eager, dispatcher, downstream libraries via ``DeviceInterface.backend_features(device)`` |

The two tiers share the same enum. Boolean capabilities ship in v1; parameterized negotiation (``supports(op, …)``) is deferred to a future RFC.

## Integration

### 1. Override ``backend_features`` on your ``DeviceInterface``

```python
from torch._dynamo.device_interface import BackendFeature, DeviceInterface

class MyDeviceInterface(DeviceInterface):
    @staticmethod
    def backend_features(device=None):
        return {BackendFeature.GPU, BackendFeature.FOREACH}
```

- Return a **fresh ``set``** on every call — callers may mutate it.
- Only advertise members your backend actually supports.
- Members not advertised are assumed unsupported (conservative default).
- The ``device`` parameter is reserved for future per-device differentiation; current implementations return the same set regardless of device.

### 2. Verify

Once your interface is registered:

- ``GPU_TYPES`` (``torch/_inductor/utils.py``) will include your device type automatically, derived from ``BackendFeature.GPU in iface.backend_features(None)``.
- Inductor codegen queries (``has_backend_feature`` / ``V.graph.has_feature``) see the inductor-codegen-tier members you advertise.
- Migration points like online-softmax (``torch/_inductor/fx_passes/post_grad.py``) check ``ONLINE_SOFTMAX in iface.backend_features(device_type)``.

### 3. Governance

- Adding a new framework-level member → PR against PyTorch upstream.
- Semantic change (rename or meaning change) to an existing member → RFC.
- OOT backends should only advertise members that exist in the upstream enum.
