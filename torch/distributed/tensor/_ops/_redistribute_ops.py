import torch

# Import _redistribute first to ensure the operator is defined before registering prop rule
import torch.distributed.tensor._redistribute  # noqa: F401
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.registration import register_prop_rule
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard


torch._library.opaque_object.register_opaque_type(DeviceMesh, typ="value")
torch._library.opaque_object.register_opaque_type(DTensorSpec, typ="value")
torch._library.opaque_object.register_opaque_type(Shard, typ="value")
torch._library.opaque_object.register_opaque_type(Replicate, typ="value")
torch._library.opaque_object.register_opaque_type(Partial, typ="value")
torch._library.opaque_object.register_opaque_type(TensorMeta, typ="value")
torch._library.opaque_object.register_opaque_type(ShardOrderEntry, typ="value")


lib = torch.library.Library("dtensor", "FRAGMENT")
torch.library.define(
    "dtensor::_redistribute",
    f"(Tensor input, {torch._library.opaque_object.get_opaque_type_name(DTensorSpec)} current_spec, {torch._library.opaque_object.get_opaque_type_name(DTensorSpec)} target_spec, *, bool async_op=False, bool is_backward=False, bool? use_graph_based_transformation=None) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
    lib=lib,
)


@torch.library.impl("dtensor::_redistribute", "CompositeExplicitAutograd", lib=lib)
def _redistribute(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
    use_graph_based_transform: bool | None = None,
) -> torch.Tensor:
    from torch.distributed.tensor._redistribute import redistribute_local_tensor

    res = redistribute_local_tensor(
        local_tensor,
        current_spec,
        target_spec,
        async_op=async_op,
        is_backward=is_backward,
        use_graph_based_transform=use_graph_based_transform,
    )
    return res


@register_prop_rule(torch.ops.dtensor._redistribute.default)
def _redistribute_prop_rule(op_schema: OpSchema) -> OutputSharding:
    input_spec, current_spec, target_spec = op_schema.args_schema[:3]
    return OutputSharding(output_spec=target_spec)
